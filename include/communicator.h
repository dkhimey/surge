#pragma once
#include <mpi.h>
#include "utils.h"
#include "hnswlib.h"

inline unsigned int make_tag(MessageType type, unsigned int id) {
    return (type << 20) | (id & 0x000FFFFF); // type=5 bits, id=20 bits (0–1,048,575)
}

// MPI tags for the peer-to-peer vector exchange during rebuilds.
constexpr int TAG_SEND_NUM    = 100000001;
constexpr int TAG_SEND_VECS   = 100000002;
constexpr int TAG_SEND_LABELS = 100000003;

class Communicator {
    MPI_Comm mpi_comm_;
    int rank;
public:
    Communicator(MPI_Comm comm = MPI_COMM_WORLD) : mpi_comm_(comm) {
        MPI_Comm_rank(mpi_comm_, &rank);
    }

    void broadcast_log_id(const std::string& log_id, int world_size) {
        for (int i = 1; i < world_size; i++) {
            MPI_Send(log_id.c_str(), log_id.size()+1, MPI_CHAR, i, LOG_SEND, mpi_comm_);
        }
    }

    void broadcast_ef_search(int ef_search, int world_size) {
        MessageHeader header;
        header.type = SET_EF_SEARCH;
        header.size = ef_search;

        for (int i = 1; i < world_size; i++) {
            MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, i, 0, mpi_comm_);
        }
    }

    void recv_log_id(std::string& log_id) {
        char buffer[32];
        MPI_Recv(buffer, 32, MPI_CHAR, 0, LOG_SEND, mpi_comm_, MPI_STATUS_IGNORE);
        log_id = std::string(buffer);
    }

    void send_header(const MessageHeader& header, int dest) {
        MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, dest, 0, mpi_comm_);
    }

    void send_vector(const float* vector, size_t dim, int dest, int tag) {
        MPI_Send(vector, dim, MPI_FLOAT, dest, tag, mpi_comm_);
    }

    void send_vector_batch(const float* vectors, size_t num_vectors, size_t dim, int dest, int tag) {
        MPI_Send(vectors, num_vectors * dim, MPI_FLOAT, dest, tag, mpi_comm_);
    }

    void recv_vector_batch(float* vectors, size_t num_vectors, size_t dim, int source, int tag) {
        MPI_Recv(vectors, num_vectors * dim, MPI_FLOAT, source, tag, mpi_comm_, MPI_STATUS_IGNORE);
    }

    void send_result_batch(const int* results, size_t num_queries, size_t num_neighbors, int tag, int dest = 0) {
        MPI_Send(results, num_queries * num_neighbors, MPI_INT, dest, tag, mpi_comm_);
    }

    void send_result(const int* results, size_t num_neighbors, int tag, int dest = 0) {
        MPI_Send(results, num_neighbors, MPI_INT, dest, tag, mpi_comm_);
    }

    void recv_result(int* results, size_t num_neighbors, int source, int tag) {
        MPI_Recv(results, num_neighbors, MPI_INT, source, tag, mpi_comm_, MPI_STATUS_IGNORE);
    }

    void recv_result_batch(int* results, size_t total_neighbors, int source, int tag) {
        MPI_Recv(results, total_neighbors, MPI_INT, source, tag, mpi_comm_, MPI_STATUS_IGNORE);
    }

    void send_ack(MessageType type, int dest, int tag) {
        MessageHeader header(type, tag);
        MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, dest, tag, mpi_comm_);
    }

    bool recv_ack(MessageType type, int source, int tag) {
        MessageHeader header;
        MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, source, tag, mpi_comm_, MPI_STATUS_IGNORE);
        return header.type == type;
    }

    void send_insert(float* insert_vector, size_t dim, int label, int dest, int tag) {
        MPI_Send(insert_vector, dim, MPI_FLOAT, dest, tag, mpi_comm_);
        MPI_Send(&label, 1, MPI_INT, dest, tag, mpi_comm_);
    }

    void send_insert_batch(float* insert_vectors, int* labels, size_t num_vecs, size_t dim, int dest, int tag) {
        MPI_Send(insert_vectors, num_vecs * dim, MPI_FLOAT, dest, tag, mpi_comm_);
        MPI_Send(labels, num_vecs, MPI_INT, dest, tag, mpi_comm_);
    }

    void recv_insert(float* insert_vector, size_t dim, int& label, int tag, int source = 0) {
        MPI_Status status;
        MPI_Recv(insert_vector, dim, MPI_FLOAT, source, tag, mpi_comm_, &status);
        MPI_Recv(&label, 1, MPI_INT, source, tag, mpi_comm_, &status);
    }

    void send_delete(int label, int dest, int tag) {
        MessageHeader header;
        header.type = DELETE_SEND;
        header.tag = tag;
        header.size = label;
        MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, dest, 0, mpi_comm_);
    }

    void send_delete_batch(int* labels, size_t num_labels, int dest, int tag) {
        MPI_Send(labels, num_labels, MPI_INT, dest, tag, mpi_comm_);
    }

    void recv_delete_batch_results(float* vecs, size_t dim, int source, int tag) {
        MPI_Status status;
        MPI_Recv(vecs, dim, MPI_FLOAT, source, tag, mpi_comm_, &status);
    }

    void recv_delete(float* vec, size_t dim, int source, int tag) {
        MPI_Status status;
        MPI_Recv(vec, dim, MPI_FLOAT, source, tag, mpi_comm_, &status);
    }

    void send_termination(int dest, int num_threads) {
        MessageHeader end_header(END_OF_COMMUNICATION, 0);
        for (int j = 0; j < num_threads; j++) {
            MPI_Send(&end_header, sizeof(MessageHeader), MPI_BYTE, dest, 0, mpi_comm_);
        }
    }

    void broadcast_termination(int world_size, int num_threads = 1) {
        for (int i = 1; i < world_size; i++) {
            send_termination(i, num_threads);
        }
    }

    void recv_header(MessageHeader& header, int source) {
        MPI_Status status;
        MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, source, 0, mpi_comm_, &status);
    }

    void recv_query(float* query, size_t dim, int tag, int source = 0) {
        MPI_Status status;
        MPI_Recv(query, dim, MPI_FLOAT, source, tag, mpi_comm_, &status);
    }

    void receive_vector_data(
        float* vec_recv_ptr,
        int* idx_recv_ptr,
        int to_recv,
        size_t dim
    ) {
        // std::cout << "[Executor " << rank << "]--receive_vector_data()-- Expecting to receive data for " << to_recv << " vectors\n";
        MPI_Recv(vec_recv_ptr,
                to_recv * dim,
                MPI_FLOAT, 0, 0, mpi_comm_, MPI_STATUS_IGNORE);

        // std::cout << "[Executor " << rank << "]--receive_vector_data() 1-- Received vector data for " << to_recv << " vectors\n";

        MPI_Recv(idx_recv_ptr,
                to_recv,
                MPI_INT, 0, 0, mpi_comm_, MPI_STATUS_IGNORE);
        // std::cout << "[Executor " << rank << "]--receive_vector_data() 2-- Received index data for " << to_recv << " vectors\n";
    }

    void broadcast_HNSW(hnswlib::HierarchicalNSW<float>* meta_HNSW, int world_size) {
        meta_HNSW->saveIndex("tmp_hnsw.bin");

        std::ifstream infile("tmp_hnsw.bin", std::ios::binary);
        std::vector<char> buffer((std::istreambuf_iterator<char>(infile)),
                                std::istreambuf_iterator<char>());

        int meta_size = buffer.size();
        for (int i = 1; i < world_size; i++) {
            MPI_Send(buffer.data(), meta_size, MPI_BYTE, i, META_HNSW_SEND, mpi_comm_);
        }
    }

    void broadcast_partitions(const std::vector<int>& partitions, int world_size) {
        for (int i = 1; i < world_size; i++) {
            MPI_Send(partitions.data(), partitions.size(), MPI_INT, i, META_PARTITIONS_SEND, mpi_comm_);
        }
    }

    void broadcast_dataset_info(int nvectors, int dim, int world_size) {
        for (int i = 1; i < world_size; i++) {
            MPI_Send(&nvectors, 1, MPI_INT, i, DATASET_INFO_SEND, mpi_comm_);
            MPI_Send(&dim, 1, MPI_INT, i, DATASET_INFO_SEND, mpi_comm_);
        }
    }

    void recv_dataset_info(int& nvectors, int& dim) {
        MPI_Status status;
        MPI_Recv(&nvectors, 1, MPI_INT, 0, DATASET_INFO_SEND, mpi_comm_, &status);
        MPI_Recv(&dim, 1, MPI_INT, 0, DATASET_INFO_SEND, mpi_comm_, &status);
    }

    void recv_HNSW(
        int meta_size, 
        hnswlib::HierarchicalNSW<float>* metaHNSW, 
        hnswlib::SpaceInterface<float>* space
    ) {
        std::vector<char> buffer(meta_size);
        MPI_Recv(buffer.data(), meta_size, MPI_BYTE, 0, META_HNSW_SEND, mpi_comm_, MPI_STATUS_IGNORE);

        // Rank-unique temp path so ranks sharing a working directory don't
        // overwrite each other's received meta-HNSW.
        const std::string recv_path =
            "tmp_hnsw_received_r" + std::to_string(rank) + ".bin";
        std::ofstream outfile(recv_path, std::ios::binary);
        outfile.write(buffer.data(), meta_size);
        outfile.close();

        metaHNSW = new hnswlib::HierarchicalNSW<float>(space, recv_path);
    }

    void recv_partitions(std::vector<int>& partitions, int kmeans_centers) {
        partitions.resize(kmeans_centers);
        MPI_Recv(partitions.data(), kmeans_centers, MPI_INT, 0, META_PARTITIONS_SEND, mpi_comm_, MPI_STATUS_IGNORE);
    }
    
    void exchange_counts(int world_size, std::vector<int>& num_to_send, std::vector<int>& num_to_recv) {
        std::vector<MPI_Request> requests;
        for (int peer = 1; peer < world_size; ++peer) {
            if (peer == rank) continue;
            // --- RECV: n ---
            requests.emplace_back();
            MPI_Irecv(&num_to_recv[peer], 1, MPI_INT, peer, TAG_SEND_NUM, mpi_comm_, &requests.back());

            // --- SEND: n ---
            requests.emplace_back();
            MPI_Isend(&num_to_send[peer], 1, MPI_INT, peer, TAG_SEND_NUM, mpi_comm_, &requests.back());
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    void exchange_vectors(
        int world_size,
        int dim,
        const std::vector<std::vector<float>>& partition_vectors,
        const std::vector<std::vector<int>>& partition_labels,
        std::vector<int>& num_to_send,
        std::vector<int>& num_to_recv,
        std::vector<float>& all_recv_vectors,
        std::vector<int>& all_recv_labels
    ) {
        std::vector<int> offset(world_size, 0);
        for (int i = 1; i < world_size; ++i)
            offset[i] = offset[i - 1] + num_to_recv[i - 1];
        
        std::vector<MPI_Request> requests;

        // Post receives for vectors and labels
        for (int peer = 1; peer < world_size; ++peer) {
            if (peer == rank) continue;
            int n_recv = num_to_recv[peer];
            if (n_recv > 0) {
                requests.emplace_back();
                MPI_Irecv(all_recv_vectors.data() + offset[peer] * dim, n_recv * dim, MPI_FLOAT, peer, TAG_SEND_VECS, mpi_comm_, &requests.back());
                requests.emplace_back();
                MPI_Irecv(all_recv_labels.data() + offset[peer], n_recv, MPI_INT, peer, TAG_SEND_LABELS, mpi_comm_, &requests.back());
            }
        }

        // Post sends for vectors and labels
        for (int peer = 1; peer < world_size; ++peer) {
            if (peer == rank) continue;
            int n_send = num_to_send[peer];
            if (n_send > 0) {
                requests.emplace_back();
                MPI_Isend(partition_vectors[peer].data(), n_send * dim, MPI_FLOAT, peer, TAG_SEND_VECS, mpi_comm_, &requests.back());
                requests.emplace_back();
                MPI_Isend(partition_labels[peer].data(), n_send, MPI_INT, peer, TAG_SEND_LABELS, mpi_comm_, &requests.back());
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Collective / bulk wrappers (Phase 2). Additive: no existing caller relies
    // on these yet; they mirror the raw-MPI patterns currently inlined in the
    // rebuild paths (index.cpp) and the dynamic experiment so those can adopt
    // them without changing wire behavior.
    // ─────────────────────────────────────────────────────────────────────────

    // Point-to-point raw byte transfer (e.g. a serialized meta-HNSW buffer).
    void send_bytes(const std::vector<char>& buf, int dest, int tag) {
        MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_BYTE, dest, tag, mpi_comm_);
    }

    void recv_bytes(std::vector<char>& buf, int count, int source, int tag) {
        buf.resize(count);
        MPI_Recv(buf.data(), count, MPI_BYTE, source, tag, mpi_comm_, MPI_STATUS_IGNORE);
    }

    // MPI_Alltoall of one int per rank: every rank learns how many elements each
    // peer will send it. recv_counts is (re)sized to match send_counts.
    void all_to_all_counts(const std::vector<int>& send_counts, std::vector<int>& recv_counts) {
        recv_counts.assign(send_counts.size(), 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, mpi_comm_);
    }

    // Self-contained all-to-all-v over per-rank buffers: exchanges counts
    // internally, then the payloads. recv_bufs[r] ends up holding what rank r
    // sent us. (Mirrors the experiments' AllToAllV and rebuild_delta rounds 2/3.)
    template<typename T>
    void all_to_all_v(const std::vector<std::vector<T>>& send_bufs,
                      std::vector<std::vector<T>>&       recv_bufs,
                      MPI_Datatype                       dtype,
                      int                                world_size) {
        std::vector<int> sc(world_size), sd(world_size, 0);
        for (int r = 0; r < world_size; ++r) sc[r] = static_cast<int>(send_bufs[r].size());
        for (int r = 1; r < world_size; ++r) sd[r] = sd[r-1] + sc[r-1];

        std::vector<T> sf;
        { size_t tot = 0; for (auto& b : send_bufs) tot += b.size(); sf.reserve(tot); }
        for (int r = 0; r < world_size; ++r)
            sf.insert(sf.end(), send_bufs[r].begin(), send_bufs[r].end());

        std::vector<int> rc(world_size, 0), rd(world_size, 0);
        MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, mpi_comm_);
        for (int r = 1; r < world_size; ++r) rd[r] = rd[r-1] + rc[r-1];
        int total_recv = rd[world_size-1] + rc[world_size-1];

        std::vector<T> rf(total_recv);
        MPI_Alltoallv(sf.data(), sc.data(), sd.data(), dtype,
                      rf.data(), rc.data(), rd.data(), dtype, mpi_comm_);

        recv_bufs.assign(world_size, {});
        for (int r = 0; r < world_size; ++r)
            recv_bufs[r].assign(rf.begin() + rd[r], rf.begin() + rd[r] + rc[r]);
    }

    // Flat all-to-all-v with caller-supplied counts/displacements (for payloads
    // already flattened with per-rank displacements, e.g. the delta rebuild's
    // vector/label rounds). A thin 1:1 wrapper over MPI_Alltoallv on mpi_comm_.
    template<typename T>
    void all_to_all_v_flat(const std::vector<T>& send_buf,
                           const std::vector<int>& send_counts, const std::vector<int>& send_displs,
                           std::vector<T>& recv_buf,
                           const std::vector<int>& recv_counts, const std::vector<int>& recv_displs,
                           MPI_Datatype dtype) {
        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), dtype,
                      recv_buf.data(), recv_counts.data(), recv_displs.data(), dtype, mpi_comm_);
    }

    // Collective broadcasts from `root` (all ranks must call).
    template<typename T>
    void bcast_value(T& v, MPI_Datatype dtype, int root = 0) {
        MPI_Bcast(&v, 1, dtype, root, mpi_comm_);
    }

    // Size-prefixed vector broadcast: non-root ranks are resized to match root.
    template<typename T>
    void bcast_vector(std::vector<T>& buf, MPI_Datatype dtype, int root = 0) {
        int n = (rank == root) ? static_cast<int>(buf.size()) : 0;
        MPI_Bcast(&n, 1, MPI_INT, root, mpi_comm_);
        if (rank != root) buf.resize(n);
        MPI_Bcast(buf.data(), n, dtype, root, mpi_comm_);
    }

    void bcast_bytes(std::vector<char>& buf, int root = 0) {
        bcast_vector<char>(buf, MPI_BYTE, root);
    }

    // Gather one size per rank onto `root` (e.g. per-shard element counts).
    void gather_sizes(unsigned long long value, std::vector<unsigned long long>& out,
                      int world_size, int root = 0) {
        if (rank == root) out.assign(world_size, 0);
        MPI_Gather(&value, 1, MPI_UNSIGNED_LONG_LONG,
                   (rank == root) ? out.data() : nullptr, 1, MPI_UNSIGNED_LONG_LONG,
                   root, mpi_comm_);
    }

    // Contribute-only participant in a gather_sizes (non-root ranks that only
    // send their value and don't collect the result).
    void gather_sizes(unsigned long long value, int root = 0) {
        MPI_Gather(&value, 1, MPI_UNSIGNED_LONG_LONG,
                   nullptr, 1, MPI_UNSIGNED_LONG_LONG, root, mpi_comm_);
    }
};