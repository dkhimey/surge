#pragma once
#include <mpi.h>
#include "utils.h"
#include "hnswlib.h"

// inline unsigned int make_tag(MessageType type, int id) {
//     return (type << 27) | (id & 0x07FFFFFF); // 27 bits for id, 5 bits for type
// }

inline unsigned int make_tag(MessageType type, unsigned int id) {
    return (type << 20) | (id & 0x000FFFFF); // type=5 bits, id=20 bits (0–1,048,575)
}

class Communicator {
    int rank;
    std::vector<MPI_Request> requests;
public:
    Communicator() { 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    }

    void broadcast_log_id(const std::string& log_id, int world_size) {
        for (int i = 1; i < world_size; i++) {
            MPI_Send(log_id.c_str(), log_id.size()+1, MPI_CHAR, i, LOG_SEND, MPI_COMM_WORLD);
        }
    }

    void broadcast_ef_search(int ef_search, int world_size) {
        MessageHeader header;
        header.type = SET_EF_SEARCH;
        header.size = ef_search;

        for (int i = 1; i < world_size; i++) {
            MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }
    }

    void recv_log_id(std::string& log_id) {
        char buffer[32];
        MPI_Recv(buffer, 32, MPI_CHAR, 0, LOG_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        log_id = std::string(buffer);
    }

    void send_header(const MessageHeader& header, int dest) {
        MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, dest, 0, MPI_COMM_WORLD);
    }

    void send_vector(const float* vector, size_t dim, int dest, int tag) {
        MPI_Send(vector, dim, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
    }

    void send_vector_batch(const float* vectors, size_t num_vectors, size_t dim, int dest, int tag) {
        MPI_Send(vectors, num_vectors * dim, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
    }

    void recv_vector_batch(float* vectors, size_t num_vectors, size_t dim, int source, int tag) {
        MPI_Recv(vectors, num_vectors * dim, MPI_FLOAT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    void send_result_batch(const int* results, size_t num_queries, size_t num_neighbors, int tag) {
        MPI_Send(results, num_queries * num_neighbors, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    void send_result(const int* results, size_t num_neighbors, int tag) {
        MPI_Send(results, num_neighbors, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    void recv_result(int* results, size_t num_neighbors, int source, int tag) {
        MPI_Recv(results, num_neighbors, MPI_INT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    void recv_result_batch(int* results, size_t total_neighbors, int source, int tag) {
        MPI_Recv(results, total_neighbors, MPI_INT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    void send_ack(MessageType type, int dest, int tag) {
        MessageHeader header(type, tag);
        MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, dest, tag, MPI_COMM_WORLD);
    }

    bool recv_ack(MessageType type, int source, int tag) {
        MessageHeader header;
        MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return header.type == type;
    }

    void send_insert(float* insert_vector, size_t dim, int label, int dest, int tag) {
        MPI_Send(insert_vector, dim, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
        MPI_Send(&label, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
    }

    void send_insert_batch(float* insert_vectors, int* labels, size_t num_vecs, size_t dim, int dest, int tag) {
        MPI_Send(insert_vectors, num_vecs * dim, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
        MPI_Send(labels, num_vecs, MPI_INT, dest, tag, MPI_COMM_WORLD);
    }

    void recv_insert(float* insert_vector, size_t dim, int& label, int tag) {
        MPI_Status status;
        MPI_Recv(insert_vector, dim, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&label, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
    }

    void send_delete(int label, int dest, int tag) {
        MessageHeader header;
        header.type = DELETE_SEND;
        header.tag = tag;
        header.size = label;
        MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, dest, 0, MPI_COMM_WORLD);
    }

    void send_delete_batch(int* labels, size_t num_labels, int dest, int tag) {
        MPI_Send(labels, num_labels, MPI_INT, dest, tag, MPI_COMM_WORLD);
    }

    void recv_delete_batch_results(float* vecs, size_t dim, int source, int tag) {
        MPI_Status status;
        MPI_Recv(vecs, dim, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
    }

    void recv_delete(float* vec, size_t dim, int source, int tag) {
        MPI_Status status;
        MPI_Recv(vec, dim, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
    }

    void send_termination(int dest, int num_threads) {
        MessageHeader end_header(END_OF_COMMUNICATION, 0);
        for (int j = 0; j < num_threads; j++) {
            MPI_Send(&end_header, sizeof(MessageHeader), MPI_BYTE, dest, 0, MPI_COMM_WORLD);
        }
    }

    void broadcast_termination(int world_size, int num_threads = 1) {
        for (int i = 1; i < world_size; i++) {
            send_termination(i, num_threads);
        }
    }

    void recv_header(MessageHeader& header, int source) {
        MPI_Status status;
        MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, source, 0, MPI_COMM_WORLD, &status);
    }

    void recv_query(float* query, size_t dim, int tag) {
        MPI_Status status;
        MPI_Recv(query, dim, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
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
                MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // std::cout << "[Executor " << rank << "]--receive_vector_data() 1-- Received vector data for " << to_recv << " vectors\n";

        MPI_Recv(idx_recv_ptr,
                to_recv,
                MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // std::cout << "[Executor " << rank << "]--receive_vector_data() 2-- Received index data for " << to_recv << " vectors\n";
    }

    void broadcast_HNSW(hnswlib::HierarchicalNSW<float>* meta_HNSW, int world_size) {
        meta_HNSW->saveIndex("tmp_hnsw.bin");

        std::ifstream infile("tmp_hnsw.bin", std::ios::binary);
        std::vector<char> buffer((std::istreambuf_iterator<char>(infile)),
                                std::istreambuf_iterator<char>());

        int meta_size = buffer.size();
        for (int i = 1; i < world_size; i++) {
            MPI_Send(buffer.data(), meta_size, MPI_BYTE, i, META_HNSW_SEND, MPI_COMM_WORLD);
        }
    }

    void broadcast_partitions(const std::vector<int>& partitions, int world_size) {
        for (int i = 1; i < world_size; i++) {
            MPI_Send(partitions.data(), partitions.size(), MPI_INT, i, META_PARTITIONS_SEND, MPI_COMM_WORLD);
        }
    }

    void broadcast_dataset_info(int nvectors, int dim, int world_size) {
        for (int i = 1; i < world_size; i++) {
            MPI_Send(&nvectors, 1, MPI_INT, i, DATASET_INFO_SEND, MPI_COMM_WORLD);
            MPI_Send(&dim, 1, MPI_INT, i, DATASET_INFO_SEND, MPI_COMM_WORLD);
        }
    }

    void recv_dataset_info(int& nvectors, int& dim) {
        MPI_Status status;
        MPI_Recv(&nvectors, 1, MPI_INT, 0, DATASET_INFO_SEND, MPI_COMM_WORLD, &status);
        MPI_Recv(&dim, 1, MPI_INT, 0, DATASET_INFO_SEND, MPI_COMM_WORLD, &status);
    }

    void recv_HNSW(
        int meta_size, 
        hnswlib::HierarchicalNSW<float>* metaHNSW, 
        hnswlib::SpaceInterface<float>* space
    ) {
        std::vector<char> buffer(meta_size);
        MPI_Recv(buffer.data(), meta_size, MPI_BYTE, 0, META_HNSW_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::ofstream outfile("tmp_hnsw_received.bin", std::ios::binary);
        outfile.write(buffer.data(), meta_size);
        outfile.close();

        metaHNSW = new hnswlib::HierarchicalNSW<float>(space, "tmp_hnsw_received.bin");
    }

    void recv_partitions(std::vector<int>& partitions, int kmeans_centers) {
        partitions.resize(kmeans_centers);
        MPI_Recv(partitions.data(), kmeans_centers, MPI_INT, 0, META_PARTITIONS_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    void exchange_counts(int world_size, std::vector<int>& num_to_send, std::vector<int>& num_to_recv) {
        std::vector<MPI_Request> requests;
        for (int peer = 1; peer < world_size; ++peer) {
            if (peer == rank) continue;
            // --- RECV: n ---
            requests.emplace_back();
            MPI_Irecv(&num_to_recv[peer], 1, MPI_INT, peer, 0, MPI_COMM_WORLD, &requests.back());

            // --- SEND: n ---
            requests.emplace_back();
            MPI_Isend(&num_to_send[peer], 1, MPI_INT, peer, 0, MPI_COMM_WORLD, &requests.back());
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
                MPI_Irecv(all_recv_vectors.data() + offset[peer] * dim, n_recv * dim, MPI_FLOAT, peer, 1, MPI_COMM_WORLD, &requests.back());
                requests.emplace_back();
                MPI_Irecv(all_recv_labels.data() + offset[peer], n_recv, MPI_INT, peer, 2, MPI_COMM_WORLD, &requests.back());
            }
        }

        // Post sends for vectors and labels
        for (int peer = 1; peer < world_size; ++peer) {
            if (peer == rank) continue;
            int n_send = num_to_send[peer];
            if (n_send > 0) {
                requests.emplace_back();
                MPI_Isend(partition_vectors[peer].data(), n_send * dim, MPI_FLOAT, peer, 1, MPI_COMM_WORLD, &requests.back());
                requests.emplace_back();
                MPI_Isend(partition_labels[peer].data(), n_send, MPI_INT, peer, 2, MPI_COMM_WORLD, &requests.back());
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
};