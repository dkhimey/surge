// shared_static_partitioning.cpp
//
// Shared-filesystem variant of static_partitioning.cpp.
//
// static_partitioning's distribute step funnels the entire dataset through
// rank 0: it alone reads every vector off local disk and pushes it out over
// the network with locked, blocking MPI_Send calls -- for large datasets
// (e.g. bigann-1B, ~512GB of vector data) this makes one node's disk
// bandwidth and one NIC the bottleneck for the whole cluster, while every
// executor's CPU and NIC sit idle waiting for data to arrive.
//
// This version assumes the dataset lives on a filesystem mounted
// identically on every node (NFS, BeeGFS, Lustre, etc). Every rank -
// including the coordinator - reads its own contiguous slice of the base
// file directly, routes those vectors locally against a replicated copy of
// the coordinator's meta-HNSW, and exchanges routed vectors with one
// MPI_Alltoallv collective per batch instead of coordinator-only
// point-to-point sends. Disk I/O, routing CPU work, and network egress are
// all spread across the whole cluster instead of bottlenecking on rank 0.
//
// This mirrors the read-from-shared-storage + Alltoallv pattern already
// used for runtime inserts in old_experiments/shared_batch_experiment.cpp,
// applied here to the initial distribute step as well.
//
// Requires: DATASETS[dataset]["base_file"] resolves to the same file
// content from every rank (shared filesystem mount).
//
// Usage:  mpirun -np <P+1> --rankfile rankfile.txt \
//             ./shared_static_partitioning <dataset> <num_partitions>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include "index.h"

namespace {

constexpr int    NCENTERS        = 10000; // TODO: hard coded, matches static_partitioning.cpp
constexpr int    EF_CONSTRUCTION = 200;   // TODO: hard coded
constexpr int    M_META          = 16;    // TODO: hard coded
constexpr int    M_SUB           = 16;    // TODO: hard coded
constexpr size_t SAMPLE_SIZE     = 100000; // TODO: hard coded

// Generic AllToAllV helper (adapted from old_experiments/shared_batch_experiment.cpp).
// send_bufs[r] holds the payload destined for rank r; recv_bufs[r] is filled
// with whatever this rank received from r. Every rank must call this the
// same number of times, in lockstep, even with empty buffers.
template <typename T>
void all_to_all_v(const std::vector<std::vector<T>>& send_bufs,
                   std::vector<std::vector<T>>& recv_bufs,
                   MPI_Datatype dtype, int world_size, MPI_Comm comm) {
    std::vector<int> sc(world_size), sd(world_size, 0);
    for (int r = 0; r < world_size; ++r) sc[r] = static_cast<int>(send_bufs[r].size());
    for (int r = 1; r < world_size; ++r) sd[r] = sd[r - 1] + sc[r - 1];

    std::vector<T> sf;
    { size_t tot = 0; for (auto& b : send_bufs) tot += b.size(); sf.reserve(tot); }
    for (int r = 0; r < world_size; ++r)
        sf.insert(sf.end(), send_bufs[r].begin(), send_bufs[r].end());

    std::vector<int> rc(world_size, 0), rd(world_size, 0);
    MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, comm);
    for (int r = 1; r < world_size; ++r) rd[r] = rd[r - 1] + rc[r - 1];
    int total_recv = rd[world_size - 1] + rc[world_size - 1];

    std::vector<T> rf(total_recv);
    MPI_Alltoallv(sf.data(), sc.data(), sd.data(), dtype,
                  rf.data(), rc.data(), rd.data(), dtype, comm);

    recv_bufs.assign(world_size, {});
    for (int r = 0; r < world_size; ++r)
        recv_bufs[r].assign(rf.begin() + rd[r], rf.begin() + rd[r] + rc[r]);
}

// Replicate the coordinator's meta-HNSW + partition map onto every rank so
// each one can route vectors locally (same logic as
// Coordinator::get_partitions_for_insert_, just replicated instead of RPC'd).
// Called by ALL ranks simultaneously; rank 0 sends, ranks 1..P receive.
void bcast_routing_state(int rank,
                          hnswlib::HierarchicalNSW<float>* coord_meta,   // non-null on rank 0
                          const std::vector<int>* coord_parts,          // non-null on rank 0
                          hnswlib::HierarchicalNSW<float>*& out_meta,   // set on rank > 0
                          std::vector<int>& out_parts,
                          hnswlib::SpaceInterface<float>* space) {
    int n = (rank == 0) ? static_cast<int>(coord_parts->size()) : 0;
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) out_parts.resize(n);
    MPI_Bcast(rank == 0 ? const_cast<int*>(coord_parts->data()) : out_parts.data(),
              n, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) out_parts = *coord_parts;

    int hnsw_size = 0;
    std::vector<char> buf;
    if (rank == 0) {
        coord_meta->saveIndex("tmp_shared_static_route_bcast.bin");
        std::ifstream f("tmp_shared_static_route_bcast.bin", std::ios::binary);
        buf.assign(std::istreambuf_iterator<char>(f), {});
        hnsw_size = static_cast<int>(buf.size());
    }
    MPI_Bcast(&hnsw_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) buf.resize(hnsw_size);
    MPI_Bcast(buf.data(), hnsw_size, MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        const std::string tmp = "tmp_shared_static_route_r" + std::to_string(rank) + ".bin";
        { std::ofstream f(tmp, std::ios::binary); f.write(buf.data(), hnsw_size); }
        out_meta = new hnswlib::HierarchicalNSW<float>(space, tmp);
        out_meta->setEf(EF_CONSTRUCTION);
        std::remove(tmp.c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset> <num_partitions>\n";
        return 1;
    }

    std::string dataset_name   = argv[1];
    int         num_partitions = std::stoi(argv[2]);

    std::string log_id  = "shared_partition_quality_" + dataset_name + "_" + std::to_string(num_partitions);
    std::string log_dir = ensure_log_dir(log_id);

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    install_mpi_terminate_handler();

    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "Error: MPI does not provide required threading level (MPI_THREAD_MULTIPLE)\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    Communicator comm;

    if (world_size == 1) {
        std::cerr << "Not Implemented: Single Node SURGE\n";
        return 1;
    }
    if (world_size != num_partitions + 1) {
        std::cerr << "ERROR: number of processes (" << world_size
                  << ") should be one more than the number of partitions ("
                  << num_partitions << ")\n";
        return 1;
    }

    const std::string base_file = DATASETS[dataset_name]["base_file"];

    // Every rank reads the (tiny) header itself instead of waiting on a
    // broadcast from rank 0 -- cheap, and avoids a serialization point.
    // Requires base_file to resolve identically on every node.
    auto [nvectors, dim] = get_dataset_info(base_file);
    std::cout << "[Rank " << rank << "] dataset: " << nvectors << " vectors, dim=" << dim
              << ", max threads=" << omp_get_max_threads() << "\n";

    // ── Build the routing layer (coordinator only; cheap: fixed-size sample) ──
    Coordinator metaIndex(dim, &comm);
    hnswlib::L2Space meta_space(dim);
    hnswlib::HierarchicalNSW<float>* local_routing_hnsw = nullptr;
    std::vector<int> routing_partitions;

    double route_build_start = MPI_Wtime();
    if (rank == 0) {
        size_t sample_size = SAMPLE_SIZE;
        std::vector<float> sample = getSample(base_file, nvectors, dim, sample_size);
        metaIndex.set_sample_data(sample.data(), sample_size);
        metaIndex.build(NCENTERS, num_partitions, EF_CONSTRUCTION, M_META);

        local_routing_hnsw = metaIndex.get_meta_hnsw(); // coordinator uses its own graph directly
        routing_partitions = metaIndex.get_partitions();
    }

    bcast_routing_state(rank, rank == 0 ? metaIndex.get_meta_hnsw() : nullptr,
                         &routing_partitions, local_routing_hnsw, routing_partitions, &meta_space);

    double route_build_end = MPI_Wtime();
    if (rank == 0)
        std::cout << "[Coordinator] Meta-index build + broadcast: "
                  << (route_build_end - route_build_start) << "s\n";

    MPI_Barrier(MPI_COMM_WORLD);

    // ── Distribute vectors ────────────────────────────────────────────────
    // Every rank (including the coordinator) reads + routes its own
    // contiguous slice of the base file; a per-batch MPI_Alltoallv scatters
    // routed vectors directly to the owning executor.
    double t0 = MPI_Wtime();

    long long chunk    = (static_cast<long long>(nvectors) + world_size - 1) / world_size;
    long long my_start = std::min<long long>(static_cast<long long>(rank) * chunk, nvectors);
    long long my_end   = std::min<long long>(my_start + chunk, nvectors);

    long long max_chunk = 0;
    MPI_Allreduce(&chunk, &max_chunk, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    long long num_batches = (max_chunk + VECTOR_BATCH_SIZE - 1) / VECTOR_BATCH_SIZE;

    int num_threads = omp_get_max_threads(); // TODO: hard coded, matches static_partitioning.cpp
    omp_set_num_threads(num_threads);

    std::vector<float> local_vectors_accum;
    std::vector<int>   local_indices_accum;
    long long          local_recv_count = 0;

    for (long long b = 0; b < num_batches; ++b) {
        long long batch_start = my_start + b * VECTOR_BATCH_SIZE;
        long long batch_n = std::max<long long>(0, std::min<long long>(VECTOR_BATCH_SIZE, my_end - batch_start));

        std::vector<std::vector<int>>   send_ids(world_size);
        std::vector<std::vector<float>> send_vecs(world_size);

        if (batch_n > 0) {
            std::vector<float> X = readVecs(base_file, dim, static_cast<int>(batch_n), static_cast<int>(batch_start));

            // Thread-local buckets, merged after the parallel region --
            // avoids lock contention on shared per-partition buffers.
            std::vector<std::vector<std::vector<int>>>   thread_ids(num_threads, std::vector<std::vector<int>>(world_size));
            std::vector<std::vector<std::vector<float>>> thread_vecs(num_threads, std::vector<std::vector<float>>(world_size));

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                #pragma omp for schedule(static)
                for (int i = 0; i < static_cast<int>(batch_n); ++i) {
                    long long global_index = batch_start + i;
                    float* vec_ptr = X.data() + static_cast<size_t>(i) * dim;

                    auto pq = local_routing_hnsw->searchKnn(vec_ptr, 1);
                    int center_id = static_cast<int>(pq.top().second);
                    int target    = routing_partitions[center_id] + 1; // executor ranks are 1..P

                    thread_ids[tid][target].push_back(static_cast<int>(global_index));
                    thread_vecs[tid][target].insert(thread_vecs[tid][target].end(), vec_ptr, vec_ptr + dim);
                }
            }

            for (int t = 0; t < num_threads; ++t) {
                for (int p = 0; p < world_size; ++p) {
                    if (thread_ids[t][p].empty()) continue;
                    send_ids[p].insert(send_ids[p].end(), thread_ids[t][p].begin(), thread_ids[t][p].end());
                    send_vecs[p].insert(send_vecs[p].end(), thread_vecs[t][p].begin(), thread_vecs[t][p].end());
                }
            }
        }

        // Collective -- every rank must call this each iteration, even with
        // empty send buffers, or the run deadlocks.
        std::vector<std::vector<int>>   recv_ids;
        std::vector<std::vector<float>> recv_vecs;
        all_to_all_v(send_ids,  recv_ids,  MPI_INT,   world_size, MPI_COMM_WORLD);
        all_to_all_v(send_vecs, recv_vecs, MPI_FLOAT, world_size, MPI_COMM_WORLD);

        if (rank != 0) {
            for (int src = 0; src < world_size; ++src) {
                size_t n = recv_ids[src].size();
                if (n == 0) continue;
                local_indices_accum.insert(local_indices_accum.end(), recv_ids[src].begin(), recv_ids[src].end());
                local_vectors_accum.insert(local_vectors_accum.end(), recv_vecs[src].begin(), recv_vecs[src].end());
                local_recv_count += static_cast<long long>(n);
            }
        }

        if (rank == 0 && (b % 10 == 0 || b == num_batches - 1)) {
            std::cout << "[Coordinator] - batch " << b << "/" << num_batches
                      << " (" << MPI_Wtime() << ")\n";
        }
    }

    double t1 = MPI_Wtime();
    double distribute_time = t1 - t0;
    if (rank == 0) std::cout << "Partition time: " << distribute_time << " seconds\n";

    // ── Gather per-partition counts for logging (rank 0) ────────────────────
    long long my_count_ll = local_recv_count;
    std::vector<long long> all_counts(world_size, 0);
    MPI_Gather(&my_count_ll, 1, MPI_LONG_LONG, all_counts.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<int> counts_per_partition(num_partitions, 0);
        for (int r = 1; r < world_size; ++r)
            counts_per_partition[r - 1] = static_cast<int>(all_counts[r]);

        std::string meta_dir = dataset_name + "_" + std::to_string(num_partitions);
        std::cout << "[Coordinator] Saving to: " << meta_dir << "\n";
        metaIndex.save(meta_dir);

        write_controller_build_json(log_dir + "/controller_build.json", metaIndex.build_metrics(),
                                    distribute_time, counts_per_partition, meta_dir + "/metaHNSW.bin");
        dump_centers(log_dir, metaIndex.centers());
        dump_partitions(log_dir, metaIndex.get_partitions());

        std::remove("tmp_shared_static_route_bcast.bin");
    } else {
        std::cout << "[Executor " << rank << "] Total vectors received: " << local_recv_count << "\n";

        Executor subIndex(rank, dim, comm);
        subIndex.set_data(local_vectors_accum.data(), local_indices_accum.data(),
                          static_cast<size_t>(local_recv_count));

        std::cout << "[Executor " << rank << " ] Building local sub-index\n";
        subIndex.build(
            EF_CONSTRUCTION,
            M_SUB,
            omp_get_max_threads() // num_building_threads, TODO: hard coded
        );

        std::string output_dir = dataset_name + "_" + std::to_string(num_partitions);
        std::filesystem::create_directories(output_dir);

        std::string filename_prefix = output_dir + "/executor_" + std::to_string(rank) + "_" + dataset_name + "_" + std::to_string(num_partitions);
        std::string sub_file = subIndex.save(filename_prefix);
        write_executor_build_json(log_dir + "/executor_" + std::to_string(rank) + "_build.json",
                                  subIndex.build_metrics(), static_cast<size_t>(local_recv_count), sub_file);

        delete local_routing_hnsw; // replicated copy owned by this rank; coordinator's is owned by metaIndex
    }

    MPI_Finalize();
}

// mpirun -np 11 --rankfile rankfile.txt bin/shared_static_partitioning bigann-1B 10
