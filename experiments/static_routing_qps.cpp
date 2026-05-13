#include <iostream>
#include <algorithm>
#include <fstream>

#include <mpi.h>
#include <pthread.h>
#include <sched.h>

#include "index.h"
#include "thread_pool.h"

#include <omp.h>

#define num_runs 3
#define ef_search 200

int main(int argc, char **argv) {
    int node, world_size;

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <num_partitions> <k> <output_file> <num_coord_threads> <window_size>\n";
        return 1;
    }

    std::string dataset_name  = argv[1];
    int num_partitions        = std::stoi(argv[2]);
    int k                     = std::stoi(argv[3]);
    std::string output_file   = argv[4];
    int num_coord_threads     = std::stoi(argv[5]);
    int window_size           = std::stoi(argv[6]);

    std::string log_id = "partition_quality_" + dataset_name + "_" + std::to_string(num_partitions);
    Log logger(log_id);

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "ERROR: MPI does not support MPI_THREAD_MULTIPLE!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    Communicator comm;

    int num_threads = 32; // executor worker threads

    if (world_size == 1) {
        std::cerr << "Not Implemented: Single Node SURGE";
        return 1;
    }

    if (world_size != num_partitions + 1) {
        std::cerr << "ERROR: number of processes does not match number of partitions in the Config\n";
        return 1;
    }

    int nvectors, dim;

    if (node == 0) { // Coordinator

        std::cout << "[Coordinator] num_coord_threads=" << num_coord_threads
                  << " window_size=" << window_size << "\n";

        std::vector<RoutingMode> modes = {
            RoutingMode::BranchingFactor,
            RoutingMode::NProbe,
            RoutingMode::RecallTarget
        };
        std::vector<int>   branching_factor_params = {1, 2, 5, 10, 15, 20, 25, 30};
        std::vector<int>   nprobe_params           = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<float> target_params           = {.6, .7, .75, .8, .85, .9, .95, .97, .98, .99};

        FileFormat format = getFileFormat(DATASETS[dataset_name]["base_file"]);
        std::pair<int,int> data_info = get_dataset_info(DATASETS[dataset_name]["base_file"]);
        nvectors = data_info.first;
        dim = data_info.second;

        comm.broadcast_dataset_info(nvectors, dim, world_size);

        Coordinator metaIndex(dim, &comm, &logger);

        std::string meta_dir = dataset_name + "_" + std::to_string(num_partitions);
        metaIndex.load(meta_dir, ef_search);

        std::string query_file = DATASETS[dataset_name]["query_file"];
        std::string gt_file    = DATASETS[dataset_name]["gt_file"];

        std::vector<float> queries = readVecs(query_file, dim);
        std::vector<std::vector<int>> ground_truth_idx = readGT(gt_file, format);

        std::cout << "[Coordinator] Read query and ground truth files\n";

        size_t num_queries  = queries.size() / dim;
        size_t gt_per_query = ground_truth_idx[0].size();
        std::cout << "  Num top vectors per query: " << gt_per_query << "\n";

        comm.broadcast_ef_search(ef_search, world_size);

        std::ofstream outfile(output_file, std::ios::out);
        if (!outfile.is_open()) {
            std::cerr << "ERROR: could not open output file: " << output_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        outfile << "mode,param,recall,qps\n";
        outfile.flush();

        auto mode_to_string = [](RoutingMode m) -> std::string {
            switch (m) {
                case RoutingMode::BranchingFactor: return "BranchingFactor";
                case RoutingMode::NProbe:          return "NProbe";
                case RoutingMode::RecallTarget:    return "RecallTarget";
            }
            return "Unknown";
        };

        // C++17 semaphore: single OMP thread dispatches tasks into the pool,
        // blocking when window_size queries are already in-flight.
        // Workers call release() on completion, unblocking the dispatcher.
        struct CountingSemaphore {
            explicit CountingSemaphore(int count) : count_(count) {}
            void acquire() {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]{ return count_ > 0; });
                --count_;
            }
            void release() {
                std::lock_guard<std::mutex> lock(mutex_);
                ++count_;
                cv_.notify_one();
            }
        private:
            std::mutex mutex_;
            std::condition_variable cv_;
            int count_;
        } sem(window_size);

        const int num_warmup_runs = 3;

        for (RoutingMode mode : modes) {
            std::vector<float> params;
            if (mode == RoutingMode::BranchingFactor) {
                for (int v : branching_factor_params)
                    params.push_back(static_cast<float>(v));
            } else if (mode == RoutingMode::NProbe) {
                for (int v : nprobe_params)
                    params.push_back(static_cast<float>(v));
            } else if (mode == RoutingMode::RecallTarget) {
                for (float v : target_params)
                    params.push_back(v);
            }

            std::string mode_name = mode_to_string(mode);

            for (float param : params) {
                std::cout << "[Coordinator] sweeping mode=" << mode_name
                          << " param=" << param << "\n";

                // Per-query recall stored atomically so tasks can write concurrently
                std::vector<std::atomic<double>> per_query_recall(num_queries);
                for (auto& v : per_query_recall) v.store(0.0);

                // --- Warmup + recall measurement ---
                #pragma omp parallel num_threads(num_coord_threads)
                #pragma omp single
                {
                    for (int r = 0; r < num_warmup_runs; r++) {
                        for (int i = 0; i < (int)num_queries; i++) {
                            sem.acquire();
                            float* query_vector = queries.data() + (i * dim);

                            #pragma omp task firstprivate(i, r, query_vector)
                            {
                                std::vector<int> results = metaIndex.handle_query(
                                    query_vector, i, k, mode, param
                                );
                                sem.release();

                                if (r == 0) {
                                    int hits = 0;
                                    for (int j = 0; j < k; j++)
                                        for (int res_id : results)
                                            if (ground_truth_idx[i][j] == res_id) hits++;
                                    per_query_recall[i].store(
                                        static_cast<double>(hits) / static_cast<double>(k)
                                    );
                                }
                            }
                        }
                    }
                    #pragma omp taskwait
                }

                double recall_sum = 0.0;
                for (auto& v : per_query_recall) recall_sum += v.load();
                double recall = recall_sum / static_cast<double>(num_queries);

                // --- Throughput ---
                double t_start = MPI_Wtime();

                #pragma omp parallel num_threads(num_coord_threads)
                #pragma omp single
                {
                    for (int r = 0; r < num_runs; r++) {
                        for (int i = 0; i < (int)num_queries; i++) {
                            sem.acquire();
                            float* query_vector = queries.data() + (i * dim);

                            #pragma omp task firstprivate(i, query_vector)
                            {
                                metaIndex.handle_query(query_vector, i, k, mode, param);
                                sem.release();
                            }
                        }
                    }
                    #pragma omp taskwait
                }

                double t_end = MPI_Wtime();
                double qps = (static_cast<double>(num_queries) * num_runs) / (t_end - t_start);

                outfile << mode_name << "," << param << ","
                        << recall << "," << qps << "\n";
                outfile.flush();

                std::cout << "  recall@" << k << ": " << recall
                          << ", qps: " << qps << "\n";
            }
        }

        comm.broadcast_termination(world_size, 1);

    } else { // Executor

        std::cout << "[Executor " << node << "] log_id: " << log_id << "\n";
        comm.recv_dataset_info(nvectors, dim);
        std::cout << "[Executor " << node << "] Received dataset info: num vectors = "
                  << nvectors << ", dimension = " << dim << "\n";

        Executor subIndex(node, dim, comm, &logger);

        std::string output_dir = dataset_name + "_" + std::to_string(num_partitions);
        std::string filename_prefix = output_dir + "/executor_" + std::to_string(node)
                                    + "_" + dataset_name + "_" + std::to_string(num_partitions);
        subIndex.load(filename_prefix, ef_search);

        std::atomic<int> num_processed = 0;
        std::queue<MessageHeader> work_queue;
        std::mutex queue_mutex;
        std::condition_variable cv;
        std::atomic<bool> end = false;

        // Single dedicated receiver — no critical section contention
        std::thread receiver([&]() {
            while (true) {
                MessageHeader header;
                comm.recv_header(header, 0);

                if (header.type == END_OF_COMMUNICATION) {
                    end = true;
                    cv.notify_all();
                    break;
                }

                if (header.type == SET_EF_SEARCH) {
                    subIndex.setEfSearch(header.size);
                    continue;
                }

                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    work_queue.push(header);
                }
                cv.notify_one();
            }
        });

        // Worker threads consume from the queue
        #pragma omp parallel num_threads(num_threads - 1)
        {
            while (true) {
                MessageHeader header;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    cv.wait(lock, [&]{ return !work_queue.empty() || end; });
                    if (end && work_queue.empty()) break;
                    header = work_queue.front();
                    work_queue.pop();
                }

                if (header.type == QUERY_BATCH_SEND) {
                    subIndex.batch_search(header.size, k, header.tag);
                    num_processed += header.size;
                } else if (header.type == QUERY_SEND) {
                    subIndex.search(header.size, header.tag);
                    num_processed++;
                }
            }
        }

        receiver.join();

        std::cout << "[Executor " << node << "] processed: " << num_processed << " queries\n";
    }

    MPI_Finalize();
}