#include <iostream>
#include <algorithm>

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

    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <num_partitions> <mode> <param> <k> <num_coord_threads> <window_size>\n";
        return 1;
    }

    std::string dataset_name = argv[1];
    int num_partitions = std::stoi(argv[2]);
    std::string mode_str = argv[3];
    float param = std::stof(argv[4]);
    int k = std::stoi(argv[5]);
    int num_coord_threads = std::stoi(argv[6]);
    int window_size       = std::stoi(argv[7]);

    RoutingMode mode;
    if (mode_str == "branching") mode = RoutingMode::BranchingFactor;
    else if (mode_str == "nprobe") mode = RoutingMode::NProbe;
    else if (mode_str == "recall") mode = RoutingMode::RecallTarget;
    else {
        std::cerr << "Invalid Routing Mode: " << mode_str << "\n";
        return 1;
    }

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

    printf("Max threads: %d\n", omp_get_max_threads());

    if (node == 0) { // Coordinator

        std::cout << "[Coordinator] num_coord_threads=" << num_coord_threads
                  << " window_size=" << window_size << "\n";

        FileFormat format = getFileFormat(DATASETS[dataset_name]["base_file"]);
        std::pair<int,int> data_info = get_dataset_info(DATASETS[dataset_name]["base_file"]);
        nvectors = data_info.first;
        dim = data_info.second;

        comm.broadcast_dataset_info(nvectors, dim, world_size);

        Coordinator metaIndex(dim, &comm, &logger);

        std::string meta_dir = dataset_name + "_" + std::to_string(num_partitions);
        metaIndex.load(meta_dir, ef_search);

        std::string query_file = DATASETS[dataset_name]["query_file"];
        std::string gt_file = DATASETS[dataset_name]["gt_file"];

        std::vector<float> queries = readVecs(query_file, dim);
        std::vector<std::vector<int>> ground_truth_idx = readGT(gt_file, format);

        std::cout << "[Coordinator] Read query and ground truth files\n";

        size_t num_queries = queries.size() / dim;
        size_t gt_per_query = ground_truth_idx[0].size();
        std::cout << "  Num top vectors per query: " << gt_per_query << "\n";

        comm.broadcast_ef_search(ef_search, world_size);

        // C++17 semaphore: single thread dispatches tasks, blocks when WINDOW_SIZE
        // queries are already in-flight. OMP worker threads call release() on
        // completion, unblocking the dispatcher. Threads are created once by OMP
        // and reused — no per-query thread creation overhead.
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

        // ------------------------------------------------------------------
        // Warmup phase: 3 passes, recall measured on first pass
        // ------------------------------------------------------------------
        std::vector<std::atomic<double>> per_query_recall(num_queries);
        for (auto& v : per_query_recall) v.store(0.0);

        #pragma omp parallel num_threads(num_coord_threads)
        #pragma omp single
        {
            for (int r = 0; r < 3; r++) {
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

        double total_recall = 0.0;
        for (auto& v : per_query_recall) total_recall += v.load();

        std::cout << "------WARMED------\n";
        std::cout << "Recall @" << k << ": " << total_recall / num_queries << "\n";

        // ------------------------------------------------------------------
        // Throughput phase: num_runs passes, no recall bookkeeping
        // ------------------------------------------------------------------
        double throughput_start = MPI_Wtime();

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

        double throughput_end = MPI_Wtime();
        std::cout << "QPS: " << num_queries * num_runs / (throughput_end - throughput_start) << "\n";

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
