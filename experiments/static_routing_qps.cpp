#include <iostream>
#include <algorithm>

#include <mpi.h>
#include "index.h"
#include "thread_pool.h"

#include <omp.h>

#define NUM_COORD_THREADS 175
#define num_runs 3
#define ef_search 200

int main(int argc, char **argv) {
    int node, world_size;

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <dataset> <num_partitions> <k> <output_file>\n";
        return 1;
    }

    std::string dataset_name  = argv[1];
    int num_partitions        = std::stoi(argv[2]);
    int k                     = std::stoi(argv[3]);
    std::string output_file   = argv[4];

    // if (argc != 6) {
    //     std::cerr << "Usage: " << argv[0] << " <dataset> <num_partitions> <mode> <param> <k>\n";
    //     return 1;
    // }

    // std::string dataset_name = argv[1];
    // int num_partitions = std::stoi(argv[2]);
    // std::string mode_str = argv[3];
    // float param = std::stof(argv[4]);
    // int k = std::stoi(argv[5]);

    // RoutingMode mode;
    // if (mode_str == "branching") mode = RoutingMode::BranchingFactor;
    // else if (mode_str == "nprobe") mode = RoutingMode::NProbe;
    // else if (mode_str == "recall") mode = RoutingMode::RecallTarget;
    // else {
    //     std::cerr << "Invalid Routing Mode: " << mode_str << "\n";
    //     return 1;
    // }

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

    int num_threads = 32; //TODO: hard coded

    // std::cout << "RUN: Hello from #" << node << std::endl;

    if (world_size == 1) {
        // TODO
        std::cerr << "Not Implemented: Single Node SURGE";
        return 1;
    }

    if (world_size != num_partitions + 1) {
        std::cerr << "ERROR: number of processes does not match number of partitions in the Config\n";
        return 1;
    }

    int nvectors, dim;

    if (node == 0) { // Coordinator
        std::vector<RoutingMode> modes = {RoutingMode::BranchingFactor, RoutingMode::NProbe, RoutingMode::RecallTarget};
        std::vector<int> branching_factor_params = {1, 2, 5, 10, 15, 20, 25, 30};
        std::vector<int> nprobe_params = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<float> target_params = {.6, .7, .75,.8, .85, .9, .95, .97, .98, .99};

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
        size_t num_neighbors = k;
        std::cout << "  Num top vectors per query: " << gt_per_query << "\n";

        size_t bf = 20;
        double total_recall = 0.0;

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

                // --- Warmup + recall measurement (recall taken from run 0) ---
                double recall_sum = 0.0;
                #pragma omp parallel for num_threads(NUM_COORD_THREADS) collapse(2) reduction(+:recall_sum)
                for (int r = 0; r < num_warmup_runs; r++) {
                    for (int i = 0; i < (int)num_queries; i++) {
                        float* query_vector = queries.data() + (i * dim);

                        std::vector<int> results = metaIndex.handle_query(
                            query_vector,
                            i,
                            k,
                            mode,
                            param
                        );

                        if (r == 0) {
                            int hits = 0;
                            for (int j = 0; j < k; j++) {
                                int gt_id = ground_truth_idx[i][j];
                                for (int res_id : results) {
                                    if (gt_id == res_id) hits++;
                                }
                            }
                            recall_sum += static_cast<double>(hits) / static_cast<double>(k);
                        }
                    }
                }
                double recall = recall_sum / static_cast<double>(num_queries);

                // --- Throughput ---
                double t_start = MPI_Wtime();
                #pragma omp parallel for num_threads(NUM_COORD_THREADS) collapse(2)
                for (int r = 0; r < num_runs; r++) {
                    for (int i = 0; i < (int)num_queries; i++) {
                        float* query_vector = queries.data() + (i * dim);
                        metaIndex.handle_query(
                            query_vector,
                            i,
                            k,
                            mode,
                            param
                        );
                    }
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


        // std::vector<int> branching_factors = {1, 2, 10, 15, 20, 30};
        // std::vector<float> recall_targets = {0.7, 0.8, 0.9, 0.95, 0.99};
        // std::vector<int> nprobe_values = {1, 2, 4, 8, 16, 32, 64, 128, 256};

        // // std::vector<int> ef_search_values = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 220, 250, 280, 300, 320, 350, 380, 400, 500};
        // // std::vector<int> ef_search_values = {200};
        // comm.broadcast_ef_search(ef_search, world_size);

        // std::string run_path = logger.log_dir +  "/sweep_results_all_partitions.txt";
        // std::ofstream outfile(run_path, std::ios::out | std::ios::app);

        // for (int bf : branching_factors) {
        //     std::cout << "[Coordinator] searching with bf = " << bf << "\n";
        //     // measure recall
        //     std::vector<int> access_rate(num_queries);
        //     std::vector<std::atomic<int>> executor_hits(num_partitions);
        //     for (auto& e : executor_hits) {
        //         e.store(0);
        //     }

        //     std::atomic<size_t> correct = 0;
        //     std::atomic<size_t> completed = 0;

        //     double recalls_total = 0.0f;
        //     size_t num_query_threads = NUM_COORD_THREADS;
        //     // if (bf < 20) {
        //     //     num_query_threads = 600;
        //     // }

        //     auto start = MPI_Wtime();
        //     std::vector<std::vector<int>> per_query_results = metaIndex.handle_queries(queries, num_neighbors, bf);
        //     auto end = MPI_Wtime();
        //     std::cout << "Time to handle all queries: " << (end - start) << " seconds\n";
        //     std::cout << "Throughput: " << (num_queries / (end - start)) << " qps\n";

        //     for (int i = 0; i < num_queries; i++) {
        //         std::vector<int>& results = per_query_results[i];
        //         std::sort(results.begin(), results.end());
        //         auto last = std::unique(results.begin(), results.end());
        //         results.erase(last, results.end());

        //         int relevant = 0;
        //         for (int id : results) {
        //             for (int j = 0; j < k; ++j) {
        //                 if (id == ground_truth_idx[i][j]) {
        //                     relevant++;
        //                 }
        //             }
        //         }

        //         double recall_i = (double)relevant /
        //                         (double)k;
        //         recalls_total += recall_i;

        //         completed++;
        //     }

        //     double recall = recalls_total / num_queries;

        //     // measure throughput and latency: TODO
        //     std::vector<double> latencies(num_queries);

        //     double throughput_start = MPI_Wtime();
        //     #pragma omp parallel for num_threads(num_query_threads)
        //     for (int r = 0; r < num_runs; r++) {
        //         std::vector<std::vector<int>> per_query_results = metaIndex.handle_queries(queries, num_neighbors, bf);
        //     }
        //     double throughput_end = MPI_Wtime();

        //     double qps = (num_queries * num_runs) /  (throughput_end - throughput_start);

        //     outfile << bf << "," << recall << "," << qps << "\n";

        //     std::cout << "  bf: " << bf << ", recall: " << recall << ", qps: " << qps << "\n";
            
        //     std::string output_id = "bf" + std::to_string(bf) + "_ef" + std::to_string(ef_search) + "_" + logger.run_id;
        //     logger.logQueryLatencies(latencies, output_id);
        //     logger.logExecutorHits(executor_hits, output_id);
        //     logger.logAccessRate(access_rate, output_id);
        // }


        // end communication from Coordinator side
        comm.broadcast_termination(world_size, num_threads);

    } else { // Executor
        // std::string log_id;
        // comm.recv_log_id(log_id);
        
        // Config config(argv[1], false, log_id);

        // std::cout << "[Executor " << node << "] log_id: " << config.log_id << "\n";
        std::cout << "[Executor " << node << "] log_id: " << log_id << "\n";
        comm.recv_dataset_info(nvectors, dim);
        std::cout << "[Executor " << node << "] Received dataset info: num vectors = " << nvectors << ", dimension = " << dim << "\n";

        Executor subIndex(node, dim, comm, &logger);

        std::string output_dir = dataset_name + "_" + std::to_string(num_partitions);
        std::string filename_prefix = output_dir + "/executor_" + std::to_string(node) + "_" + dataset_name + "_" + std::to_string(num_partitions);
        subIndex.load(filename_prefix, ef_search);

        std::atomic<int> num_processed = 0;
        std::atomic<bool> end = false;
        omp_set_num_threads(num_threads);
        #pragma omp parallel
        {
            while (!end) {
                MessageHeader header;
                MPI_Status status;

                #pragma omp critical
                {
                    comm.recv_header(header, 0);
                    if (header.type == END_OF_COMMUNICATION) {
                        end = true;
                    }
                }

                if (end) break;
    
                if (header.type == SET_EF_SEARCH) {
                    subIndex.setEfSearch(header.size);
                    continue;
                }

                if (header.type == QUERY_BATCH_SEND) {
                    subIndex.batch_search(header.size, k, header.tag);
                    num_processed+= header.size;
                }

                if (header.type == QUERY_SEND) {
                    subIndex.search(header.size, header.tag);
                    num_processed++;
                }
            
            }
        }

        std::cout << "[Executor " << node << "] proccessed: " << num_processed << " queries\n";
    }

    MPI_Finalize();
}