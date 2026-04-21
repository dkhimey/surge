#include <cmath>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <omp.h>

#include "index.h"

int main(int argc, char **argv) {
    int node, world_size;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset> <num_partitions> <sample_size>\n";
        return 1;
    }

    std::string dataset_name = argv[1];
    int num_partitions = std::stoi(argv[2]);
    int sample_size = std::stoi(argv[3]);

    // add time stamp in year_month_day_log_id format
    auto now = std::time(nullptr);
    auto* tm = std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y%m%d_%H%M%S");
    std::string log_id = "theoretical_partition_quality_" + dataset_name + "_" + std::to_string(num_partitions) + "_" + oss.str();
    std::filesystem::create_directories(log_id);
    Log logger(log_id);

    std::pair<int,int> data_info = get_dataset_info(DATASETS[dataset_name]["base_file"]);
    std::cout << "Dataset has " << data_info.first << " vectors with dimension " << data_info.second << "\n";
    
    int nvectors = data_info.first;
    int dim = data_info.second;
    
    std::vector<float> query_vectors = readVecs(DATASETS[dataset_name]["query_file"], dim);
    int num_queries = query_vectors.size() / dim;

    printf("Max threads: %d\n", omp_get_max_threads());

    // build only meta-HNSW
    std::vector<float> sample = getSample(DATASETS[dataset_name]["base_file"], 
                                              nvectors, dim, sample_size);

    std::cout << "Initialized Sample " << sample_size << " samples\n";

    Coordinator metaIndex(dim, &logger);
    metaIndex.setSampleData(sample.data(), sample_size);

    int ncenters = 10000;
    int ef_construction = 200;
    int M_meta = 16;
    int k = 10;

    std::cout << "Building meta-HNSW index\n";

    metaIndex.build(ncenters, num_partitions, ef_construction, M_meta);
    metaIndex.save(log_id);

    FileFormat format = getFileFormat(DATASETS[dataset_name]["base_file"]);

    std::cout << "Reading ground truth data from " << DATASETS[dataset_name]["gt_file"] << "\n";
    std::vector<std::vector<int>> gt_indices = readGTBin(DATASETS[dataset_name]["gt_file"]);
    std::cout << "Finished reading ground truth data\n";

    std::cout << "Memory mapping dataset file " << DATASETS[dataset_name]["base_file"] << "\n";
    int fd = open(DATASETS[dataset_name]["base_file"].c_str(), O_RDONLY);

    struct stat st;
    fstat(fd, &st);

    void* ptr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "mmap failed: " << strerror(errno) << "\n";
        close(fd);
        return 1;
    }

    std::cout << "Finished memory mapping dataset file\n";

    // compute theoretical recall:
    // 1. check where each of the k gt vectors is assigned
    std::cout << "Computing ground truth partition assignments\n";
    std::vector<std::vector<size_t>> gt_partitions(num_queries);
    bool format_error = false;
    # pragma omp parallel for
    for (int i = 0; i < num_queries; i++) {
        size_t gt_k = std::min(static_cast<size_t>(k), gt_indices[i].size());
        gt_partitions[i].reserve(gt_k);
        for (size_t j = 0; j < gt_k; ++j) {
            int idx = gt_indices[i][j];
            if (idx < 0 || idx >= nvectors) {
                std::cerr << "GT index " << idx << " out of bounds (nvectors=" << nvectors << ")\n";
                continue;
            }
            std::vector<float> vec(dim);
            if (format == U8BIN) {
                uint8_t* data_u8 = reinterpret_cast<uint8_t*>(static_cast<char*>(ptr) + 8);
                for (int d = 0; d < dim; d++) {
                    vec[d] = static_cast<float>(data_u8[static_cast<size_t>(idx) * dim + d]);
                }
            } else if (format == I8BIN) {
                int8_t* data_i8 = reinterpret_cast<int8_t*>(static_cast<char*>(ptr) + 8);
                for (int d = 0; d < dim; d++) {
                    vec[d] = static_cast<float>(data_i8[static_cast<size_t>(idx) * dim + d]);
                }
            } else if (format == FBIN) {
                float* data_f = reinterpret_cast<float*>(static_cast<char*>(ptr) + 8);
                for (int d = 0; d < dim; d++) {
                    vec[d] = data_f[static_cast<size_t>(idx) * dim + d];
                }
            } else {
                std::cerr << "Unsupported format for vector conversion\n";
                #pragma omp critical
                {
                    format_error = true;
                }
            }
            // route gt vec
            size_t gt_partition = metaIndex.getCurrentPartition(vec.data());
            gt_partitions[i].push_back(gt_partition);
        }
    }

    if (format_error) {
        munmap(ptr, st.st_size);
        close(fd);
        return 1;
    }

    std::cout << "Finished computing ground truth partition assignments\n";

    std::ofstream out(log_id + "/routing_metrics.csv");
    out << "mode,param,recall,activation,imbalance,query_time_s\n";

    std::ofstream raw_counts_out(log_id + "/routing_partition_counts.csv");
    raw_counts_out << "mode,param,partition_id,count\n";

    auto write_metrics = [&](const std::string& mode_name, double param, const std::vector<std::vector<size_t>>& query_partitions, double query_time_s) {
        std::vector<long long> per_partition_counts(num_partitions, 0);
        double total_recall = 0.0;
        double total_activation = 0.0;

        for (int i = 0; i < num_queries; i++) {
            const std::vector<size_t>& visited_parts = query_partitions[i];
            for (size_t partition_id : visited_parts) {
                if (partition_id < static_cast<size_t>(num_partitions)) {
                    per_partition_counts[partition_id] += 1;
                }
            }

            total_activation += static_cast<double>(visited_parts.size()) / static_cast<double>(num_partitions);

            int hits = 0;
            for (size_t part : gt_partitions[i]) {
                if (std::find(visited_parts.begin(), visited_parts.end(), part) != visited_parts.end()) {
                    hits += 1;
                }
            }

            if (!gt_partitions[i].empty()) {
                total_recall += static_cast<double>(hits) / static_cast<double>(gt_partitions[i].size());
            }
        }

        double mean = std::accumulate(per_partition_counts.begin(), per_partition_counts.end(), 0.0) /
                      static_cast<double>(per_partition_counts.size());
        double variance = 0.0;
        for (long long count : per_partition_counts) {
            double diff = static_cast<double>(count) - mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(per_partition_counts.size());
        double stddev = std::sqrt(variance);
        double imbalance = mean > 0.0 ? stddev / mean : 0.0;

        double average_recall = total_recall / static_cast<double>(num_queries);
        double average_activation = total_activation / static_cast<double>(num_queries);

        out << mode_name << ','
            << param << ','
            << average_recall << ','
            << average_activation << ','
            << imbalance << ','
            << query_time_s << '\n';

        for (int partition_id = 0; partition_id < num_partitions; partition_id++) {
            raw_counts_out << mode_name << ','
                           << param << ','
                           << partition_id << ','
                           << per_partition_counts[partition_id] << '\n';
        }

        std::cout << mode_name
                  << " param=" << param
                  << " recall=" << average_recall
                  << " activation=" << average_activation
                  << " imbalance=" << imbalance
                  << " time_s=" << query_time_s
                  << "\n";
    };


    // 2. check where query would be routed to
        // routing strategy: branching factor
        // branching factors: iterate
    std::cout << "Computing routing metrics for different strategies\n";
    std::vector<int> branching_factors = {1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50};
    for (int bf: branching_factors) {
           auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<size_t>> query_partitions = metaIndex.route_queries(query_vectors, RoutingMode::BranchingFactor, bf);
           double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
           write_metrics("branching_factor", bf, query_partitions, elapsed);
    }
    std::cout << "Finished computing routing metrics for branching factor strategy\n";

        // routing stategy: recall target
    std::vector<float> recall_targets = {0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99};
    for (float rt: recall_targets) {
           auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<size_t>> query_partitions = metaIndex.route_queries(query_vectors, RoutingMode::RecallTarget, rt);
           double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
           write_metrics("recall_target", rt, query_partitions, elapsed);
    }
    std::cout << "Finished computing routing metrics for recall target strategy\n";

        // routing stratefy: nprobe
    std::vector<int> nprobes = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int np: nprobes) {
           auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<size_t>> query_partitions = metaIndex.route_queries(query_vectors, RoutingMode::NProbe, np);
           double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
           write_metrics("nprobe", np, query_partitions, elapsed);
    }

    munmap(ptr, st.st_size);
    close(fd);
}