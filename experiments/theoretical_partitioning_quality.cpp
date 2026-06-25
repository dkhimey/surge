#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <chrono>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <omp.h>

#include "index.h"

// Helper function to save cached GT vectors with metadata
void saveCachedGTVectors(const std::string& filepath, const std::vector<float>& vectors, size_t num_vectors, size_t dim) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << "\n";
        return;
    }
    
    uint32_t num_vecs = static_cast<uint32_t>(num_vectors);
    uint32_t vector_dim = static_cast<uint32_t>(dim);
    
    file.write(reinterpret_cast<const char*>(&num_vecs), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&vector_dim), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(vectors.data()), vectors.size() * sizeof(float));
    file.close();
    
    std::cout << "Saved cached GT vectors to " << filepath << "\n";
}

// Helper function to load cached GT vectors with metadata
bool loadCachedGTVectors(const std::string& filepath, std::vector<float>& vectors, size_t& num_vectors, size_t& dim) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    uint32_t num_vecs, vector_dim;
    file.read(reinterpret_cast<char*>(&num_vecs), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&vector_dim), sizeof(uint32_t));
    
    num_vectors = static_cast<size_t>(num_vecs);
    dim = static_cast<size_t>(vector_dim);
    
    vectors.resize(num_vectors * dim);
    file.read(reinterpret_cast<char*>(vectors.data()), num_vectors * dim * sizeof(float));
    
    if (!file) {
        std::cerr << "Failed to read cached GT vectors from " << filepath << "\n";
        return false;
    }
    
    file.close();
    std::cout << "Loaded cached GT vectors from " << filepath << "\n";
    return true;
}

int main(int argc, char **argv) {
    int node, world_size;

    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <num_partitions> <sample_size> [existing_routing_dir]\n"
                  << "  If existing_routing_dir is given, the routing layer is loaded from\n"
                  << "  those files instead of being built from scratch.\n";
        return 1;
    }

    std::string dataset_name = argv[1];
    int num_partitions = std::stoi(argv[2]);
    int sample_size = std::stoi(argv[3]);
    std::string existing_routing_dir = (argc == 5) ? argv[4] : "";
    const int ef_search = 200;

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

    Coordinator metaIndex(dim, &logger);
    int k = 10;

    if (!existing_routing_dir.empty()) {
        // Start the sweep from an existing routing layer instead of building one.
        // Coordinator::load restores the meta-HNSW, centers, counts, label map and
        // partition assignment written by a previous Coordinator::save.
        std::cout << "Loading existing routing files from " << existing_routing_dir << "\n";
        metaIndex.load(existing_routing_dir, ef_search);

        // The loaded partition assignment is authoritative; override num_partitions
        // so the metrics arrays below are sized to the actual shard count.
        const std::vector<int>& loaded_partitions = metaIndex.getPartitions();
        if (!loaded_partitions.empty()) {
            int loaded_num_partitions =
                *std::max_element(loaded_partitions.begin(), loaded_partitions.end()) + 1;
            if (loaded_num_partitions != num_partitions) {
                std::cout << "Overriding num_partitions " << num_partitions
                          << " -> " << loaded_num_partitions
                          << " from loaded routing state\n";
                num_partitions = loaded_num_partitions;
            }
        }
    } else {
        // build only meta-HNSW
        std::vector<float> sample = getSample(DATASETS[dataset_name]["base_file"],
                                                  nvectors, dim, sample_size);

        std::cout << "Initialized Sample " << sample_size << " samples\n";

        metaIndex.setSampleData(sample.data(), sample_size);

        int ncenters = 10000;
        int ef_construction = 200;
        int M_meta = 16;

        std::cout << "Building meta-HNSW index\n";

        metaIndex.build(ncenters, num_partitions, ef_construction, M_meta);
        metaIndex.save(log_id);
    }

    FileFormat format = getFileFormat(DATASETS[dataset_name]["base_file"]);

    std::cout << "Reading ground truth data from " << DATASETS[dataset_name]["gt_file"] << "\n";
    std::vector<std::vector<int>> gt_indices = readGTBin(DATASETS[dataset_name]["gt_file"]);
    std::cout << "Finished reading ground truth data\n";

    // Check if cached GT vectors exist
    std::string cache_dir = std::filesystem::path(DATASETS[dataset_name]["base_file"]).parent_path().string();
    std::string cached_gt_path = cache_dir + "/cached_gt_vectors_" + dataset_name + ".bin";
    
    std::vector<float> preloaded_vecs;
    std::vector<int> needed_indices;
    
    if (std::filesystem::exists(cached_gt_path)) {
        // Load from cache
        std::cout << "Found cached GT vectors at " << cached_gt_path << "\n";
        size_t cached_num_vectors, cached_dim;
        if (loadCachedGTVectors(cached_gt_path, preloaded_vecs, cached_num_vectors, cached_dim)) {
            std::cout << "Successfully loaded " << cached_num_vectors << " vectors of dimension " << cached_dim << " from cache\n";
            // Recreate needed_indices from gt_indices for the idx_to_buf mapping
            needed_indices.reserve(num_queries * k);
            for (int i = 0; i < num_queries; i++) {
                size_t gt_k = std::min(static_cast<size_t>(k), gt_indices[i].size());
                for (size_t j = 0; j < gt_k; j++) {
                    int idx = gt_indices[i][j];
                    if (idx >= 0 && idx < nvectors)
                        needed_indices.push_back(idx);
                }
            }
            std::sort(needed_indices.begin(), needed_indices.end());
            needed_indices.erase(std::unique(needed_indices.begin(), needed_indices.end()), needed_indices.end());
            // Update DATASETS with cache path if not already set
            if (DATASETS[dataset_name].find("gt_vectors_path") == DATASETS[dataset_name].end()) {
                DATASETS[dataset_name]["gt_vectors_path"] = cached_gt_path;
            }
        } else {
            std::cerr << "Failed to load cached GT vectors, will regenerate\n";
        }
    }
    
    // If cache doesn't exist or failed to load, generate it
    if (preloaded_vecs.empty()) {
        std::cout << "Generating cached GT vectors\n";
        
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

        FileFormat format = getFileFormat(DATASETS[dataset_name]["base_file"]);

        // Collect unique GT indices sorted by value so the preload reads the file
        // in roughly sequential order rather than randomly across 200GB.
        needed_indices.clear();
        needed_indices.reserve(num_queries * k);
        for (int i = 0; i < num_queries; i++) {
            size_t gt_k = std::min(static_cast<size_t>(k), gt_indices[i].size());
            for (size_t j = 0; j < gt_k; j++) {
                int idx = gt_indices[i][j];
                if (idx >= 0 && idx < nvectors)
                    needed_indices.push_back(idx);
            }
        }
        std::sort(needed_indices.begin(), needed_indices.end());
        needed_indices.erase(std::unique(needed_indices.begin(), needed_indices.end()), needed_indices.end());

        std::cout << "Pre-loading " << needed_indices.size() << " unique GT vectors into memory\n";
        preloaded_vecs.resize(needed_indices.size() * dim);

        if (format == U8BIN) {
            uint8_t* data_u8 = reinterpret_cast<uint8_t*>(static_cast<char*>(ptr) + 8);
            for (size_t i = 0; i < needed_indices.size(); i++) {
                int idx = needed_indices[i];
                for (int d = 0; d < dim; d++)
                    preloaded_vecs[i * dim + d] = static_cast<float>(data_u8[static_cast<size_t>(idx) * dim + d]);
            }
        } else if (format == I8BIN) {
            int8_t* data_i8 = reinterpret_cast<int8_t*>(static_cast<char*>(ptr) + 8);
            for (size_t i = 0; i < needed_indices.size(); i++) {
                int idx = needed_indices[i];
                for (int d = 0; d < dim; d++)
                    preloaded_vecs[i * dim + d] = static_cast<float>(data_i8[static_cast<size_t>(idx) * dim + d]);
            }
        } else if (format == FBIN) {
            float* data_f = reinterpret_cast<float*>(static_cast<char*>(ptr) + 8);
            for (size_t i = 0; i < needed_indices.size(); i++) {
                int idx = needed_indices[i];
                std::memcpy(preloaded_vecs.data() + i * dim,
                            data_f + static_cast<size_t>(idx) * dim,
                            dim * sizeof(float));
            }
        } else {
            std::cerr << "Unsupported format for vector conversion\n";
            munmap(ptr, st.st_size);
            close(fd);
            return 1;
        }
        
        // Save to cache
        saveCachedGTVectors(cached_gt_path, preloaded_vecs, needed_indices.size(), dim);
        
        // Update DATASETS with cache path
        DATASETS[dataset_name]["gt_vectors_path"] = cached_gt_path;
        
        munmap(ptr, st.st_size);
        close(fd);
    }

    std::unordered_map<int, size_t> idx_to_buf;
    idx_to_buf.reserve(needed_indices.size());
    for (size_t i = 0; i < needed_indices.size(); i++)
        idx_to_buf[needed_indices[i]] = i;

    std::cout << "Computing ground truth partition assignments\n";
    std::vector<std::vector<size_t>> gt_partitions(num_queries);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_queries; i++) {
        size_t gt_k = std::min(static_cast<size_t>(k), gt_indices[i].size());
        gt_partitions[i].reserve(gt_k);
        for (size_t j = 0; j < gt_k; ++j) {
            int idx = gt_indices[i][j];
            auto it = idx_to_buf.find(idx);
            if (it == idx_to_buf.end()) continue;
            float* vec = preloaded_vecs.data() + it->second * dim;
            gt_partitions[i].push_back(metaIndex.getCurrentPartition(vec));
        }
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

    return 0;
}