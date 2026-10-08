// theoretical_partitioning_quality_from_build.cpp
//
// Variant of theoretical_partitioning_quality that loads an existing routing
// layer (written by Coordinator::save) instead of sampling and building one.
// Sweeps each routing mode and reports theoretical recall, partition activation
// and imbalance against the ground truth. Writes routing_metrics.csv.
//
// Usage:
//   ./bin/theoretical_partitioning_quality_from_build <dataset> <routing_dir> [ef_search]
//
// <routing_dir> must contain the files read by Coordinator::load:
//   metaHNSW.bin, partitions_.bin, centers_pos.bin, centers_counts.bin,
//   labels_to_centers.bin
// ef_search defaults to 200, matching the ef left on the meta-HNSW after build().

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <omp.h>

#include "index.h"

namespace fs = std::filesystem;

static const char* kRoutingFiles[] = {
    "metaHNSW.bin", "partitions_.bin", "centers_pos.bin",
    "centers_counts.bin", "labels_to_centers.bin",
};

static void saveCachedGTVectors(const std::string& filepath, const std::vector<float>& vectors,
                                size_t num_vectors, size_t dim) {
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
    std::cout << "Saved cached GT vectors to " << filepath << "\n";
}

static bool loadCachedGTVectors(const std::string& filepath, std::vector<float>& vectors,
                                size_t& num_vectors, size_t& dim) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;
    uint32_t num_vecs = 0, vector_dim = 0;
    file.read(reinterpret_cast<char*>(&num_vecs), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&vector_dim), sizeof(uint32_t));
    num_vectors = num_vecs;
    dim = vector_dim;
    vectors.resize(num_vectors * dim);
    file.read(reinterpret_cast<char*>(vectors.data()), num_vectors * dim * sizeof(float));
    return static_cast<bool>(file);
}

// Copies the GT vectors (sorted by base index) out of the base file as float32.
static bool gatherGTVectors(const std::string& base_file, const std::vector<int>& needed_indices,
                            int dim, std::vector<float>& out) {
    int fd = open(base_file.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "open failed: " << strerror(errno) << "\n";
        return false;
    }
    struct stat st;
    fstat(fd, &st);
    void* ptr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "mmap failed: " << strerror(errno) << "\n";
        close(fd);
        return false;
    }

    out.resize(needed_indices.size() * dim);
    const char* data = static_cast<const char*>(ptr) + 8;  // skip (n, dim) header
    FileFormat format = getFileFormat(base_file);
    bool ok = true;
    for (size_t i = 0; i < needed_indices.size() && ok; i++) {
        size_t off = static_cast<size_t>(needed_indices[i]) * dim;
        float* dst = out.data() + i * dim;
        switch (format) {
            case U8BIN: {
                const uint8_t* src = reinterpret_cast<const uint8_t*>(data) + off;
                for (int d = 0; d < dim; d++) dst[d] = static_cast<float>(src[d]);
                break;
            }
            case I8BIN: {
                const int8_t* src = reinterpret_cast<const int8_t*>(data) + off;
                for (int d = 0; d < dim; d++) dst[d] = static_cast<float>(src[d]);
                break;
            }
            case FBIN:
                std::memcpy(dst, reinterpret_cast<const float*>(data) + off, dim * sizeof(float));
                break;
            default:
                std::cerr << "Unsupported base format for GT vector gathering\n";
                ok = false;
        }
    }
    munmap(ptr, st.st_size);
    close(fd);
    return ok;
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset> <routing_dir> [ef_search]\n";
        return 1;
    }
    std::string dataset_name = argv[1];
    std::string routing_dir = argv[2];
    int ef_search = (argc == 4) ? std::stoi(argv[3]) : 200;
    const int k = 10;

    if (DATASETS.find(dataset_name) == DATASETS.end()) {
        std::cerr << "Unknown dataset '" << dataset_name << "' (add it to DATASETS in src/utils.cpp)\n";
        return 1;
    }
    // Coordinator::load does not check its inputs, so fail early on missing files.
    for (const char* f : kRoutingFiles) {
        if (!fs::exists(fs::path(routing_dir) / f)) {
            std::cerr << "Missing " << (fs::path(routing_dir) / f) << "\n";
            if (std::string(f) == "partitions_.bin" && fs::exists(fs::path(routing_dir) / "partitions.bin"))
                std::cerr << "  Found partitions.bin (pre-rename name); symlink it as partitions_.bin.\n";
            return 1;
        }
    }

    auto now = std::time(nullptr);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");

    std::pair<int, int> data_info = get_dataset_info(DATASETS[dataset_name]["base_file"]);
    int nvectors = data_info.first;
    int dim = data_info.second;
    std::cout << "Dataset has " << nvectors << " vectors with dimension " << dim << "\n";

    std::vector<float> query_vectors = readVecs(DATASETS[dataset_name]["query_file"], dim);
    int num_queries = query_vectors.size() / dim;
    printf("Max threads: %d\n", omp_get_max_threads());

    Coordinator metaIndex(dim);
    std::cout << "Loading routing layer from " << routing_dir << " (ef_search=" << ef_search << ")\n";
    metaIndex.load(routing_dir, ef_search);

    const std::vector<int>& partitions = metaIndex.get_partitions();
    if (partitions.empty()) {
        std::cerr << "Loaded routing layer has no partitions\n";
        return 1;
    }
    int num_partitions = *std::max_element(partitions.begin(), partitions.end()) + 1;
    std::cout << "Loaded " << partitions.size() << " centers in " << num_partitions << " partitions\n";

    std::string log_id = "theoretical_partition_quality_" + dataset_name + "_" +
                         std::to_string(num_partitions) + "_" + oss.str();
    fs::create_directories(log_id);
    {
        std::ofstream src(log_id + "/routing_source.txt");
        src << "routing_dir=" << fs::absolute(routing_dir).string() << "\n"
            << "ef_search=" << ef_search << "\n";
    }

    std::cout << "Reading ground truth data from " << DATASETS[dataset_name]["gt_file"] << "\n";
    std::vector<std::vector<int>> gt_indices = readGTBin(DATASETS[dataset_name]["gt_file"]);

    // Unique GT indices, sorted so the base file is read roughly sequentially.
    std::vector<int> needed_indices;
    needed_indices.reserve(static_cast<size_t>(num_queries) * k);
    for (int i = 0; i < num_queries; i++) {
        size_t gt_k = std::min(static_cast<size_t>(k), gt_indices[i].size());
        for (size_t j = 0; j < gt_k; j++) {
            int idx = gt_indices[i][j];
            if (idx >= 0 && idx < nvectors) needed_indices.push_back(idx);
        }
    }
    std::sort(needed_indices.begin(), needed_indices.end());
    needed_indices.erase(std::unique(needed_indices.begin(), needed_indices.end()), needed_indices.end());

    // Shared with theoretical_partitioning_quality and the Python oracles.
    std::string cache_dir = fs::path(DATASETS[dataset_name]["base_file"]).parent_path().string();
    std::string cached_gt_path = cache_dir + "/cached_gt_vectors_" + dataset_name + ".bin";

    std::vector<float> preloaded_vecs;
    size_t cached_n = 0, cached_dim = 0;
    if (fs::exists(cached_gt_path) &&
        loadCachedGTVectors(cached_gt_path, preloaded_vecs, cached_n, cached_dim)) {
        if (cached_n == needed_indices.size() && cached_dim == static_cast<size_t>(dim)) {
            std::cout << "Loaded " << cached_n << " cached GT vectors from " << cached_gt_path << "\n";
        } else {
            std::cout << "Cached GT vectors are stale (" << cached_n << "x" << cached_dim << ", expected "
                      << needed_indices.size() << "x" << dim << "); regenerating\n";
            preloaded_vecs.clear();
        }
    }
    if (preloaded_vecs.empty()) {
        std::cout << "Gathering " << needed_indices.size() << " unique GT vectors from base file\n";
        if (!gatherGTVectors(DATASETS[dataset_name]["base_file"], needed_indices, dim, preloaded_vecs))
            return 1;
        saveCachedGTVectors(cached_gt_path, preloaded_vecs, needed_indices.size(), dim);
    }

    std::unordered_map<int, size_t> idx_to_buf;
    idx_to_buf.reserve(needed_indices.size());
    for (size_t i = 0; i < needed_indices.size(); i++) idx_to_buf[needed_indices[i]] = i;

    std::cout << "Computing ground truth partition assignments\n";
    std::vector<std::vector<size_t>> gt_partitions(num_queries);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_queries; i++) {
        size_t gt_k = std::min(static_cast<size_t>(k), gt_indices[i].size());
        gt_partitions[i].reserve(gt_k);
        for (size_t j = 0; j < gt_k; ++j) {
            auto it = idx_to_buf.find(gt_indices[i][j]);
            if (it == idx_to_buf.end()) continue;
            float* vec = preloaded_vecs.data() + it->second * dim;
            gt_partitions[i].push_back(metaIndex.get_current_partition(vec));
        }
    }

    std::ofstream out(log_id + "/routing_metrics.csv");
    out << "mode,param,recall,activation,imbalance,query_time_s\n";
    std::ofstream raw_counts_out(log_id + "/routing_partition_counts.csv");
    raw_counts_out << "mode,param,partition_id,count\n";

    auto run_mode = [&](const std::string& mode_name, RoutingMode mode, double param) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<size_t>> query_partitions =
            metaIndex.route_queries(query_vectors, mode, static_cast<float>(param));
        double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        std::vector<long long> per_partition_counts(num_partitions, 0);
        double total_recall = 0.0;
        double total_activation = 0.0;
        for (int i = 0; i < num_queries; i++) {
            const std::vector<size_t>& visited = query_partitions[i];
            for (size_t pid : visited)
                if (pid < static_cast<size_t>(num_partitions)) per_partition_counts[pid] += 1;
            total_activation += static_cast<double>(visited.size()) / num_partitions;

            int hits = 0;
            for (size_t part : gt_partitions[i])
                if (std::find(visited.begin(), visited.end(), part) != visited.end()) hits += 1;
            if (!gt_partitions[i].empty())
                total_recall += static_cast<double>(hits) / gt_partitions[i].size();
        }

        // Imbalance = coefficient of variation of per-partition query load.
        double mean = std::accumulate(per_partition_counts.begin(), per_partition_counts.end(), 0.0) /
                      num_partitions;
        double variance = 0.0;
        for (long long c : per_partition_counts) variance += (c - mean) * (c - mean);
        variance /= num_partitions;
        double imbalance = mean > 0.0 ? std::sqrt(variance) / mean : 0.0;

        double recall = total_recall / num_queries;
        double activation = total_activation / num_queries;
        out << mode_name << ',' << param << ',' << recall << ',' << activation << ','
            << imbalance << ',' << elapsed << '\n';
        for (int p = 0; p < num_partitions; p++)
            raw_counts_out << mode_name << ',' << param << ',' << p << ',' << per_partition_counts[p] << '\n';
        std::cout << mode_name << " param=" << param << " recall=" << recall
                  << " activation=" << activation << " imbalance=" << imbalance
                  << " time_s=" << elapsed << "\n";
    };

    std::cout << "Computing routing metrics\n";
    for (int bf : {1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50})
        run_mode("branching_factor", RoutingMode::BranchingFactor, bf);

    // Union of the Figure 2 sweep (from 0.5) and the current sweep (from 0.65).
    for (double rt : {0.5, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99})
        run_mode("recall_target", RoutingMode::RecallTarget, rt);

    // nprobe = num_partitions is trivially recall 1.0; the plot scripts append it.
    for (int np = 1; np < num_partitions; np++)
        run_mode("nprobe", RoutingMode::NProbe, np);

    std::cout << "Results written to " << log_id << "\n";
    return 0;
}
