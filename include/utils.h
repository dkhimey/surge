#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mpi.h>
#include <random>
#include <climits>
#include <filesystem>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <map>
#include <vector>
#include <stdexcept>
#include "json.hpp"
#include "vector_utils.h"

using json = nlohmann::json;

extern std::unordered_map<std::string, std::map<std::string, std::string>> DATASETS;

typedef enum {
    FVECS,
    BVECS,
    I8BIN,
    U8BIN,
    FBIN
} FileFormat;

typedef enum {
    SAMPLE_REQUEST,
    VECTOR_REQUEST,
    SAMPLE_SEND,
    VECTOR_SEND,
    QUERY_SEND,
    INSERT_SEND,
    INSERT_SUCCESS,
    LOG_SEND,
    END_OF_COMMUNICATION,
    PARTIAL_REBUILD_REQUEST,
    FULL_REBUILD_REQUEST,
    CHECKPOINT_REQUEST,
    META_HNSW_SEND,
    META_PARTITIONS_SEND,
    REBUILD_SUCCESS,
    DELETE_SEND,
    DELETE_SUCCESS,
    NOT_FOUND,
    QUERY_STREAM,
    END_OF_STREAM,
    INSERT_STREAM,
    SET_EF_SEARCH,
    QUERY_BATCH_SEND,
    INSERT_BATCH_SEND,
    INSERT_BATCH_SUCCESS,
    DELETE_BATCH_SEND,
    DELETE_BATCH_SUCCESS,
    DATASET_INFO_SEND,
    SIZE_REQUEST,          // coordinator asks an executor for its current element count
    INPLACE_REBUILD_REQUEST // delta rebuild: mark-delete departing + insert arriving
} MessageType;

typedef struct MessageHeader {
    MessageType type;
    size_t size;
    int tag;

    MessageHeader() {}

    MessageHeader(MessageType type, size_t size) :
        type(type), size(size) {
    }
} MessageHeader;

typedef enum { 
    IDLE, 
    REBUILDING, 
    DRAINING 
} RebuildState;

typedef struct BenchmarkParams {
    FileFormat format; 
    std::string base_file;
    std::string query_file;
    std::string ground_truth_file;
    std::string insert_file;
    std::string ground_truth_prefix;

    size_t branching_factor;
    size_t num_neighbors_request;
    size_t num_closest_from_ground;
    size_t num_to_insert;

    BenchmarkParams() {}

    BenchmarkParams(const std::string& file_format,
                    const std::string& base_file, 
                    const std::string& query_file, 
                    const std::string& ground_truth_file,
                    const std::string& insert_file,
                    const std::string& ground_truth_prefix,
                    size_t branching_factor,
                    size_t num_neighbors_request,
                    size_t num_closest_from_ground,
                    size_t num_to_insert_) : 
                base_file(base_file), 
                query_file(query_file), 
                ground_truth_file(ground_truth_file),
                insert_file(insert_file),
                ground_truth_prefix(ground_truth_prefix),
                branching_factor(branching_factor),
                num_neighbors_request(num_neighbors_request),
                num_closest_from_ground(num_closest_from_ground) {
                    num_to_insert = num_to_insert_;
                    if (file_format == "bvecs") format = BVECS;
                    if (file_format == "fvecs") format = FVECS;
                    if (file_format == "i8bin") format = I8BIN;
                    if (file_format == "u8bin") format = U8BIN;
                    if (file_format == "fbin") format = FBIN;
                }
} BenchmarkParams;

typedef struct IndexFiles {
    std::string meta_dir;
    std::string sub_prefix;

    IndexFiles() {};
    IndexFiles(const std::string& meta_dir,
               const std::string& sub_prefix) : 
            meta_dir(meta_dir), sub_prefix(sub_prefix) {}
} IndexFiles;

typedef struct ReIndexParams {
    int full_threshold, partial_threshold;

    ReIndexParams() {}
    ReIndexParams(int full_threshold,
                  int partial_threshold) : 
                  full_threshold(full_threshold), 
                  partial_threshold(partial_threshold) {
                    assert(full_threshold >= partial_threshold);
                  }

} ReIndexParams;

typedef struct Config {
    size_t num_vectors;
    size_t dim;
    size_t sample_size;
    size_t kmeans_centers;
    size_t M_meta;
    size_t M_sub;
    size_t ef_construction;
    size_t num_partitions;
    size_t num_building_threads;
    size_t VECTOR_BATCH_SIZE;
    float kmeans_epsilon;
    bool mips;
    bool check_recalls;
    bool check_logs = false;
    std::string log_id;

    json raw_data;

    BenchmarkParams benchmark_params;
    IndexFiles index_files;
    ReIndexParams reindex_params;

    Config() {
        auto now = std::time(nullptr);
        auto* tm = std::localtime(&now);
        std::ostringstream oss;
        oss << std::put_time(tm, "%Y%m%d_%H%M%S");
        log_id = oss.str();
        std::cout << "LOG ID " << log_id << "\n";
    }

    Config(const std::string& filename, bool build = true, std::string log_id_ = "") {
        std::ifstream input(filename);
        if (!input.is_open()) {
            throw std::runtime_error("Cannot open config file: " + filename);
        }

        input >> raw_data;

        num_vectors = raw_data["num_vectors"];
        dim = raw_data["dim"];
        sample_size = raw_data["sample_size"];
        kmeans_centers= raw_data["kmeans_centers"];
        kmeans_epsilon = raw_data["kmeans_EPSILON"];
        M_meta = raw_data["M_meta"];
        M_sub = raw_data["M_sub"];
        ef_construction = raw_data["ef_construction"];
        num_partitions = raw_data["num_partitions"];
        num_building_threads = raw_data["num_building_threads"];
        VECTOR_BATCH_SIZE = raw_data["VECTOR_BATCH_SIZE"];
        mips = raw_data["mips"];
        check_recalls = raw_data["check_recalls"];
        check_logs = raw_data["check_logs"];

        size_t num_to_insert = 0;
        if (raw_data["benchmark_params"].contains("num_to_insert")) {
            num_to_insert = raw_data["benchmark_params"]["num_to_insert"];
        }

        std::string ground_truth_prefix = "";
        if (raw_data["benchmark_params"].contains("ground_truth_prefix")) {
            ground_truth_prefix = raw_data["benchmark_params"]["ground_truth_prefix"];
        }

        benchmark_params = BenchmarkParams(raw_data["benchmark_params"]["file_format"],
                                         raw_data["benchmark_params"]["base_file"],
                                         raw_data["benchmark_params"]["query_file"],
                                         raw_data["benchmark_params"]["ground_truth_file"],
                                         raw_data["benchmark_params"]["base_file"],
                                         ground_truth_prefix,
                                         raw_data["benchmark_params"]["branching_factor"],
                                         raw_data["benchmark_params"]["num_neighbors_request"],
                                         raw_data["benchmark_params"]["num_closest_from_ground"],
                                         num_to_insert);

        index_files = IndexFiles(raw_data["index_files"]["meta_dir"],
                                 raw_data["index_files"]["sub_prefix"]);

        reindex_params = ReIndexParams(raw_data["reindex_params"]["full_threshold"],
                                       raw_data["reindex_params"]["partial_threshold"]);

        if (build) {
            auto now = std::time(nullptr);
            auto* tm = std::localtime(&now);
            std::ostringstream oss;
            oss << std::put_time(tm, "%Y%m%d_%H%M%S");
            log_id = oss.str();

            raw_data["last_build_id"] = log_id;
            std::ofstream outputFile(filename);
            outputFile << raw_data.dump(4);
            outputFile.close();
        } else if (log_id_ == "") {
            log_id = raw_data["last_build_id"];
        } else {
            log_id = log_id_;
        }
    }
} Config;

typedef struct Log {
    size_t correct, total;

    size_t kmeans_num_iterations;
    size_t num_elements;

    double index_build_time;

    double partition_time, kmeans_time, karlsuhe_time;
    double search_time, send_time;
    double edge_cut_ratio;

    json config_data;
    std::string log_dir;
    std::string run_id;
    std::string meta_index_path;  // set explicitly when not using a config file
    std::string sub_index_path;   // set explicitly when not using a config file

    Log(std::string& log_id) {
        // TODO
        log_dir = "logs/run_" + log_id;
        if (!std::filesystem::exists(log_dir)) {
            std::filesystem::create_directories(log_dir);
        }

    }


    Log(std::string& log_id, json raw_data) {
        config_data = raw_data;
        log_dir = "logs/run_" + log_id;
        if (!std::filesystem::exists(log_dir)) {
            std::filesystem::create_directories(log_dir);
        }

        auto now = std::time(nullptr);
        auto* tm = std::localtime(&now);
        std::ostringstream oss;
        oss << std::put_time(tm, "%Y%m%d_%H%M%S");
        run_id = oss.str();

        std::string config_out = log_dir + "/config.json";
        std::ofstream outputFile(config_out);
        outputFile << raw_data.dump(4);
        outputFile.close();
    }

    void saveControllerLog(std::vector<int> per_partition_counts) {
        std::string filename = log_dir + "/controller_build.json";
        json data;
        data["index_build_time"] = index_build_time;
        data["index_size"] = 0;

        std::string index_path;
        if (!meta_index_path.empty()) {
            index_path = meta_index_path;
        } else if (config_data.contains("index_files") &&
                   config_data["index_files"].contains("meta_dir") &&
                   config_data["index_files"]["meta_dir"].is_string()) {
            index_path = std::string(config_data["index_files"]["meta_dir"]) + "/metaHNSW.bin";
        }
        if (!index_path.empty() && std::filesystem::exists(index_path)) {
            data["index_size"] = std::filesystem::file_size(index_path);
        }

        data["partition_time"] = partition_time;
        data["kmeans_num_iterations"] = kmeans_num_iterations;
        data["kmeans_time"] = kmeans_time;
        data["karlsuhe_time"] = karlsuhe_time;
        data["edge_cut_ratio"] = edge_cut_ratio;
        data["search_time"] = search_time;
        data["send_time"] = send_time;
        data["per_partition_counts"] = per_partition_counts;

        std::ofstream outputFile(filename);
        outputFile << data.dump(4);
        outputFile.close();
    }

    void saveExecutorLog(size_t node) {
        char node_str[10];
        snprintf(node_str, 10, "%lu", node);
        std::string filename = log_dir + "/executor_" + node_str + "_build.json";
        json data;
        data["index_build_time"] = index_build_time;
        data["index_size"] = 0;

        std::string index_path;
        if (!sub_index_path.empty()) {
            index_path = sub_index_path;
        } else if (config_data.contains("index_files") &&
                   config_data["index_files"].contains("sub_prefix") &&
                   config_data["index_files"]["sub_prefix"].is_string()) {
            char suffix[10];
            snprintf(suffix, 10, "_%lu.bin", node);
            index_path = std::string(config_data["index_files"]["sub_prefix"]) + suffix;
        }
        if (!index_path.empty() && std::filesystem::exists(index_path)) {
            data["index_size"] = std::filesystem::file_size(index_path);
        }

        data["num_elements"] = num_elements;

        std::ofstream outputFile(filename);
        outputFile << data.dump(4);
        outputFile.close();
    }

    void saveResult(size_t correct, size_t total, double query_time) {
        config_data["result"]["correct"] = correct;
        config_data["result"]["total"] = total;
        config_data["result"]["query_total_time"] = query_time;
        config_data["result"]["qps"] = total / query_time;

        std::string config_out = log_dir + "/config_" + run_id + ".json";
        std::ofstream outputFile(config_out);
        outputFile << config_data.dump(4);
        outputFile.close();
    }

    void saveResult(double recall, size_t correct, size_t total, double query_time) {
        config_data["result"]["recall"] = recall;
        config_data["result"]["correct"] = correct;
        config_data["result"]["total"] = total;
        config_data["result"]["query_total_time"] = query_time;
        config_data["result"]["qps"] = total / query_time;

        std::string config_out = log_dir + "/config_" + run_id + ".json";
        std::ofstream outputFile(config_out);
        outputFile << config_data.dump(4);
        outputFile.close();
    }

    void logQueryLatencies(std::vector<double>& vec, std::string id) {
        std::string output_filename = log_dir + "/query_latencies_" + id + ".bin";
        logDoubleVec(vec, output_filename);
        std::cout << "Latencies saved to " << output_filename << "\n";
    }

    void logQueryLatencies(std::vector<double>& vec) {
        std::string output_filename = log_dir + "/query_latencies_" + run_id + ".bin";
        logDoubleVec(vec, output_filename);
        std::cout << "Latencies saved to " << output_filename << "\n";
    }

    void logCenters(std::vector<float>& centers) {
        std::string output_filename = log_dir + "/centers.bin";

        logFloatVec(centers, output_filename);
        std::cout << "Centers saved to " << output_filename << "\n";
    }

    void logPartitions(std::vector<int>& partitions, bool all = false) {
        std::string output_filename;
        if (all) output_filename = log_dir + "/all_partitions.bin";
        else output_filename = log_dir + "/center_partitions.bin";

        logIntVec(partitions, output_filename);
        std::cout << "Center Partitions saved to " << output_filename << "\n";
    }

    void logPartitionDists(std::vector<float>& distances) {
        std::string output_filename = log_dir + "/partition_dists.bin";
        logFloatVec(distances, output_filename);
        std::cout << "Distances saved to " << output_filename << "\n";
    }

    void logInsertDists(std::vector<float>& distances, int step) {
        std::string output_filename = log_dir + "/insert_" + std::to_string(step) + "_dists.bin";
        logFloatVec(distances, output_filename);
        std::cout << "Distances saved to " << output_filename << "\n";
    }

    void logInsertDists(std::vector<float>& distances, std::string step) {
        std::string output_filename = log_dir + "/insert_" + step + "_dists.bin";
        logFloatVec(distances, output_filename);
        std::cout << "Distances saved to " << output_filename << "\n";
    }

    void logExecutorHits(std::vector<std::atomic<int>>& vec) {
        std::string output_filename = log_dir + "/executor_hits_" + run_id + ".bin";

        std::vector<int> snapshot;
        snapshot.reserve(vec.size());
        for (int i = 0; i < vec.size(); i++) {
            snapshot.push_back(vec[i].load());  // load atomic safely
        }

        logIntVec(snapshot, output_filename);
    }

    void logExecutorHits(std::vector<std::atomic<int>>& vec, std::string id) {
        std::string output_filename = log_dir + "/executor_hits_" + id + ".bin";

        std::vector<int> snapshot;
        snapshot.reserve(vec.size());
        for (int i = 0; i < vec.size(); i++) {
            snapshot.push_back(vec[i].load());  // load atomic safely
        }

        logIntVec(snapshot, output_filename);
    }

    void logAccessRate(std::vector<int>& vec) {
        std::string output_filename = log_dir + "/access_rate_" + run_id + ".bin";
        logIntVec(vec, output_filename);
    }

     void logAccessRate(std::vector<int>& vec, std::string id) {
        std::string output_filename = log_dir + "/access_rate_" + id + ".bin";
        logIntVec(vec, output_filename);
    }

    void logRepartition(double hnsw_time, double bottom_layer, 
                        double partition_time, double partition_relabel) {
        std::string output_filename = log_dir + "/rebuild_" + run_id + ".bin";

        std::cout << "[Coordinator] - meta hnsw time: " << hnsw_time << "\n";
        std::cout << "[Coordinator] - bottom layer graph build time: " << bottom_layer << "\n";
        std::cout << "[Coordinator] - bottom layer partition time: " << partition_time << "\n";
        std::cout << "[Coordinator] - bottom layer relabel time: " << partition_relabel << "\n";
    }

    void logCoordinatorRebuild(MessageType rebuild_type,
                               double total_repartition_time, 
                               double hnsw_serialization_time, 
                               double reorganization_time) {
        std::string output_filename = log_dir + "/rebuild_" + run_id + ".bin";

        std::string type;
        if (rebuild_type == FULL_REBUILD_REQUEST) type = "FULL REBUILD";
        else type = "PARTIAL REBUILD";

        std::cout << "[Coordinator] - rebuild type: " << type << "\n";
        std::cout << "[Coordinator] - total repartition time: " << total_repartition_time << "\n";
        std::cout << "[Coordinator] - hnsw serialization time: " << hnsw_serialization_time << "\n";
        std::cout << "[Coordinator] - rebuild reorganization time: " << reorganization_time << "\n";
        
    }

    void logExecutorRebuild() {

    }
    
} Log;

FileFormat getFileFormat(const std::string& filename);
std::pair<int, int> get_dataset_info(const std::string& base_file);
std::vector<float> getSample(const std::string& filename, size_t max_elements, size_t dim, size_t sample_size, size_t start_offset = 0);
std::vector<float> readFvecs(const std::string& filename, size_t vector_dim, int n = INT_MAX, int offset = 0);
std::vector<float> readBvecs(const std::string& filename, size_t vector_dim, int n = INT_MAX, int offset = 0);
std::vector<float> readI8bin(const std::string& filename, size_t vector_dim, int n = INT_MAX, int offset = 0);
std::vector<float> readU8bin(const std::string& filename, size_t vector_dim, int n = INT_MAX, int offset = 0);
std::vector<float> readFbin(const std::string& filename, size_t vector_dim, int n = INT_MAX, int offset = 0);
std::vector<std::vector<int>> readGTIvecs(const std::string& filename);
std::vector<std::vector<int>> readGTBin(const std::string& filename);
std::vector<float> readVecs(const std::string& filename, size_t vector_dim, int n = INT_MAX, int offset = 0);
std::vector<std::vector<int>> readGT(const std::string& filename, FileFormat format);
int maximum_matching(const std::vector<std::vector<int>>& cost, std::vector<int>& assignment);
int kmeans(float* sample, size_t n_points, size_t dim, size_t n_centers, float* centers, int* counts,
           float EPSILON = 1e-4f, int EPOCHS = 100, bool verbose = true);

struct RunbookStep {
    int         step_num = -1;
    std::string operation;         // "insert" | "delete" | "search"
    int         start    = -1;
    int         end      = -1;     // exclusive upper bound (Python slice convention)
};

namespace runbook_detail {
// Trim leading/trailing whitespace and quotes.
inline std::string trim(const std::string& s) {
    const std::string ws = " \t\r\n'\"";
    size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}
}

inline std::vector<RunbookStep> load_runbook(const std::string& path, const std::string& dataset_key);