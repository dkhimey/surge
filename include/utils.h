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
#include <exception>
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

typedef struct Log {
    size_t correct, total;

    size_t kmeans_num_iterations;
    size_t num_elements;

    double index_build_time;

    double partition_time, kmeans_time, karlsuhe_time;
    double search_time, send_time;
    double edge_cut_ratio;

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


    void saveControllerLog(std::vector<int> per_partition_counts) {
        std::string filename = log_dir + "/controller_build.json";
        json data;
        data["index_build_time"] = index_build_time;
        data["index_size"] = 0;

        std::string index_path;
        if (!meta_index_path.empty()) {
            index_path = meta_index_path;
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
        }
        if (!index_path.empty() && std::filesystem::exists(index_path)) {
            data["index_size"] = std::filesystem::file_size(index_path);
        }

        data["num_elements"] = num_elements;

        std::ofstream outputFile(filename);
        outputFile << data.dump(4);
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

std::vector<RunbookStep> load_runbook(const std::string& path, const std::string& dataset_key);

// a std::terminate handler that turns an uncaught exception into a
// clean, rank-tagged MPI_Abort. Without it, an exception escaping main() calls
// abort() on a single rank, leaving peers deadlocked in MPI calls. Call once,
// immediately after MPI_Init, in every MPI driver's main().
inline void install_mpi_terminate_handler() {
    std::set_terminate([]() {
        int rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        try {
            if (auto e = std::current_exception()) std::rethrow_exception(e);
        } catch (const std::exception& ex) {
            std::cerr << "[rank " << rank << "] FATAL: " << ex.what() << "\n";
        } catch (...) {
            std::cerr << "[rank " << rank << "] FATAL: unknown exception\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    });
}