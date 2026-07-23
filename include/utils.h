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

// Timing/quality measurements produced by Coordinator::build / Executor::build.
// The index populates this; experiments read it back and persist it.
struct BuildMetrics {
    double index_build_time      = 0.0;  // meta-HNSW (coordinator) or sub-HNSW (executor) build
    double kmeans_time           = 0.0;
    size_t kmeans_num_iterations = 0;
    double kaffpa_time           = 0.0;  // KaHIP graph partitioning
    double edge_cut_ratio        = 0.0;
};

// Create logs/run_<log_id>/ if needed and return its path.
std::string ensure_log_dir(const std::string& log_id);

// Write the per-role build-metrics JSON. index_file (if it exists) contributes an
// "index_size" field via std::filesystem::file_size.
void write_controller_build_json(const std::string& out_path, const BuildMetrics& m,
                                 double distribute_time,
                                 const std::vector<int>& per_partition_counts,
                                 const std::string& index_file = "");
void write_executor_build_json(const std::string& out_path, const BuildMetrics& m,
                               size_t num_elements, const std::string& index_file = "");

// Binary dumps of build artifacts into `dir` (thin wrappers over logFloatVec/logIntVec).
void dump_centers(const std::string& dir, const std::vector<float>& centers);
void dump_partitions(const std::string& dir, const std::vector<int>& partitions, bool all = false);
void dump_partition_dists(const std::string& dir, const std::vector<float>& dists);

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