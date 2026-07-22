// =============================================================================
// runbook_centers.cpp
//
// Per-step routing-state generator.  Replays a Big-ANN streaming runbook and,
// for every update step, produces the coordinator's routing state used by the
// theoretical-recall pipeline and by dynamic_runbook_experiment's
// --init-state-dir starting state.
//
// KaHIP partitioning of the emitted base layer is handled separately by
// runbook_partitions_parallel.cpp.
//
// Output per update step:
//   step_NNNNNN_centers.csv        - k x dim float32 centroids
//   step_NNNNNN_center_counts.csv  - k int32 cluster sizes (plain decimal)
//   step_NNNNNN_hnsw.bin           - hnswlib routing index (for recall eval)
//   step_NNNNNN_base_layer.csv     - base-layer edge list (input to partitioner)
//
// Build:   make experiments        (produces bin/runbook_centers)
//
// Run (example):
//   ./bin/runbook_centers \
//       --dataset  msturing-100M-clustered \
//       --centers  10000 \
//       --out-dir  cluster_history_msturing-100M-clustered_10000
//
// KMeans and HNSW assignment are seeded (mt19937(42), hnswlib seed 100) so runs
// are reproducible.
//
// Optional flags:
//   --initial-centers <centers.csv>
//   --load-state-assignments <point_to_centroid.csv>
//       Start from a fixed centroid set and point->centroid assignment instead
//       of running the first-batch KMeans + HNSW assignment.  Use both together
//       to resume from a previously generated step-1 state.
//   --ignore-zero-counts      Build HNSW over all k nodes but drop edges
//                             incident to empty (zero-count) centroids.
//   --skip-zero-count-inserts Do not insert empty centroids into the HNSW at all.
// =============================================================================
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hnswlib.h"
#include "vector_utils.h"
#include "utils.h"

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Must match hnsw_per_step.cpp
constexpr double MIN_EDGE_DIST = 1e-4;

// KMeans defaults (sklearn KMeans defaults: max_iter=300, tol=1e-4)
// Here tol is absolute squared centroid shift (see difference #4 above)
constexpr int    KMEANS_MAX_ITER = 300;
constexpr float  KMEANS_TOL      = 1e-4f;   // squared-shift threshold
constexpr int    KMEANS_SEED     = 42;
constexpr int    SAMPLE_LIMIT    = 100000;   // max vectors sampled for initial KMeans

// HNSW construction parameters (must match Python _build_hnsw and hnsw_per_step)
constexpr int    HNSW_M              = 16;
constexpr int    HNSW_EF_CONSTRUCTION = 200;
constexpr int    HNSW_EF_SEARCH       = 200;

// Batch size for HNSW queries (matches Python's batch_size=100000)
constexpr int    QUERY_BATCH_SIZE = 100000;


// ─────────────────────────────────────────────────────────────────────────────
// Runbook types and parser
// ─────────────────────────────────────────────────────────────────────────────

struct RunbookStep {
    std::string operation;   // "insert" | "delete" | "search"
    int start = -1;
    int end   = -1;          // exclusive upper bound (Python slice convention)
};

// Trim leading/trailing whitespace and single-quotes from a string
static std::string trim(const std::string& s) {
    const std::string ws = " \t\r\n'\"";
    size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

// Parse the NeurIPS runbook YAML format:
//
//   msturing-30M-clustered:
//     max_pts: 10292043
//     1:
//       operation: 'insert'
//       start: 0
//       end: 38806
//     2:
//       operation: 'search'
//     ...
//
// Returns steps sorted by their integer key, search steps included (caller skips).
static std::vector<RunbookStep> load_runbook(const std::string& path,
                                              const std::string& dataset_key)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open runbook: " + path);

    std::map<int, RunbookStep> step_map;
    bool       in_dataset = false;
    int        cur_step   = -1;
    RunbookStep cur;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

        // Count leading spaces to determine indent level
        int indent = 0;
        while (indent < (int)line.size() && line[indent] == ' ') ++indent;
        const std::string content = line.substr(indent);

        if (indent == 0) {
            // Top-level key line (e.g. "msturing-30M-clustered:")
            if (cur_step >= 0) { step_map[cur_step] = cur; cur_step = -1; }
            size_t colon = content.find(':');
            std::string key = trim(colon != std::string::npos ? content.substr(0, colon) : content);
            in_dataset = (key == dataset_key);

        } else if (in_dataset && indent == 2) {
            // Second-level key: either a step number ("1:") or metadata ("max_pts:")
            if (cur_step >= 0) { step_map[cur_step] = cur; cur_step = -1; }
            size_t colon = content.find(':');
            if (colon == std::string::npos) continue;
            std::string key = trim(content.substr(0, colon));
            bool all_digits = !key.empty() &&
                              std::all_of(key.begin(), key.end(),
                                          [](unsigned char c){ return std::isdigit(c); });
            if (all_digits) {
                cur_step = std::stoi(key);
                cur = RunbookStep{};
            }

        } else if (in_dataset && indent == 4 && cur_step >= 0) {
            // Step field ("operation:", "start:", "end:")
            size_t colon = content.find(':');
            if (colon == std::string::npos) continue;
            std::string key = trim(content.substr(0, colon));
            std::string val = trim(content.substr(colon + 1));
            if      (key == "operation") cur.operation = val;
            else if (key == "start")     cur.start     = std::stoi(val);
            else if (key == "end")       cur.end       = std::stoi(val);
        }
    }
    if (cur_step >= 0) step_map[cur_step] = cur;

    std::vector<RunbookStep> result;
    result.reserve(step_map.size());
    for (auto& [k, v] : step_map) result.push_back(v);
    return result;
}


// ─────────────────────────────────────────────────────────────────────────────
// Vector file I/O
//
// Supported formats (all share the same 8-byte header):
//   .fbin   – float32  (4 bytes/element)
//   .u8bin  – uint8    (1 byte/element)  → promoted to float32 on read
//   .i8bin  – int8     (1 byte/element)  → promoted to float32 on read
//
// Format is detected from the file extension; use --vector-type to override.
// ─────────────────────────────────────────────────────────────────────────────

enum class VecType { f32, u8, i8 };

static VecType vec_type_from_path(const std::string& path) {
    if (path.size() >= 6 && path.substr(path.size() - 6) == ".u8bin") return VecType::u8;
    if (path.size() >= 6 && path.substr(path.size() - 6) == ".i8bin") return VecType::i8;
    return VecType::f32;  // .fbin or anything else
}

static std::string vec_type_name(VecType vt) {
    switch (vt) {
        case VecType::u8:  return "uint8";
        case VecType::i8:  return "int8";
        default:           return "float32";
    }
}

// Returns {num_points, dim}  – header format is identical for all three types
static std::pair<uint32_t, uint32_t> fbin_header(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + filename);
    uint32_t n, d;
    f.read(reinterpret_cast<char*>(&n), sizeof n);
    f.read(reinterpret_cast<char*>(&d), sizeof d);
    if (!f) throw std::runtime_error("Failed to read header: " + filename);
    return {n, d};
}

// Read vectors [start, end) and return them as float32.
// bytes_per_elem: 4 for float32, 1 for uint8/int8.
static std::vector<float> read_vec_slice(const std::string& filename,
                                          uint32_t start, uint32_t end,
                                          VecType vtype)
{
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + filename);

    uint32_t num_points, dim;
    f.read(reinterpret_cast<char*>(&num_points), sizeof num_points);
    f.read(reinterpret_cast<char*>(&dim),        sizeof dim);

    if (end > num_points)
        throw std::runtime_error(
            "read_vec_slice: [" + std::to_string(start) + "," +
            std::to_string(end) + ") exceeds " + std::to_string(num_points));

    const uint32_t count  = end - start;
    const size_t   n_elem = static_cast<size_t>(count) * dim;

    std::vector<float> data(n_elem);

    if (vtype == VecType::f32) {
        f.seekg(8 + static_cast<uint64_t>(start) * dim * sizeof(float), std::ios::beg);
        f.read(reinterpret_cast<char*>(data.data()),
               static_cast<std::streamsize>(n_elem * sizeof(float)));
        if (!f)
            throw std::runtime_error("read_vec_slice(f32): short read for [" +
                                      std::to_string(start) + "," + std::to_string(end) + ")");
    } else if (vtype == VecType::u8) {
        f.seekg(8 + static_cast<uint64_t>(start) * dim * sizeof(uint8_t), std::ios::beg);
        std::vector<uint8_t> raw(n_elem);
        f.read(reinterpret_cast<char*>(raw.data()),
               static_cast<std::streamsize>(n_elem));
        if (!f)
            throw std::runtime_error("read_vec_slice(u8): short read for [" +
                                      std::to_string(start) + "," + std::to_string(end) + ")");
        for (size_t i = 0; i < n_elem; ++i)
            data[i] = static_cast<float>(raw[i]);
    } else {  // i8
        f.seekg(8 + static_cast<uint64_t>(start) * dim * sizeof(int8_t), std::ios::beg);
        std::vector<int8_t> raw(n_elem);
        f.read(reinterpret_cast<char*>(raw.data()),
               static_cast<std::streamsize>(n_elem));
        if (!f)
            throw std::runtime_error("read_vec_slice(i8): short read for [" +
                                      std::to_string(start) + "," + std::to_string(end) + ")");
        for (size_t i = 0; i < n_elem; ++i)
            data[i] = static_cast<float>(raw[i]);
    }

    return data;
}


// ─────────────────────────────────────────────────────────────────────────────
// KMeans++ initialization + Lloyd's iterations (single run)
//
// Returns the converged centres and the final inertia (sum of squared distances
// from each sample to its assigned centre).  The caller picks the best of
// n_init runs via run_kmeans_best_of_n.
// ─────────────────────────────────────────────────────────────────────────────
static std::pair<std::vector<float>, double> run_kmeans_once(
    const float* sample,
    int   n_points,
    int   dim,
    int   k,
    std::mt19937& gen,       // caller-supplied RNG (advanced per call)
    int   max_iter = KMEANS_MAX_ITER,
    float tol      = KMEANS_TOL,
    bool  verbose  = false)
{
    // ── k-means++ initialisation ─────────────────────────────────────────────
    std::vector<float> centers(static_cast<size_t>(k) * dim);
    std::vector<float> min_dist(n_points, std::numeric_limits<float>::max());

    {
        std::uniform_int_distribution<int> uid(0, n_points - 1);
        int first = uid(gen);
        std::copy(sample + first * dim, sample + (first + 1) * dim, centers.data());
    }

    for (int c = 1; c < k; ++c) {
        const float* prev = centers.data() + static_cast<size_t>(c - 1) * dim;
        double total = 0.0;
        for (int i = 0; i < n_points; ++i) {
            float d = computeEuclideanDistance(sample + static_cast<size_t>(i) * dim, prev, dim);
            if (d < min_dist[i]) min_dist[i] = d;
            total += min_dist[i];
        }
        int next = n_points - 1;
        if (total > 0.0) {
            double r = std::uniform_real_distribution<double>(0.0, total)(gen);
            double cumsum = 0.0;
            for (int i = 0; i < n_points; ++i) {
                cumsum += min_dist[i];
                if (cumsum >= r) { next = i; break; }
            }
        } else {
            next = std::uniform_int_distribution<int>(0, n_points - 1)(gen);
        }
        std::copy(sample + static_cast<size_t>(next) * dim,
                  sample + static_cast<size_t>(next + 1) * dim,
                  centers.data() + static_cast<size_t>(c) * dim);
    }

    // ── Lloyd's iterations ───────────────────────────────────────────────────
    std::vector<int>    labels(n_points);
    std::vector<double> new_sums(static_cast<size_t>(k) * dim);
    std::vector<int>    cnt(k);

    for (int iter = 0; iter < max_iter; ++iter) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_points; ++i) {
            const float* pt = sample + static_cast<size_t>(i) * dim;
            int   best   = 0;
            float best_d = computeEuclideanDistance(pt, centers.data(), dim);
            for (int c = 1; c < k; ++c) {
                float d = computeEuclideanDistance(pt, centers.data() + static_cast<size_t>(c) * dim, dim);
                if (d < best_d) { best_d = d; best = c; }
            }
            labels[i] = best;
        }

        std::fill(new_sums.begin(), new_sums.end(), 0.0);
        std::fill(cnt.begin(),      cnt.end(),      0);
        for (int i = 0; i < n_points; ++i) {
            int c = labels[i];
            ++cnt[c];
            const float* pt = sample + static_cast<size_t>(i) * dim;
            double* dst     = new_sums.data() + static_cast<size_t>(c) * dim;
            for (int d = 0; d < dim; ++d) dst[d] += pt[d];
        }

        float max_shift = 0.0f;
        for (int c = 0; c < k; ++c) {
            if (cnt[c] == 0) continue;
            float*  center = centers.data() + static_cast<size_t>(c) * dim;
            double* ns     = new_sums.data() + static_cast<size_t>(c) * dim;
            float shift = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float nv    = static_cast<float>(ns[d] / cnt[c]);
                float delta = center[d] - nv;
                shift      += delta * delta;
                center[d]   = nv;
            }
            if (shift > max_shift) max_shift = shift;
        }
        if (verbose)
            std::cout << "    iter " << std::setw(3) << (iter + 1)
                      << "/" << max_iter << "  max_shift=" << max_shift << "\n";
        if (max_shift <= tol) {
            if (verbose) std::cout << "    Converged at iter " << (iter + 1) << "\n";
            break;
        }
    }

    // ── Inertia (sum of squared distances) ───────────────────────────────────
    // Used to select the best run.  Same definition as sklearn's inertia_.
    double inertia = 0.0;
    for (int i = 0; i < n_points; ++i)
        inertia += static_cast<double>(
            computeEuclideanDistance(sample + static_cast<size_t>(i) * dim,
                                     centers.data() + static_cast<size_t>(labels[i]) * dim, dim));

    return {std::move(centers), inertia};
}

// ─────────────────────────────────────────────────────────────────────────────
// run_kmeans_best_of_n
//
// Runs k-means++ n_init times (each with an independent seeded RNG derived from
// base_seed) and returns the centres with the lowest inertia.
//
// This matches sklearn KMeans(random_state=42) behaviour:
//   sklearn <= 1.1 : n_init=10 by default → pass n_init=10 to match exactly.
//   sklearn >= 1.2 : n_init='auto'=1 for k-means++ → pass n_init=1.
//   Check with: from sklearn.cluster import KMeans; print(KMeans().n_init)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> run_kmeans(
    const float* sample,
    int   n_points,
    int   dim,
    int   k,
    int   n_init   = 1,
    int   base_seed = KMEANS_SEED,
    int   max_iter  = KMEANS_MAX_ITER,
    float tol       = KMEANS_TOL,
    bool  verbose   = false)
{
    std::vector<float> best_centers;
    double best_inertia = std::numeric_limits<double>::max();

    // Each restart uses a deterministically derived seed so the run is fully
    // reproducible while still producing independent initializations.
    for (int run = 0; run < n_init; ++run) {
        if (verbose || n_init > 1)
            std::cout << "    KMeans run " << (run + 1) << "/" << n_init << "\n";

        // Derive a per-run seed from base_seed + run (same trick sklearn uses
        // internally when random_state is an integer).
        std::mt19937 gen(static_cast<uint32_t>(base_seed + run));

        auto [centers, inertia] = run_kmeans_once(
            sample, n_points, dim, k, gen, max_iter, tol, verbose);

        if (verbose || n_init > 1)
            std::cout << "    inertia=" << inertia
                      << (inertia < best_inertia ? "  ← best\n" : "\n");

        if (inertia < best_inertia) {
            best_inertia = inertia;
            best_centers = std::move(centers);
        }
    }
    return best_centers;
}


// ─────────────────────────────────────────────────────────────────────────────
// Load initial centres from a CSV file (for diagnosis: bypass C++ KMeans and
// use centres produced by the Python pipeline at step 1 to isolate whether the
// quality gap is in KMeans or in subsequent HNSW routing).
//
// File format: k rows × dim columns of float values separated by commas —
// identical to the step_NNNNNN_centers.csv output format.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> load_centers_csv(const std::string& path, int k, int dim) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open initial-centers file: " + path);

    std::vector<float> centers;
    centers.reserve(static_cast<size_t>(k) * dim);

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ','))
            centers.push_back(std::stof(cell));
    }

    const size_t expected = static_cast<size_t>(k) * dim;
    if (centers.size() != expected)
        throw std::runtime_error(
            "load_centers_csv: expected " + std::to_string(expected) +
            " values, got " + std::to_string(centers.size()) +
            " in " + path);
    return centers;
}


// ─────────────────────────────────────────────────────────────────────────────
// Load point_to_centroid assignments from a CSV file
//
// File format: one line per vector, "global_id,centroid_index"
// (no header).  Generate from Python's point_to_centroid.npy with:
//   import numpy as np
//   d = np.load('point_to_centroid.npy', allow_pickle=True).item()
//   with open('point_to_centroid.csv', 'w') as f:
//       for gid, cid in d.items():
//           f.write(f'{gid},{cid}\n')
// ─────────────────────────────────────────────────────────────────────────────
static std::unordered_map<uint32_t, int>
load_assignments_csv(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open assignments file: " + path);

    std::unordered_map<uint32_t, int> assignments;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const char* p = line.c_str();
        char* end;
        uint32_t gid = static_cast<uint32_t>(std::strtoul(p, &end, 10));
        if (*end != ',')
            throw std::runtime_error("load_assignments_csv: bad line: " + line);
        int cid = static_cast<int>(std::strtol(end + 1, nullptr, 10));
        assignments[gid] = cid;
    }
    std::cout << "  Loaded " << assignments.size() << " assignments from " << path << "\n";
    return assignments;
}


// ─────────────────────────────────────────────────────────────────────────────
// HNSW wrapper
//
// Wraps hnswlib::HierarchicalNSW with identical parameters to the Python
// _build_hnsw helper (M=16, ef_construction=200, ef_search=200) and to the
// HNSW built by hnsw_per_step.cpp.  Because both use the same hnswlib default
// random_seed=100, a graph built here from a given set of centres will be
// structurally identical to what hnsw_per_step.cpp would produce from the same
// centres CSV, so the assignment HNSW can serve double duty for step 4.
// ─────────────────────────────────────────────────────────────────────────────
struct HnswWrapper {
    // Note: space must outlive index, so it is declared first (destroyed last)
    std::unique_ptr<hnswlib::L2Space>                  space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>>   index;
    int k;
    int dim;
};

// Build a fresh HNSW over k centres (labels 0 … k-1).
// flat_centers: [k × dim] float32 array.
static std::unique_ptr<HnswWrapper> build_hnsw_all(
    const float* flat_centers, int k, int dim,
    int M = HNSW_M, int ef_construction = HNSW_EF_CONSTRUCTION, int ef_search = HNSW_EF_SEARCH)
{
    auto h = std::make_unique<HnswWrapper>();
    h->k   = k;
    h->dim = dim;
    h->space = std::make_unique<hnswlib::L2Space>(static_cast<size_t>(dim));
    h->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        h->space.get(),
        static_cast<size_t>(k),
        static_cast<size_t>(M),
        static_cast<size_t>(ef_construction));
    h->index->ef_ = static_cast<size_t>(ef_search);

    for (int i = 0; i < k; ++i)
        h->index->addPoint(flat_centers + static_cast<size_t>(i) * dim,
                           static_cast<hnswlib::labeltype>(i));
    return h;
}

// Build HNSW skipping zero-count nodes; returns (wrapper, external_label_map)
// where external_label_map[internal_id] = original centroid index.
// Matches hnsw_per_step --skip-zero-count-inserts.
static std::pair<std::unique_ptr<HnswWrapper>, std::vector<int>>
build_hnsw_skip_zero(const float* flat_centers, const std::vector<int64_t>& counts,
                     int k, int dim,
                     int M = HNSW_M, int ef_construction = HNSW_EF_CONSTRUCTION,
                     int ef_search = HNSW_EF_SEARCH)
{
    int nonzero = 0;
    for (int c = 0; c < k; ++c) if (counts[c] > 0) ++nonzero;

    auto h = std::make_unique<HnswWrapper>();
    h->k   = nonzero;
    h->dim = dim;
    h->space = std::make_unique<hnswlib::L2Space>(static_cast<size_t>(dim));
    h->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        h->space.get(),
        static_cast<size_t>(std::max(nonzero, 1)),
        static_cast<size_t>(M),
        static_cast<size_t>(ef_construction));
    h->index->ef_ = static_cast<size_t>(ef_search);

    std::vector<int> label_map;  // internal_id -> external centroid label
    label_map.reserve(nonzero);
    for (int i = 0; i < k; ++i) {
        if (counts[i] == 0) continue;
        h->index->addPoint(flat_centers + static_cast<size_t>(i) * dim,
                           static_cast<hnswlib::labeltype>(i));
        label_map.push_back(i);
    }
    return {std::move(h), std::move(label_map)};
}

// Query nearest centroid (k=1) for each of n_vecs vectors.
// Returns one label per vector.  Thread-safe: hnswlib searchKnn is const.
static std::vector<int> hnsw_query1(const HnswWrapper& h,
                                     const float* vecs, int n_vecs)
{
    std::vector<int> labels(n_vecs);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_vecs; ++i) {
        auto result = h.index->searchKnn(
            vecs + static_cast<size_t>(i) * h.dim, 1);
        labels[i] = static_cast<int>(result.top().second);
    }
    return labels;
}


// ─────────────────────────────────────────────────────────────────────────────
// Centroid utilities
// ─────────────────────────────────────────────────────────────────────────────

// Compute float32 centres from double accumulators.
// Mirrors Python: cluster_sums / np.maximum(counts, 1)[:, None]  .astype(np.float32)
static std::vector<float> compute_centers(const std::vector<double>&  sums,
                                           const std::vector<int64_t>& counts,
                                           int k, int dim)
{
    std::vector<float> centers(static_cast<size_t>(k) * dim);
    for (int c = 0; c < k; ++c) {
        double denom = static_cast<double>(std::max(counts[c], static_cast<int64_t>(1)));
        const double* s = sums.data()    + static_cast<size_t>(c) * dim;
        float*        ct = centers.data() + static_cast<size_t>(c) * dim;
        for (int d = 0; d < dim; ++d)
            ct[d] = static_cast<float>(s[d] / denom);
    }
    return centers;
}


// ─────────────────────────────────────────────────────────────────────────────
// CSV / binary output helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string step_prefix(int step) {
    std::ostringstream oss;
    oss << "step_" << std::setw(6) << std::setfill('0') << step;
    return oss.str();
}

// Save centres CSV.
// Matches Python: np.savetxt(path, centres.astype(np.float32), delimiter=",")
// np.savetxt uses '%.18e' format for float arrays.
static void save_centers_csv(const fs::path& out_dir, int step,
                               const std::vector<float>& centers, int k, int dim)
{
    auto path = out_dir / (step_prefix(step) + "_centers.csv");
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path.string());

    f << std::scientific << std::setprecision(18);
    for (int c = 0; c < k; ++c) {
        const float* row = centers.data() + static_cast<size_t>(c) * dim;
        for (int d = 0; d < dim; ++d) {
            if (d > 0) f << ',';
            f << row[d];
        }
        f << '\n';
    }
}

// Save counts CSV.
// Python writes counts in '%.18e' scientific notation (np.savetxt default for
// all dtypes including int32).  That causes std::stoi in hnsw_per_step.cpp and
// runbook_partitions_parallel.cpp to truncate at the decimal point, silently
// corrupting node weights.  We intentionally write plain decimal integers so
// std::stoi parses correctly — see difference #8 in the header comment.
static void save_counts_csv(const fs::path& out_dir, int step,
                              const std::vector<int64_t>& counts, int k)
{
    auto path = out_dir / (step_prefix(step) + "_center_counts.csv");
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path.string());
    for (int c = 0; c < k; ++c)
        f << static_cast<int32_t>(counts[c]) << '\n';
}

// Save serialised HNSW index (for use by compute_theoretical_recall.py).
static void save_hnsw_bin(const fs::path& out_dir, int step,
                           const HnswWrapper& h)
{
    auto path = out_dir / (step_prefix(step) + "_hnsw.bin");
    h.index->saveIndex(path.string());
}

// Euclidean distance (not squared) – matches hnsw_per_step.cpp
static double euclidean_distance_d(const float* a, const float* b, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
        double diff = a[i] - b[i];
        s += diff * diff;
    }
    return std::sqrt(s);
}

// Save base-layer edge list CSV.
// Three modes to match hnsw_per_step.cpp:
//
//   mode 0 (default)       – all nodes, no zero-count filtering in output
//   mode 1 (ignore-zero)   – HNSW has all nodes; edges to/from zero-count
//                            nodes suppressed  (--ignore-zero-count)
//   mode 2 (skip-insert)   – zero-count nodes not inserted; external labels
//                            used  (--skip-zero-count-inserts)
//
// For mode 0 and 1 the HNSW must have been built with all k nodes (internal
// id == external label).  For mode 2 provide the label_map from
// build_hnsw_skip_zero().
static void save_base_layer_csv(
    const fs::path&             out_dir,
    int                         step,
    const HnswWrapper&          h,
    const std::vector<float>&   centers,   // [k_orig × dim]
    const std::vector<int64_t>& counts,    // [k_orig]
    int                         k_orig,
    int                         dim,
    int                         mode,      // 0, 1, or 2
    const std::vector<int>&     label_map) // only used for mode 2
{
    auto path = out_dir / (step_prefix(step) + "_base_layer.csv");
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path.string());

    if (mode == 2) {
        // Iterate over internal IDs; look up external labels via label_map
        size_t cur_count = h.index->getCurrentElementCount();
        for (size_t iid = 0; iid < cur_count; ++iid) {
            int ext_i = label_map[iid];

            auto* bottom  = h.index->get_linklist_at_level(iid, 0);
            int   nlinks  = h.index->getListCount(bottom);
            const auto* links =
                reinterpret_cast<const hnswlib::tableint*>(bottom + 1);

            for (int j = 0; j < nlinks; ++j) {
                size_t nb_iid = links[j];
                if (nb_iid >= cur_count) continue;
                int ext_j = label_map[nb_iid];

                double dist = euclidean_distance_d(
                    centers.data() + static_cast<size_t>(ext_i) * dim,
                    centers.data() + static_cast<size_t>(ext_j) * dim, dim);
                if (!std::isfinite(dist) || dist < MIN_EDGE_DIST) continue;
                f << ext_i << ',' << ext_j << ',' << dist << '\n';
            }
        }
    } else {
        // mode 0 or 1: internal_id == external label
        for (int i = 0; i < k_orig; ++i) {
            if (mode == 1 && counts[i] == 0) continue;

            auto* bottom  = h.index->get_linklist_at_level(i, 0);
            int   nlinks  = h.index->getListCount(bottom);
            const auto* links =
                reinterpret_cast<const hnswlib::tableint*>(bottom + 1);

            for (int j = 0; j < nlinks; ++j) {
                int nb = static_cast<int>(links[j]);
                if (nb < 0 || nb >= k_orig) continue;
                if (mode == 1 && counts[nb] == 0) continue;

                double dist = euclidean_distance_d(
                    centers.data() + static_cast<size_t>(i)  * dim,
                    centers.data() + static_cast<size_t>(nb) * dim, dim);
                if (!std::isfinite(dist) || dist < MIN_EDGE_DIST) continue;
                f << i << ',' << nb << ',' << dist << '\n';
            }
        }

        // for (size_t i = 0; i < k_orig; i++) {
        //     // Low-level access to base layer
        //     auto bottom = h.index->get_linklist_at_level(i, 0);
        //     int nlinks = h.index->getListCount(bottom);
        //     hnswlib::tableint *links = (hnswlib::tableint *)(bottom + 1);

        //     // output edge weight == distance between i, j
        //     for (int j = 0; j < nlinks; j++) {
        //         // float dist = dist_func(centers[i].data(), centers[links[j]].data(), &dim);
        //         double dist = euclidean_distance(centers[i], centers[links[j]]);

        //         if (!std::isfinite(dist)) continue;
        //         if (dist < MIN_EDGE_DIST) continue;

        //         f << i << "," << links[j] << "," << dist << "\n";
        //     }
        // }
    
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << "\n"
        << "  --dataset              <key>   dataset key; base_file and runbook are read from\n"
        << "                                 the DATASETS registry in src/utils.cpp\n"
        << "  --centers              <k>     number of clusters (default: 1000)\n"
        << "  [--out-dir             <dir>]  output directory\n"
        << "  [--kmeans-n-init       <n>]    KMeans restarts, keep best inertia (default: 1)\n"
        << "                                 sklearn<=1.1 used n_init=10; >=1.2 k-means++ uses 1\n"
        << "                                 Use 10 to match old sklearn behaviour.\n"
        << "\n"
        << "  Diagnostic isolation flags (use together to skip step-1 assignment):\n"
        << "  [--initial-centers     <csv>]  Skip KMeans: load step-1 centres from this CSV.\n"
        << "                                 Alone this is NOT a true bypass (C++ re-runs\n"
        << "                                 assignment and recomputes centres, diverging from\n"
        << "                                 Python).  Use with --load-state-assignments for a\n"
        << "                                 true bypass that starts step 2 in identical state.\n"
        << "  [--load-state-assignments <csv>]\n"
        << "                                 Load Python's point_to_centroid as a CSV of\n"
        << "                                 'global_id,centroid' lines.  Must be used together\n"
        << "                                 with --initial-centers.  C++ will populate\n"
        << "                                 point_to_centroid and cluster_sums from these labels\n"
        << "                                 WITHOUT running any HNSW query, and will build the\n"
        << "                                 step-2 assignment HNSW from the loaded centres\n"
        << "                                 (identical to Python's step-2 HNSW).\n"
        << "                                 Generate the CSV from Python's .npy file:\n"
        << "                                   import numpy as np\n"
        << "                                   d = np.load('point_to_centroid.npy', allow_pickle=True).item()\n"
        << "                                   with open('point_to_centroid.csv','w') as f:\n"
        << "                                       for gid,cid in d.items(): f.write(f'{gid},{cid}\\n')\n"
        << "\n"
        << "  [--ef-search           <ef>]   HNSW ef for vector→centroid assignment (default: 200)\n"
        << "                                 Increase to reduce routing approximation errors\n"
        << "  [--vector-type         <type>] Override auto-detected element type.\n"
        << "                                 Values: f32 (default/.fbin), u8 (.u8bin), i8 (.i8bin)\n"
        << "                                 Detected automatically from the file extension.\n"
        << "  [--ignore-zero-counts]         suppress edges incident to zero-count nodes\n"
        << "  [--skip-zero-count-inserts]    skip zero-count nodes from HNSW construction\n"
        << "  [--verbose]                    show KMeans iteration output\n";
    std::exit(1);
}

int main(int argc, char* argv[]) {
    std::string runbook_path, dataset_key, vector_path, out_dir;
    std::string initial_centers_file;
    std::string load_state_assignments_file;
    std::string vector_type_override;
    int  k           = 1000;
    int  kmeans_ninit = 1;         // match sklearn>=1.2 default for k-means++
    int  ef_search   = HNSW_EF_SEARCH;
    int  mode        = 0;
    bool verbose     = false;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--dataset"                  && i + 1 < argc) dataset_key                 = argv[++i];
        else if (a == "--centers"                  && i + 1 < argc) k                           = std::stoi(argv[++i]);
        else if (a == "--out-dir"                  && i + 1 < argc) out_dir                     = argv[++i];
        else if (a == "--kmeans-n-init"            && i + 1 < argc) kmeans_ninit                = std::stoi(argv[++i]);
        else if (a == "--initial-centers"          && i + 1 < argc) initial_centers_file        = argv[++i];
        else if (a == "--load-state-assignments"   && i + 1 < argc) load_state_assignments_file = argv[++i];
        else if (a == "--ef-search"                && i + 1 < argc) ef_search                   = std::stoi(argv[++i]);
        else if (a == "--vector-type"              && i + 1 < argc) vector_type_override        = argv[++i];
        else if (a == "--ignore-zero-counts"       || a == "--ignore-zero-count")  mode = 1;
        else if (a == "--skip-zero-count-inserts"  || a == "--skip-zero-count-insert") mode = 2;
        else if (a == "--verbose")  verbose = true;
        else if (a == "--help" || a == "-h") usage(argv[0]);
    }

    if (!load_state_assignments_file.empty() && initial_centers_file.empty()) {
        std::cerr << "Error: --load-state-assignments requires --initial-centers "
                     "(the post-assignment centres from the Python step-1 output).\n";
        return 1;
    }

    if (dataset_key.empty()) {
        std::cerr << "Error: --dataset is required.\n";
        usage(argv[0]);
    }

    // Resolve the base vectors and runbook from the DATASETS registry (src/utils.cpp).
    if (auto it = DATASETS.find(dataset_key); it != DATASETS.end()) {
        const auto& cfg = it->second;
        if (auto f = cfg.find("base_file"); f != cfg.end()) vector_path  = f->second;
        if (auto f = cfg.find("runbook");   f != cfg.end()) runbook_path = f->second;
    }

    if (vector_path.empty() || runbook_path.empty()) {
        std::cerr << "Error: dataset key '" << dataset_key << "' is not in the DATASETS "
                     "registry (or is missing base_file/runbook). Add it to src/utils.cpp.\n";
        usage(argv[0]);
    }

    if (out_dir.empty())
        out_dir = "cluster_history_" + dataset_key + "_" + std::to_string(k);

    // ── Vector element type ───────────────────────────────────────────────────
    if (!vector_type_override.empty()) {
        if      (vector_type_override == "f32")  vtype = VecType::f32;
        else if (vector_type_override == "u8")   vtype = VecType::u8;
        else if (vector_type_override == "i8")   vtype = VecType::i8;
        else {
            std::cerr << "Error: unknown --vector-type '" << vector_type_override
                      << "' (expected f32, u8, or i8)\n";
            return 1;
        }
    }

    // ── Dataset info ─────────────────────────────────────────────────────────
    auto [total_pts, dim_u] = fbin_header(vector_path);
    const int dim = static_cast<int>(dim_u);
    std::cout << "Dataset : " << total_pts << " points, dim=" << dim
              << "  type=" << vec_type_name(vtype) << "\n";

    // ── Runbook ───────────────────────────────────────────────────────────────
    auto steps = load_runbook(runbook_path, dataset_key);
    if (steps.empty()) {
        std::cerr << "No steps loaded from runbook for key '" << dataset_key << "'\n";
        return 1;
    }
    std::cout << "Runbook : " << steps.size() << " steps loaded\n";
    std::cout << "Output  : " << out_dir << "\n";
    std::cout << "Centres : " << k << "\n";
    if (initial_centers_file.empty()) {
        std::cout << "KMeans  : n_init=" << kmeans_ninit
                  << " seed=" << KMEANS_SEED
                  << " max_iter=" << KMEANS_MAX_ITER << "\n";
    } else if (load_state_assignments_file.empty()) {
        std::cout << "KMeans  : BYPASSED — loading centres from " << initial_centers_file << "\n"
                  << "          (step-1 HNSW assignment will still run; "
                     "use --load-state-assignments for a true bypass)\n";
    } else {
        std::cout << "KMeans  : BYPASSED — loading centres from " << initial_centers_file << "\n"
                  << "Step-1  : BYPASSED — loading assignments from "
                  << load_state_assignments_file << "\n"
                  << "          (C++ will start step 2 in identical state to Python)\n";
    }
    std::cout << "HNSW    : M=" << HNSW_M << " ef_construction=" << HNSW_EF_CONSTRUCTION
              << " ef_search(assign)=" << ef_search << "\n";
    std::cout << "Mode    : " << (mode == 0 ? "all nodes" :
                                  mode == 1 ? "ignore-zero-counts" :
                                              "skip-zero-count-inserts") << "\n\n";

    fs::create_directories(out_dir);

    // ── Pipeline state ────────────────────────────────────────────────────────
    // cluster_sums[c*dim + d] = sum of component d over all currently-assigned
    //                           vectors in cluster c  (float64, same as Python)
    const size_t sums_size = static_cast<size_t>(k) * dim;
    std::vector<double>  cluster_sums(sums_size, 0.0);
    std::vector<int64_t> counts(k, 0LL);

    // point_to_centroid maps global vector ID → centroid index.
    // Using unordered_map mirrors Python's dict.
    std::unordered_map<uint32_t, int> point_to_centroid;
    point_to_centroid.reserve(static_cast<size_t>(total_pts));

    // HNSW built on updated centres after every update step.
    // Used for both vector assignment (step 3) and base-layer extraction (step 4).
    std::unique_ptr<HnswWrapper>  assign_hnsw;
    bool initialised = false;

    // ── Main loop ─────────────────────────────────────────────────────────────
    for (int step_id = 1; step_id <= static_cast<int>(steps.size()); ++step_id) {
        const auto& step = steps[step_id - 1];
        const auto& op   = step.operation;

        std::cout << "Step " << std::setw(3) << step_id << " | op=" << op;
        if (op == "insert" || op == "delete")
            std::cout << "  [" << step.start << ", " << step.end << ")";
        std::cout << "\n";

        // Search steps carry no vectors and do not change the index
        if (op == "search") continue;

        // Read vectors for this step
        auto vecs = read_vec_slice(vector_path,
                                    static_cast<uint32_t>(step.start),
                                    static_cast<uint32_t>(step.end),
                                    vtype);
        const int n_vecs = step.end - step.start;

        // ── INSERT ───────────────────────────────────────────────────────────
        if (op == "insert") {

            if (!initialised) {
                // ── First insert: KMeans (or loaded centres) → assign → update ─

                std::vector<float> init_centers;

                if (!initial_centers_file.empty()) {
                    // ── Diagnostic bypass: load centres from file ─────────────
                    // Use the Python pipeline's step_000001_centers.csv here to
                    // isolate whether divergence is in KMeans or HNSW routing.
                    std::cout << "  Loading initial centres from " << initial_centers_file << "\n";
                    init_centers = load_centers_csv(initial_centers_file, k, dim);
                } else {
                    // ── Normal path: run KMeans on a sample ──────────────────
                    const int sample_size = std::min(SAMPLE_LIMIT, n_vecs);
                    std::cout << "  Sampling " << sample_size << " / " << n_vecs << "\n";

                    std::vector<int> idx(n_vecs);
                    std::iota(idx.begin(), idx.end(), 0);
                    {
                        std::mt19937 sgen(static_cast<uint32_t>(KMEANS_SEED));
                        std::shuffle(idx.begin(), idx.end(), sgen);
                    }
                    idx.resize(sample_size);

                    std::vector<float> sample(static_cast<size_t>(sample_size) * dim);
                    for (int i = 0; i < sample_size; ++i)
                        std::copy(vecs.data() + static_cast<size_t>(idx[i]) * dim,
                                  vecs.data() + static_cast<size_t>(idx[i] + 1) * dim,
                                  sample.data() + static_cast<size_t>(i) * dim);

                    std::cout << "  Running KMeans(k=" << k << ", n_init=" << kmeans_ninit
                              << ", max_iter=" << KMEANS_MAX_ITER << ")...\n";
                    init_centers = run_kmeans(sample.data(), sample_size, dim, k,
                                             kmeans_ninit, KMEANS_SEED,
                                             KMEANS_MAX_ITER, KMEANS_TOL, verbose);
                    std::cout << "  KMeans done\n";
                }

                // Reset accumulators (they are zero-initialised, but be explicit)
                std::fill(cluster_sums.begin(), cluster_sums.end(), 0.0);
                std::fill(counts.begin(),       counts.end(),       0LL);

                if (!load_state_assignments_file.empty()) {
                    // ── Full step-1 bypass: load Python's assignments ─────────
                    // Populate point_to_centroid and cluster_sums from the saved
                    // labels, then build HNSW from the loaded (Python) centres
                    // without running any HNSW query or recomputing centres.
                    // This puts C++ in bit-exact identical state to Python after
                    // step 1, so divergence in step 2+ will reveal bugs in the
                    // incremental update logic rather than in the initial assignment.
                    auto loaded_asgn = load_assignments_csv(load_state_assignments_file);

                    for (int i = 0; i < n_vecs; ++i) {
                        uint32_t gid = static_cast<uint32_t>(step.start + i);
                        auto it = loaded_asgn.find(gid);
                        if (it == loaded_asgn.end())
                            throw std::runtime_error(
                                "load-state-assignments: no label for global_id " +
                                std::to_string(gid));
                        int lbl = it->second;
                        point_to_centroid[gid] = lbl;
                        const float* v = vecs.data() + static_cast<size_t>(i) * dim;
                        double* s      = cluster_sums.data() + static_cast<size_t>(lbl) * dim;
                        for (int d = 0; d < dim; ++d) s[d] += v[d];
                        ++counts[lbl];
                    }

                    // Build HNSW from the loaded (Python) centres — identical to
                    // the HNSW Python uses for step-2 assignment.
                    assign_hnsw = build_hnsw_all(init_centers.data(), k, dim,
                                                 HNSW_M, HNSW_EF_CONSTRUCTION, ef_search);
                    initialised = true;

                    // Export the step-1 label→centroid assignment (one centroid
                    // id per line, batch order; global id = step.start + line)
                    // so downstream tools (e.g. dynamic_runbook_experiment.cpp)
                    // can load this starting state.  Matches the debug dump
                    // produced for subsequent insert steps.
                    {
                        auto lpath = fs::path(out_dir) /
                            (step_prefix(step_id) + "_labels.csv");
                        std::ofstream lf(lpath);
                        for (int i = 0; i < n_vecs; ++i)
                            lf << point_to_centroid[static_cast<uint32_t>(step.start + i)] << '\n';
                        std::cout << "  [debug] Saved " << n_vecs
                                  << " labels → " << lpath.string() << "\n";
                    }

                } else {
                    // ── Normal first-insert path (or --initial-centers alone) ──
                    // Build HNSW on initial centres (before re-estimation from data)
                    assign_hnsw = build_hnsw_all(init_centers.data(), k, dim,
                                                 HNSW_M, HNSW_EF_CONSTRUCTION, ef_search);

                    // Assign ALL vectors in this batch to nearest centre via HNSW
                    // (matches Python: no batching on first insert)
                    auto labels = hnsw_query1(*assign_hnsw, vecs.data(), n_vecs);

                    // Export the step-1 label→centroid assignment (one centroid
                    // id per line, batch order; global id = step.start + line)
                    // so downstream tools (e.g. dynamic_runbook_experiment.cpp)
                    // can load this starting state.  Matches the debug dump
                    // produced for subsequent insert steps.  These labels reflect
                    // the assignment HNSW built from the initial centres, i.e. the
                    // exact point_to_centroid stored below.
                    {
                        auto lpath = fs::path(out_dir) /
                            (step_prefix(step_id) + "_labels.csv");
                        std::ofstream lf(lpath);
                        for (int i = 0; i < n_vecs; ++i) lf << labels[i] << '\n';
                        std::cout << "  [debug] Saved " << n_vecs
                                  << " labels → " << lpath.string() << "\n";
                    }

                    for (int i = 0; i < n_vecs; ++i) {
                        uint32_t gid = static_cast<uint32_t>(step.start + i);
                        int lbl = labels[i];
                        point_to_centroid[gid] = lbl;
                        const float* v = vecs.data() + static_cast<size_t>(i) * dim;
                        double* s      = cluster_sums.data() + static_cast<size_t>(lbl) * dim;
                        for (int d = 0; d < dim; ++d) s[d] += v[d];
                        ++counts[lbl];
                    }

                    // Re-estimate centres from actual assignments, then rebuild HNSW.
                    // This two-phase init (KMeans seed → assign → recompute → rebuild)
                    // exactly mirrors the Python code.
                    auto updated_centers = compute_centers(cluster_sums, counts, k, dim);
                    assign_hnsw = build_hnsw_all(updated_centers.data(), k, dim,
                                                 HNSW_M, HNSW_EF_CONSTRUCTION, ef_search);
                    initialised = true;
                }

            } else {
                // ── Subsequent insert: assign via HNSW in batches ─────────────
                std::vector<int> labels(n_vecs);
                for (int bs = 0; bs < n_vecs; bs += QUERY_BATCH_SIZE) {
                    int be = std::min(bs + QUERY_BATCH_SIZE, n_vecs);
                    int bn = be - bs;
                    auto bl = hnsw_query1(*assign_hnsw,
                                          vecs.data() + static_cast<size_t>(bs) * dim, bn);
                    std::copy(bl.begin(), bl.end(), labels.begin() + bs);
                }

                // Debug dump: save labels so we can diff against Python output
                {
                    auto lpath = fs::path(out_dir) /
                        (step_prefix(step_id) + "_labels.csv");
                    std::ofstream lf(lpath);
                    for (int i = 0; i < n_vecs; ++i) lf << labels[i] << '\n';
                    std::cout << "  [debug] Saved " << n_vecs
                              << " labels → " << lpath.string() << "\n";
                }

                // ── Accumulate per-cluster sums in float32, then add to float64
                // cluster_sums.
                //
                // Python's equivalent (lines 171-181 of the .py):
                //   sums_new = np.zeros_like(cluster_centers_)   # float64
                //   for c in range(k):
                //       members = new_vecs[labels == c]          # float32
                //       sums_new[c] = members.sum(axis=0)        # float32 sum
                //   cluster_sums += sums_new                     # float64 += promoted-float32
                //
                // members.sum(axis=0) returns float32 (numpy default for float32 input).
                // Promoting float32→float64 gives a different value than summing float64
                // individually (the float32 sum has rounding error that is then
                // "locked in" when promoted).  Using float64 individual additions
                // produces more accurate sums, but the ~1-ULP difference in some
                // centroid dimensions causes slightly different HNSW routing for
                // boundary vectors.  Over many insert steps those routing differences
                // accumulate in cluster_sums, which amplifies during delete phases
                // (fewer vectors → each centroid is more sensitive to wrong sums).
                //
                // Solution: replicate Python's float32-batch accumulation exactly.
                // ─────────────────────────────────────────────────────────────────
                {
                    // Per-cluster float32 batch sums (= Python's sums_new, stored as float32)
                    std::vector<float> sums_new(static_cast<size_t>(k) * dim, 0.0f);

                    for (int i = 0; i < n_vecs; ++i) {
                        uint32_t gid = static_cast<uint32_t>(step.start + i);
                        int lbl = labels[i];
                        point_to_centroid[gid] = lbl;
                        const float* v  = vecs.data()      + static_cast<size_t>(i)   * dim;
                        float*       sn = sums_new.data()  + static_cast<size_t>(lbl) * dim;
                        for (int d = 0; d < dim; ++d) sn[d] += v[d];   // float32 accumulation
                        ++counts[lbl];
                    }

                    // Add float32 batch sums to float64 cluster_sums
                    // (= Python's cluster_sums += sums_new, which promotes float32→float64
                    // element-by-element)
                    for (int c = 0; c < k; ++c) {
                        double*      s  = cluster_sums.data() + static_cast<size_t>(c) * dim;
                        const float* sn = sums_new.data()     + static_cast<size_t>(c) * dim;
                        for (int d = 0; d < dim; ++d)
                            s[d] += static_cast<double>(sn[d]);
                    }
                }

                auto updated_centers = compute_centers(cluster_sums, counts, k, dim);
                assign_hnsw = build_hnsw_all(updated_centers.data(), k, dim,
                                             HNSW_M, HNSW_EF_CONSTRUCTION, ef_search);
            }

        // ── DELETE ───────────────────────────────────────────────────────────
        } else if (op == "delete") {

            for (int i = 0; i < n_vecs; ++i) {
                uint32_t pid = static_cast<uint32_t>(step.start + i);
                auto it = point_to_centroid.find(pid);
                if (it == point_to_centroid.end())
                    throw std::runtime_error(
                        "Point " + std::to_string(pid) +
                        " not found in active set during deletion");

                int lbl = it->second;
                if (counts[lbl] < 0)
                    throw std::runtime_error(
                        "Cluster " + std::to_string(lbl) +
                        " has negative count during deletion");

                const float* v = vecs.data() + static_cast<size_t>(i) * dim;
                double* s      = cluster_sums.data() + static_cast<size_t>(lbl) * dim;
                for (int d = 0; d < dim; ++d) s[d] -= v[d];
                --counts[lbl];
                point_to_centroid.erase(it);
            }

            auto updated_centers = compute_centers(cluster_sums, counts, k, dim);
            assign_hnsw = build_hnsw_all(updated_centers.data(), k, dim,
                                         HNSW_M, HNSW_EF_CONSTRUCTION, ef_search);

        } else {
            std::cerr << "  Unknown operation '" << op << "', skipping.\n";
            continue;
        }

        // ── Save step outputs ─────────────────────────────────────────────────
        auto centers = compute_centers(cluster_sums, counts, k, dim);

        // Step 3 outputs (for runbook_partitions_parallel.cpp)
        save_centers_csv(out_dir, step_id, centers, k, dim);
        save_counts_csv(out_dir, step_id, counts, k);

        // Step 4 outputs (HNSW binary for recall eval; base layer for step 5)
        save_hnsw_bin(out_dir, step_id, *assign_hnsw);

        if (mode == 2) {
            auto [skip_hnsw, label_map] =
                build_hnsw_skip_zero(centers.data(), counts, k, dim);
            save_base_layer_csv(out_dir, step_id, *skip_hnsw,
                                centers, counts, k, dim, mode, label_map);
        } else {
            static const std::vector<int> empty_map;
            save_base_layer_csv(out_dir, step_id, *assign_hnsw,
                                centers, counts, k, dim, mode, empty_map);
        }

        std::cout << "  Saved " << step_prefix(step_id)
                  << "  active=" << point_to_centroid.size() << "\n";
    }

    std::cout << "\nDone. Output in: " << out_dir << "\n";
    return 0;
}
