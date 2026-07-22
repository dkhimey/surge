// runbook_centers.cpp
// Replays a streaming runbook and produces per-step routing state (centres, HNSW index, etc.).
// Output per step: step_NNNNNN_{centers.csv, center_counts.csv, hnsw.bin,
// base_layer.csv}. KMeans and HNSW assignment are seeded (mt19937(42), hnswlib
// seed 100) so runs are reproducible; see --help for the full flag list.
//
// Usage:  ./bin/runbook_centers --dataset <dataset> --centers 10000 \
//             --out-dir cluster_history_<dataset>_10000
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

// Constants
constexpr double MIN_EDGE_DIST      = 1e-4;      // Must match hnsw_per_step.cpp
constexpr int    KMEANS_MAX_ITER    = 300;       // Match sklearn defaults
constexpr float  KMEANS_TOL         = 1e-4f;     // Squared centroid shift threshold
constexpr int    KMEANS_SEED        = 42;
constexpr int    SAMPLE_LIMIT       = 100000;
constexpr int    HNSW_M             = 16;        // Must match Python parameters
constexpr int    HNSW_EF_CONSTRUCTION = 200;
constexpr int    HNSW_EF_SEARCH     = 200;
constexpr int    QUERY_BATCH_SIZE   = 100000;


// Read header {num_points, dim} from binary file
static std::pair<uint32_t, uint32_t> fbin_header(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + filename);
    uint32_t n, d;
    f.read(reinterpret_cast<char*>(&n), sizeof n);
    f.read(reinterpret_cast<char*>(&d), sizeof d);
    if (!f) throw std::runtime_error("Failed to read header: " + filename);
    return {n, d};
}


static std::vector<float> read_vec_slice(const std::string& filename,
                                          uint32_t start, uint32_t end)
{
    auto [num_points, dim] = fbin_header(filename);
    uint32_t count = end - start;
    if (end > num_points)
        throw std::runtime_error(
            "read_vec_slice: [" + std::to_string(start) + "," +
            std::to_string(end) + ") exceeds " + std::to_string(num_points));
    return readVecs(filename, dim, static_cast<int>(count), static_cast<int>(start));
}

// Run one KMeans++ initialization + Lloyd's iterations; return centres and inertia
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
    // k-means++ initialization
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

    // Lloyd's iterations
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

    // Compute inertia (sum of squared distances)
    double inertia = 0.0;
    for (int i = 0; i < n_points; ++i)
        inertia += static_cast<double>(
            computeEuclideanDistance(sample + static_cast<size_t>(i) * dim,
                                     centers.data() + static_cast<size_t>(labels[i]) * dim, dim));

    return {std::move(centers), inertia};
}

// Run k-means++ n_init times; return centres with lowest inertia
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


// Load initial centres from CSV file (for diagnosis/bypass of C++ KMeans)
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


// Load point_to_centroid assignments from CSV (one line per vector: global_id,centroid_index)
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


// HNSW wrapper with parameters matching Python _build_hnsw
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


// Compute float32 centres from double accumulators
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


static std::string step_prefix(int step) {
    std::ostringstream oss;
    oss << "step_" << std::setw(6) << std::setfill('0') << step;
    return oss.str();
}

// Save centres CSV in scientific notation (matches Python np.savetxt)
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

// Save counts as plain decimal integers (not scientific notation) to avoid stoi truncation
static void save_counts_csv(const fs::path& out_dir, int step,
                              const std::vector<int64_t>& counts, int k)
{
    auto path = out_dir / (step_prefix(step) + "_center_counts.csv");
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path.string());
    for (int c = 0; c < k; ++c)
        f << static_cast<int32_t>(counts[c]) << '\n';
}

// Save HNSW index binary
static void save_hnsw_bin(const fs::path& out_dir, int step,
                           const HnswWrapper& h)
{
    auto path = out_dir / (step_prefix(step) + "_hnsw.bin");
    h.index->saveIndex(path.string());
}

// Euclidean distance (not squared)
static double euclidean_distance_d(const float* a, const float* b, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
        double diff = a[i] - b[i];
        s += diff * diff;
    }
    return std::sqrt(s);
}

// Save base-layer edge list CSV
// mode 0: all nodes  |  mode 1: suppress zero-count edges  |  mode 2: skip zero-count nodes
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
    }
}


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

    auto [total_pts, dim_u] = fbin_header(vector_path);
    const int dim = static_cast<int>(dim_u);
    std::cout << "Dataset : " << total_pts << " points, dim=" << dim << "\n";

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

    // Pipeline state
    const size_t sums_size = static_cast<size_t>(k) * dim;
    std::vector<double>  cluster_sums(sums_size, 0.0);
    std::vector<int64_t> counts(k, 0LL);
    std::unordered_map<uint32_t, int> point_to_centroid;
    point_to_centroid.reserve(static_cast<size_t>(total_pts));
    std::unique_ptr<HnswWrapper>  assign_hnsw;
    bool initialised = false;

    for (int step_id = 1; step_id <= static_cast<int>(steps.size()); ++step_id) {
        const auto& step = steps[step_id - 1];
        const auto& op   = step.operation;

        std::cout << "Step " << std::setw(3) << step_id << " | op=" << op;
        if (op == "insert" || op == "delete")
            std::cout << "  [" << step.start << ", " << step.end << ")";
        std::cout << "\n";

        if (op == "search") continue;

        auto vecs = read_vec_slice(vector_path,
                                    static_cast<uint32_t>(step.start),
                                    static_cast<uint32_t>(step.end));
        const int n_vecs = step.end - step.start;

        if (op == "insert") {

            if (!initialised) {
                std::vector<float> init_centers;

                if (!initial_centers_file.empty()) {
                    std::cout << "  Loading initial centres from " << initial_centers_file << "\n";
                    init_centers = load_centers_csv(initial_centers_file, k, dim);
                } else {
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

                std::fill(cluster_sums.begin(), cluster_sums.end(), 0.0);
                std::fill(counts.begin(),       counts.end(),       0LL);

                if (!load_state_assignments_file.empty()) {
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

                    assign_hnsw = build_hnsw_all(init_centers.data(), k, dim,
                                                 HNSW_M, HNSW_EF_CONSTRUCTION, ef_search);
                    initialised = true;
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
                    assign_hnsw = build_hnsw_all(init_centers.data(), k, dim,
                                                 HNSW_M, HNSW_EF_CONSTRUCTION, ef_search);

                    auto labels = hnsw_query1(*assign_hnsw, vecs.data(), n_vecs);
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

                    auto updated_centers = compute_centers(cluster_sums, counts, k, dim);
                    assign_hnsw = build_hnsw_all(updated_centers.data(), k, dim,
                                                 HNSW_M, HNSW_EF_CONSTRUCTION, ef_search);
                    initialised = true;
                }

            } else {
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

                // Replicate Python's float32-batch accumulation for numerical precision
                {
                    std::vector<float> sums_new(static_cast<size_t>(k) * dim, 0.0f);

                    for (int i = 0; i < n_vecs; ++i) {
                        uint32_t gid = static_cast<uint32_t>(step.start + i);
                        int lbl = labels[i];
                        point_to_centroid[gid] = lbl;
                        const float* v  = vecs.data()      + static_cast<size_t>(i)   * dim;
                        float*       sn = sums_new.data()  + static_cast<size_t>(lbl) * dim;
                        for (int d = 0; d < dim; ++d) sn[d] += v[d];
                        ++counts[lbl];
                    }

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

        auto centers = compute_centers(cluster_sums, counts, k, dim);
        save_centers_csv(out_dir, step_id, centers, k, dim);
        save_counts_csv(out_dir, step_id, counts, k);
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
