#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <limits>
#include <yaml-cpp/yaml.h>
#include <omp.h>
#include "utils.h"
#include "hnswlib.h"

constexpr double MIN_EDGE_DIST = 1e-4;

// ---------------------------------------------------------------------------
// Exact nearest-centroid assignment
//
// For each query vector x, compute squared L2 to every center using the
// identity:  ||x - c||² = ||x||² + ||c||² - 2·(x·c)
//
// c2[k] = ||center_k||² is precomputed once per center-update and reused
// across all vectors, keeping it hot in L1/L2.
//
// Vectors are processed in tiles of ASSIGN_TILE so that the centers block
// (K × dim floats) stays resident in L3 while we stream through the tile.
// OpenMP distributes tiles across cores.
// ---------------------------------------------------------------------------
constexpr int ASSIGN_TILE = 64;   // vectors per tile  (~25 KB working set for K=1000,dim=100)

// Recompute c2[k] = ||center_k||² for all k.
// Called once after centers are updated; O(K·D), negligible.
static void recompute_c2(const float* centers, int num_centers, int dim,
                         float* c2)
{
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < num_centers; c++) {
        const float* cp = centers + c * dim;
        float s = 0.0f;
        for (int d = 0; d < dim; d++) s += cp[d] * cp[d];
        c2[c] = s;
    }
}

// Recompute center positions from cluster_sums / center_counts.
// Called after every insert/delete reduction.
static void recompute_centers(const double* cluster_sums,
                              const int*    center_counts,
                              int num_centers, int dim,
                              float* centers)
{
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < num_centers; c++) {
        int divisor = std::max(center_counts[c], 1);
        const double* cs = cluster_sums + c * dim;
        float*        cp = centers       + c * dim;
        for (int d = 0; d < dim; d++)
            cp[d] = static_cast<float>(cs[d] / divisor);
    }
}

// ---------------------------------------------------------------------------
// Assign a contiguous block of `n` vectors (starting at new_vecs[0]) to
// their nearest center.  Writes results into `out_labels[0..n)`.
//
// Thread-private accumulators (thread_sums, thread_counts) are updated
// in-place so the caller can reduce them after this call.
// ---------------------------------------------------------------------------
static void assign_exact(
    const float* __restrict__ new_vecs,
    int n, int dim,
    const float* __restrict__ centers,
    const float* __restrict__ c2,
    int num_centers,
    int* __restrict__ out_labels,
    int num_threads,
    double* __restrict__ thread_center_sums,    // [num_threads * num_centers * dim]
    int*    __restrict__ thread_center_counts)  // [num_threads * num_centers]
{
    #pragma omp parallel for schedule(dynamic, ASSIGN_TILE)
    for (int tile_start = 0; tile_start < n; tile_start += ASSIGN_TILE) {
        int tile_end  = std::min(tile_start + ASSIGN_TILE, n);
        int tid       = omp_get_thread_num();
        double* ts    = thread_center_sums   + tid * num_centers * dim;
        int*    tc    = thread_center_counts + tid * num_centers;

        for (int i = tile_start; i < tile_end; i++) {
            const float* x = new_vecs + i * dim;

            // Compute ||x||²
            float x2 = 0.0f;
            for (int d = 0; d < dim; d++) x2 += x[d] * x[d];

            // Find nearest center via exact L2² = x2 + c2[c] - 2·dot(x, c)
            float best_dist = std::numeric_limits<float>::max();
            int   best_c    = 0;
            for (int c = 0; c < num_centers; c++) {
                const float* cp = centers + c * dim;
                float dot = 0.0f;
                // Inner dim loop is contiguous for both x and cp;
                // compiler auto-vectorises with AVX2/AVX-512.
                for (int d = 0; d < dim; d++) dot += x[d] * cp[d];
                float dist = x2 + c2[c] - 2.0f * dot;
                if (dist < best_dist) { best_dist = dist; best_c = c; }
            }

            out_labels[i] = best_c;

            // Accumulate into thread-private sums (no atomics needed)
            double* tsc = ts + best_c * dim;
            for (int d = 0; d < dim; d++) tsc[d] += x[d];
            tc[best_c]++;
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce per-thread accumulators into cluster_sums / center_counts,
// then recompute center positions and c2.
// ---------------------------------------------------------------------------
static void reduce_and_recompute(
    int num_centers, int dim, int num_threads,
    const double* thread_center_sums,
    const int*    thread_center_counts,
    double* cluster_sums,
    int*    center_counts,
    float*  centers,
    float*  c2,
    bool subtract = false)
{
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < num_centers; c++) {
        int   delta = 0;
        double* cs  = cluster_sums + c * dim;
        for (int t = 0; t < num_threads; t++) {
            delta += thread_center_counts[t * num_centers + c];
            const double* ts = thread_center_sums + (t * num_centers + c) * dim;
            for (int d = 0; d < dim; d++)
                cs[d] += subtract ? -ts[d] : ts[d];
        }
        if (subtract) {
            int clamped = std::min(delta, center_counts[c]);
            center_counts[c] -= clamped;
        } else {
            center_counts[c] += delta;
        }
    }
    recompute_centers(cluster_sums, center_counts, num_centers, dim, centers);
    recompute_c2(centers, num_centers, dim, c2);
}

// ---------------------------------------------------------------------------
// Build HNSW on the current centers and extract its base layer.
// Only called at save/query steps — not after every insert/delete.
// ---------------------------------------------------------------------------
static void build_and_save(
    const std::string& output_dir,
    int stepNum,
    int num_centers, int dim,
    const float*  centers,
    const int*    center_counts,
    const int*    vecid_to_centerid,
    int           nvectors,
    hnswlib::L2Space& space)
{
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << stepNum;
    std::string step_str = oss.str();

    auto t0 = std::chrono::high_resolution_clock::now();

    // Build HNSW — only at save steps (not every insert/delete)
    hnswlib::HierarchicalNSW<float> hnsw(&space, num_centers);
    // addPoint is thread-safe in hnswlib
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_centers; i++)
        hnsw.addPoint(centers + i * dim, i);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  HNSW build time: "
              << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    // --- Write centers CSV ---
    {
        std::ofstream f(output_dir + "/step_" + step_str + "_centers.csv");
        for (int c = 0; c < num_centers; c++) {
            const float* cp = centers + c * dim;
            for (int d = 0; d < dim; d++) {
                f << cp[d];
                if (d < dim - 1) f << ',';
            }
            f << '\n';
        }
    }

    // --- Write counts CSV ---
    {
        std::ofstream f(output_dir + "/step_" + step_str + "_center_counts.csv");
        for (int c = 0; c < num_centers; c++)
            f << center_counts[c] << '\n';
    }

    // --- Write vecid->centerid mapping (binary) ---
    {
        std::ofstream f(output_dir + "/step_" + step_str + "_vecid_to_centerid.bin",
                        std::ios::binary);
        f.write(reinterpret_cast<const char*>(vecid_to_centerid),
                nvectors * sizeof(int));
    }

    // --- Save HNSW index ---
    hnsw.saveIndex(output_dir + "/step_" + step_str + "_hnsw.bin");

    // --- Extract and save base layer ---
    {
        std::ofstream f(output_dir + "/step_" + step_str + "_base_layer.csv");
        for (int internal_id = 0; internal_id < num_centers; internal_id++) {
            auto label_i  = hnsw.getExternalLabel(internal_id);
            auto bottom   = hnsw.get_linklist_at_level(internal_id, 0);
            int  nlinks   = hnsw.getListCount(bottom);
            auto* links   = reinterpret_cast<hnswlib::tableint*>(bottom + 1);
            const float* vi = centers + label_i * dim;
            for (int j = 0; j < nlinks; j++) {
                auto label_j  = hnsw.getExternalLabel(links[j]);
                const float* vj = centers + label_j * dim;
                float dist = computeEuclideanDistance(vi, vj, dim);
                if (!std::isfinite(dist) || dist < MIN_EDGE_DIST) continue;
                f << label_i << ',' << label_j << ',' << dist << '\n';
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <num_centers> <output_dir>\n";
        return 1;
    }

    const std::string dataset_name = argv[1];
    const int         num_centers  = std::stoi(argv[2]);
    const std::string output_dir   = argv[3];

    {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            std::cerr << "Failed to create output directory '" << output_dir
                      << "': " << ec.message() << "\n";
            return 1;
        }
    }

    if (DATASETS.find(dataset_name) == DATASETS.end()) {
        std::cerr << "Unknown dataset: " << dataset_name << "\n";
        return 1;
    }

    const std::string base_file    = DATASETS[dataset_name]["base_file"];
    const std::string runbook_file = DATASETS[dataset_name]["runbook"];

    auto [nvectors, dim] = get_dataset_info(base_file);
    std::cout << "Dataset: " << dataset_name
              << "  nvectors=" << nvectors << "  dim=" << dim
              << "  num_centers=" << num_centers << "\n";

    // --- Load and sort runbook steps numerically ---
    // yaml-cpp iterates map keys in insertion / lexicographic order, which
    // gives wrong ordering for multi-digit step numbers ("10" < "2").
    YAML::Node root = YAML::LoadFile(runbook_file);
    YAML::Node yaml_steps = root[dataset_name];
    if (!yaml_steps) {
        std::cerr << "Key '" << dataset_name << "' not found in runbook.\n";
        return 1;
    }

    std::vector<std::pair<int, YAML::Node>> sorted_steps;
    sorted_steps.reserve(yaml_steps.size());
    for (const auto& kv : yaml_steps) {
        try {
            int n = std::stoi(kv.first.as<std::string>());
            sorted_steps.emplace_back(n, kv.second);
        } catch (...) {
            std::cerr << "Skipping non-numeric key: "
                      << kv.first.as<std::string>() << "\n";
        }
    }

    // --- Persistent state ---
    std::vector<int>    vecid_to_centerid(nvectors, -1);
    std::vector<float>  centers(num_centers * dim, 0.0f);
    std::vector<float>  c2(num_centers, 0.0f);      // ||center_k||², kept in sync
    std::vector<double> cluster_sums(num_centers * dim, 0.0);
    std::vector<int>    center_counts(num_centers, 0);
    bool                initialized = false;

    hnswlib::L2Space space(dim);

    // Hoist thread-private accumulator buffers (large; reused every step)
    const int num_threads = omp_get_max_threads();
    std::cout << "OpenMP threads: " << num_threads << "\n";
    std::vector<double> thread_center_sums(num_threads * num_centers * dim, 0.0);
    std::vector<int>    thread_center_counts(num_threads * num_centers, 0);

    // Temporary label buffer (reused across insert steps)
    std::vector<int> labels;

    // --- Step loop ---
    for (auto& [stepNum, step] : sorted_steps) {
        if (!step.IsMap() || step.size() == 0) {
            std::cout << "Skipping invalid step " << stepNum << "\n";
            continue;
        }

        const std::string op = step["operation"].as<std::string>();
        std::cout << "\nStep " << stepNum << " [" << op << "]\n";

        // ----------------------------------------------------------------
        if (op == "insert") {
            const int start = step["start"].as<int>();
            const int end   = step["end"].as<int>();
            const int n     = end - start;
            std::cout << "  range=[" << start << ", " << end << ")  n=" << n << "\n";

            auto io_t0 = std::chrono::high_resolution_clock::now();
            std::vector<float> new_vecs = readVecs(base_file, dim, n, start);
            auto io_t1 = std::chrono::high_resolution_clock::now();
            std::cout << "  I/O time: "
                      << std::chrono::duration<double>(io_t1 - io_t0).count() << "s\n";

            if (!initialized) {
                // ---- First insert: run k-means on a sample to seed centers ----
                const int sample_size = std::min(n, 100000);
                std::vector<size_t> idx(n);
                std::iota(idx.begin(), idx.end(), 0);
                std::shuffle(idx.begin(), idx.end(),
                             std::mt19937{std::random_device{}()});
                idx.resize(sample_size);

                std::vector<float> sample(sample_size * dim);
                for (int i = 0; i < sample_size; i++)
                    std::copy(new_vecs.data() + idx[i] * dim,
                              new_vecs.data() + (idx[i] + 1) * dim,
                              sample.data() + i * dim);

                // kmeans populates centers; reset counts/sums before full assignment below
                kmeans(sample.data(), sample_size, dim, num_centers,
                       centers.data(), center_counts.data());
                center_counts.assign(num_centers, 0);
                cluster_sums.assign(num_centers * dim, 0.0);

                // Precompute c2 for the seeded centers
                recompute_c2(centers.data(), num_centers, dim, c2.data());
                initialized = true;
            }

            // ---- Exact assignment + accumulation (parallel, tiled) ----
            labels.resize(n);
            std::fill(thread_center_sums.begin(),   thread_center_sums.end(),   0.0);
            std::fill(thread_center_counts.begin(), thread_center_counts.end(), 0);

            auto assign_t0 = std::chrono::high_resolution_clock::now();
            assign_exact(new_vecs.data(), n, dim,
                         centers.data(), c2.data(), num_centers,
                         labels.data(),
                         num_threads,
                         thread_center_sums.data(),
                         thread_center_counts.data());
            auto assign_t1 = std::chrono::high_resolution_clock::now();
            std::cout << "  Assign time: "
                      << std::chrono::duration<double>(assign_t1 - assign_t0).count() << "s\n";

            // Write vecid->centerid for every vector in this batch
            // (each index is unique, so no race)
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
                vecid_to_centerid[start + i] = labels[i];

            // ---- Reduce thread accumulators, recompute centers + c2 ----
            auto reduce_t0 = std::chrono::high_resolution_clock::now();
            reduce_and_recompute(num_centers, dim, num_threads,
                                 thread_center_sums.data(),
                                 thread_center_counts.data(),
                                 cluster_sums.data(),
                                 center_counts.data(),
                                 centers.data(), c2.data(),
                                 /*subtract=*/false);
            auto reduce_t1 = std::chrono::high_resolution_clock::now();
            std::cout << "  Reduce+recompute time: "
                      << std::chrono::duration<double>(reduce_t1 - reduce_t0).count() << "s\n";
            std::cout << "  Total insert step time: "
                      << std::chrono::duration<double>(reduce_t1 - assign_t0).count() << "s\n";

        // ----------------------------------------------------------------
        } else if (op == "delete") {
            const int start = step["start"].as<int>();
            const int end   = step["end"].as<int>();
            const int n     = end - start;
            std::cout << "  range=[" << start << ", " << end << ")  n=" << n << "\n";

            if (!initialized) {
                std::cerr << "  Warning: delete before any insert; skipping.\n";
                continue;
            }

            auto io_t0 = std::chrono::high_resolution_clock::now();
            std::vector<float> del_vecs = readVecs(base_file, dim, n, start);
            auto io_t1 = std::chrono::high_resolution_clock::now();
            std::cout << "  I/O time: "
                      << std::chrono::duration<double>(io_t1 - io_t0).count() << "s\n";

            std::fill(thread_center_sums.begin(),   thread_center_sums.end(),   0.0);
            std::fill(thread_center_counts.begin(), thread_center_counts.end(), 0);

            auto del_t0 = std::chrono::high_resolution_clock::now();
            // Deletion uses the stored vecid->centerid mapping — no search needed.
            // Each vec_id in [start, end) is unique within this step.
            #pragma omp parallel for schedule(dynamic, ASSIGN_TILE)
            for (int i = 0; i < n; i++) {
                const int vec_id = start + i;
                if (vec_id < 0 || vec_id >= nvectors) continue;

                const int center_id = vecid_to_centerid[vec_id];
                if (center_id < 0 || center_id >= num_centers) continue;

                int    tid = omp_get_thread_num();
                int*    tc = thread_center_counts.data() + tid * num_centers + center_id;
                double* ts = thread_center_sums.data()   + (tid * num_centers + center_id) * dim;
                const float* dv = del_vecs.data() + i * dim;
                for (int d = 0; d < dim; d++) ts[d] += dv[d];
                (*tc)++;

                vecid_to_centerid[vec_id] = -1;  // tombstone; unique per step
            }
            auto del_t1 = std::chrono::high_resolution_clock::now();
            std::cout << "  Delete accumulate time: "
                      << std::chrono::duration<double>(del_t1 - del_t0).count() << "s\n";

            auto reduce_t0 = std::chrono::high_resolution_clock::now();
            reduce_and_recompute(num_centers, dim, num_threads,
                                 thread_center_sums.data(),
                                 thread_center_counts.data(),
                                 cluster_sums.data(),
                                 center_counts.data(),
                                 centers.data(), c2.data(),
                                 /*subtract=*/true);
            auto reduce_t1 = std::chrono::high_resolution_clock::now();
            std::cout << "  Reduce+recompute time: "
                      << std::chrono::duration<double>(reduce_t1 - reduce_t0).count() << "s\n";
            std::cout << "  Total delete step time: "
                      << std::chrono::duration<double>(reduce_t1 - del_t0).count() << "s\n";

        // ----------------------------------------------------------------
        } else {
            // Save / query step — build HNSW here (and only here)
            if (!initialized) {
                std::cerr << "  Warning: save before any insert; skipping.\n";
                continue;
            }
            build_and_save(output_dir, stepNum,
                           num_centers, dim,
                           centers.data(), center_counts.data(),
                           vecid_to_centerid.data(), nvectors,
                           space);
        }
    }

    return 0;
}