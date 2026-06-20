// shared_batch_experiment_sweep.cpp
//
// Variant of shared_batch_experiment that sweeps all three routing modes and
// their full parameter grids during search steps rather than accepting a single
// mode/param on the command line.  Insert, delete, and rebuild steps run once
// as normal; each search step runs once per (mode, param) combination and
// records throughput + recall separately for each combination.
//
// ─── Usage ───────────────────────────────────────────────────────────────────
//  mpirun -np <P+1> ./shared_batch_experiment_sweep \
//      <dataset> <num_partitions> <full_threshold> <k> <gt_prefix> <output_file>
//
//  Rebuild policy: delta always — shadow-delete departing elements and insert
//  arriving ones using replace_deleted=true (keeps existing graph topology).
//  Falls back to a full reconstruction if any shard's tombstone ratio exceeds
//  TOMBSTONE_RATIO_THRESHOLD (currently 0.5 = 50% unused slots).  This check
//  fires independently of the center-movement rebuild threshold: a shard
//  exceeding the ratio forces a full rebuild even when checkNeedRebuild() would
//  otherwise report that no rebuild is needed.
//
// ─── Sweep parameter grids ───────────────────────────────────────────────────
//  BranchingFactor : {20, 25, 30, 40, 50, 60, 80}
//  NProbe          : {5, 6, 7, 8, 9}
//  RecallTarget    : {0.85, 0.90, 0.95, 0.97, 0.98, 0.99}
//
// ─── Output CSV columns ──────────────────────────────────────────────────────
//  All rows:
//    step, operation, mode, param,
//    range_start, range_end, time_s, throughput,
//    recall@<k>, theoretical_recall@<k>, avg_parts_searched
//
//  theoretical_recall@<k> is computed during the sweep (search rows only;
//  -1 elsewhere).  For each query it equals
//    | gt_partitions(q) ∩ visited_partitions(q, mode, param) | / | gt_partitions(q) |
//  where gt_partitions(q) is the set of partitions holding q's top-K GT
//  neighbours, looked up via a fresh routing_hnsw->searchKnn(gt_vec, 1) pass
//  — matching compute_theoretical_recall_updated.py exactly.  Computed
//  outside the timed search window so throughput is unaffected.
//  Rebuild rows (operation="rebuild") additionally populate:
//    centers_moved, elements_moved,
//    repart_hnsw_s, repart_bottom_s, repart_kaffpa_s, repart_relabel_s,
//    exec_iterate_s, exec_exchange_s, exec_graph_s, dorebuild_wall_s
//  (non-rebuild rows carry -1 in those rebuild-specific columns)
//  All rows end with shard_0_size, …, shard_<P-1>_size
//  Rebuild rows additionally populate remaining_deleted_slots
//  (-1 for full rebuilds; 0+ for delta rebuilds).
//
//  For non-search rows, mode="" and param=-1.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

// POSIX mmap headers — used by BaseMmap below to sparse-read GT vectors
// from base_file for theoretical-recall computation (matches the offline
// Python script's np.memmap pattern).
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>
#include <unordered_set>

#include "index.h"

// ─── mmap helper for sparse GT vector reads ─────────────────────────────────
//
// Mirrors compute_theoretical_recall_updated.py's mmap_fbin / mmap_u8bin:
// open base_file, mmap it read-only, and copy individual vectors out at
// arbitrary indices into a float buffer.  Supports FBIN (float32) and
// U8BIN (uint8 → float32 widen).  Throws on other formats.
//
// Used only for theoretical-recall setup (once per search step, outside
// the timed query window), so per-vector copy cost is irrelevant.
class BaseMmap {
public:
    BaseMmap(const std::string& path, FileFormat format)
        : format_(format)
    {
        if (format != FBIN && format != U8BIN) {
            throw std::runtime_error(
                "BaseMmap: only FBIN/U8BIN supported for theoretical recall");
        }
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("BaseMmap: cannot open " + path);
        struct stat st;
        if (fstat(fd_, &st) != 0) {
            close(fd_);
            throw std::runtime_error("BaseMmap: fstat failed for " + path);
        }
        size_ = static_cast<size_t>(st.st_size);
        base_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("BaseMmap: mmap failed for " + path);
        }
        const uint32_t* hdr = static_cast<const uint32_t*>(base_);
        n_   = hdr[0];
        dim_ = hdr[1];
    }
    ~BaseMmap() {
        if (base_ && base_ != MAP_FAILED) munmap(base_, size_);
        if (fd_ >= 0) close(fd_);
    }
    BaseMmap(const BaseMmap&)            = delete;
    BaseMmap& operator=(const BaseMmap&) = delete;

    uint32_t n()   const { return n_; }
    uint32_t dim() const { return dim_; }

    // Copy vector at index `id` into `out` (resized to dim()).  Widens
    // u8 → float on the fly.
    void readVector(uint32_t id, std::vector<float>& out) const {
        out.resize(dim_);
        const char* payload = static_cast<const char*>(base_) + 8;
        if (format_ == FBIN) {
            const float* p = reinterpret_cast<const float*>(payload)
                           + static_cast<size_t>(id) * dim_;
            std::memcpy(out.data(), p, dim_ * sizeof(float));
        } else { // U8BIN
            const uint8_t* p = reinterpret_cast<const uint8_t*>(payload)
                             + static_cast<size_t>(id) * dim_;
            for (size_t i = 0; i < dim_; ++i)
                out[i] = static_cast<float>(p[i]);
        }
    }

private:
    int        fd_     = -1;
    void*      base_   = nullptr;
    size_t     size_   = 0;
    uint32_t   n_      = 0;
    uint32_t   dim_    = 0;
    FileFormat format_;
};

// ─── Build hyper-parameters ──────────────────────────────────────────────────
static constexpr int    NCENTERS             = 10000;
static constexpr int    EF_CONSTRUCTION      = 200;
static constexpr int    M_META               = 16;
static constexpr int    M_SUB                = 16;
static constexpr int    NUM_BUILDING_THREADS = 32;
static constexpr int    EF_SEARCH            = 200;
// Meta-HNSW (routing) search ef.  Kept separate from EF_SEARCH (the executor
// sub-index search ef) so routing effort can be tuned without touching local
// search quality.  Set to 100 to match compute_theoretical_recall_updated.py's
// router.set_ef(max(100, k_rt)), so the theoretical-recall column produced here
// is computed with the same routing accuracy as the offline theoretical pipeline.
static constexpr int    EF_ROUTING           = 100;
static constexpr size_t SAMPLE_SIZE          = 100000;
// If the maximum tombstone ratio across all executor shards meets or exceeds
// this threshold during a delta-rebuild event, a full rebuild is used instead.
static constexpr double TOMBSTONE_RATIO_THRESHOLD = 0.5;
// Write a checkpoint every this many runbook steps.  The new checkpoint is
// fully written and fsync'd (via close()) before the previous one is removed,
// so disk space is never exhausted by holding two copies simultaneously in the
// typical case (unless a single checkpoint exceeds half the available space).
static constexpr size_t CHECKPOINT_INTERVAL       = 50;

// ─── Sweep parameter grids (visible to all ranks so loop counts agree) ───────
static const std::vector<int>   BRANCHING_FACTOR_PARAMS = {20, 40, 60, 80};
static const std::vector<int>   NPROBE_PARAMS           = {3, 5, 6, 7, 8, 9};
static const std::vector<float> TARGET_PARAMS           = {.90f, .95f, .97f, .99f};

// Build the ordered (mode, param) list once.  Both coordinator and executor
// iterate this same list so their MPI collective calls stay in lockstep.
static std::vector<std::pair<RoutingMode, float>> build_sweep_combos()
{
    std::vector<std::pair<RoutingMode, float>> combos;
    for (int v : BRANCHING_FACTOR_PARAMS)
        combos.push_back({RoutingMode::BranchingFactor, static_cast<float>(v)});
    for (int v : NPROBE_PARAMS)
        combos.push_back({RoutingMode::NProbe, static_cast<float>(v)});
    for (float v : TARGET_PARAMS)
        combos.push_back({RoutingMode::RecallTarget, v});
    return combos;
}

static std::string mode_to_string(RoutingMode m)
{
    switch (m) {
        case RoutingMode::BranchingFactor: return "BranchingFactor";
        case RoutingMode::NProbe:          return "NProbe";
        case RoutingMode::RecallTarget:    return "RecallTarget";
    }
    return "Unknown";
}

// ─── Runbook ──────────────────────────────────────────────────────────────────
struct RunbookStep {
    int         step_num  = -1;
    std::string operation;
    int         start     = -1;
    int         end       = -1;
};

static std::vector<RunbookStep> parse_runbook(const std::string& path,
                                               const std::string& dataset_key)
{
    YAML::Node root    = YAML::LoadFile(path);
    YAML::Node dataset = root[dataset_key];
    if (!dataset)
        throw std::runtime_error("Key '" + dataset_key + "' not found in runbook: " + path);

    std::map<int, RunbookStep> step_map;
    for (auto it = dataset.begin(); it != dataset.end(); ++it) {
        const std::string key = it->first.as<std::string>();
        bool all_digits = !key.empty() &&
            std::all_of(key.begin(), key.end(),
                        [](unsigned char c){ return std::isdigit(c); });
        if (!all_digits) continue;
        YAML::Node node = it->second;
        if (!node.IsMap()) continue;
        RunbookStep rs;
        rs.step_num  = std::stoi(key);
        rs.operation = node["operation"].as<std::string>();
        if (node["start"]) rs.start = node["start"].as<int>();
        if (node["end"])   rs.end   = node["end"].as<int>();
        step_map[rs.step_num] = rs;
    }
    std::vector<RunbookStep> result;
    result.reserve(step_map.size());
    for (auto& [k, v] : step_map) result.push_back(v);
    return result;
}

// ─── routeQuery: local routing without Coordinator object ────────────────────
static std::set<int> routeQuery(
    float*                           vec,
    hnswlib::HierarchicalNSW<float>* hnsw,
    const std::vector<int>&          partitions,
    int                              num_partitions,
    RoutingMode                      mode,
    float                            param,
    const std::vector<int>&          center_counts,
    const std::vector<int>&          part_size)
{
    std::set<int> target_ranks;

    if (mode == RoutingMode::BranchingFactor) {
        int bf = static_cast<int>(param);
        auto pq = hnsw->searchKnnCloserFirst(vec, static_cast<size_t>(bf));
        std::unordered_set<int> seen;
        for (auto& [dist, cid] : pq) {
            int shard = partitions[static_cast<int>(cid)];
            if (seen.insert(shard).second)
                target_ranks.insert(shard + 1);
        }

    } else if (mode == RoutingMode::NProbe) {
        int nprobe   = static_cast<int>(param);
        size_t cur_k = static_cast<size_t>(nprobe);
        const size_t ncenters = hnsw->getCurrentElementCount();
        std::vector<std::pair<float, hnswlib::labeltype>> centers;
        while (true) {
            cur_k   = std::min(cur_k, ncenters);
            centers = hnsw->searchKnnCloserFirst(vec, cur_k);
            std::unordered_set<int> unique_parts;
            for (auto& [d, cid] : centers)
                unique_parts.insert(partitions[static_cast<int>(cid)]);
            if (static_cast<int>(unique_parts.size()) >= nprobe || cur_k >= ncenters)
                break;
            cur_k *= 10;
        }
        std::unordered_set<int> seen;
        for (auto& [d, cid] : centers) {
            int shard = partitions[static_cast<int>(cid)];
            if (seen.insert(shard).second) {
                target_ranks.insert(shard + 1);
                if (static_cast<int>(target_ranks.size()) == nprobe) break;
            }
        }

    } else { // RecallTarget
        float recall_target = std::clamp(param, 0.0f, 1.0f);
        const size_t ncenters = hnsw->getCurrentElementCount();
        size_t knn = std::min<size_t>(50, ncenters);
        auto centers = hnsw->searchKnnCloserFirst(vec, knn);
        if (centers.empty()) return target_ranks;

        // Size weight: number of routing centers per partition (precomputed by
        // caller), matching compute_theoretical_recall_updated.py
        // (part_size = np.bincount(partitions)).
        const double d0 = static_cast<double>(centers[0].first) + 1e-10;

        std::vector<double> part_probs(static_cast<size_t>(num_partitions), 0.0);
        for (size_t r = 0; r < centers.size(); r++) {
            const double rel_d   = static_cast<double>(centers[r].first) / d0;
            const int    cid     = static_cast<int>(centers[r].second);
            const int    pid     = partitions[cid];
            const double size_wt = static_cast<double>(part_size[static_cast<size_t>(pid)]);
            part_probs[static_cast<size_t>(pid)] += size_wt * std::exp(-1.0 * rel_d);
        }

        double prob_sum = std::accumulate(part_probs.begin(), part_probs.end(), 0.0);
        if (prob_sum <= 0.0) {
            target_ranks.insert(partitions[static_cast<int>(centers[0].second)] + 1);
            return target_ranks;
        }
        for (double& p : part_probs) p /= prob_sum;

        std::vector<int> ordered(num_partitions);
        std::iota(ordered.begin(), ordered.end(), 0);
        std::sort(ordered.begin(), ordered.end(),
                [&](int a, int b){ return part_probs[a] > part_probs[b]; });

        double acc = 0.0;
        for (int pid : ordered) {
            if (part_probs[static_cast<size_t>(pid)] <= 0.0) break;
            target_ranks.insert(pid + 1);
            acc += part_probs[static_cast<size_t>(pid)];
            if (acc >= recall_target) break;
        }
        if (target_ranks.empty())
            target_ranks.insert(partitions[static_cast<int>(centers[0].second)] + 1);
    }

    return target_ranks;
}

// ─── Generic AllToAllV helper ─────────────────────────────────────────────────
template<typename T>
static void AllToAllV(const std::vector<std::vector<T>>& send_bufs,
                      std::vector<std::vector<T>>&       recv_bufs,
                      MPI_Datatype                       dtype,
                      int world_size, MPI_Comm comm)
{
    std::vector<int> sc(world_size), sd(world_size, 0);
    for (int r = 0; r < world_size; ++r) sc[r] = static_cast<int>(send_bufs[r].size());
    for (int r = 1; r < world_size; ++r) sd[r] = sd[r-1] + sc[r-1];

    std::vector<T> sf;
    { size_t tot = 0; for (auto& b : send_bufs) tot += b.size(); sf.reserve(tot); }
    for (int r = 0; r < world_size; ++r)
        sf.insert(sf.end(), send_bufs[r].begin(), send_bufs[r].end());

    std::vector<int> rc(world_size, 0), rd(world_size, 0);
    MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, comm);
    for (int r = 1; r < world_size; ++r) rd[r] = rd[r-1] + rc[r-1];
    int total_recv = rd[world_size-1] + rc[world_size-1];
    std::vector<T> rf(total_recv);
    MPI_Alltoallv(sf.data(), sc.data(), sd.data(), dtype,
                  rf.data(), rc.data(), rd.data(), dtype, comm);

    recv_bufs.assign(world_size, {});
    for (int r = 0; r < world_size; ++r)
        recv_bufs[r].assign(rf.begin() + rd[r], rf.begin() + rd[r] + rc[r]);
}

// ─── Broadcast routing state from coordinator to all executors ────────────────
static void bcastRoutingState(
    int                                      rank,
    int                                      dim,
    hnswlib::HierarchicalNSW<float>*         coord_meta,
    const std::vector<int>*                  coord_parts,
    const std::unordered_map<int,int>*       coord_ltc,
    hnswlib::HierarchicalNSW<float>*&        out_meta,
    std::vector<int>&                        out_parts,
    std::unordered_map<int,int>&             out_ltc,
    hnswlib::SpaceInterface<float>*          space,
    bool                                     include_ltc)
{
    // 1. Partitions
    {
        int n = (rank == 0) ? static_cast<int>(coord_parts->size()) : 0;
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) out_parts.resize(n);
        MPI_Bcast(
            (rank == 0) ? const_cast<int*>(coord_parts->data()) : out_parts.data(),
            n, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) out_parts = *coord_parts;
    }
    // 2. Meta-HNSW bytes
    {
        int hnsw_size = 0;
        std::vector<char> buf;
        if (rank == 0) {
            coord_meta->saveIndex("tmp_shared_sweep_bcast.bin");
            std::ifstream f("tmp_shared_sweep_bcast.bin", std::ios::binary);
            buf.assign(std::istreambuf_iterator<char>(f), {});
            hnsw_size = static_cast<int>(buf.size());
        }
        MPI_Bcast(&hnsw_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) buf.resize(hnsw_size);
        MPI_Bcast(buf.data(), hnsw_size, MPI_BYTE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            const std::string tmp = "tmp_shared_sweep_r" + std::to_string(rank) + ".bin";
            { std::ofstream f(tmp, std::ios::binary); f.write(buf.data(), hnsw_size); }
            delete out_meta;
            out_meta = new hnswlib::HierarchicalNSW<float>(space, tmp);
            out_meta->setEf(EF_SEARCH);
        }
    }
    // 3. (Optional) label_to_center
    if (include_ltc) {
        int n_lc = (rank == 0) ? static_cast<int>(coord_ltc->size()) : 0;
        MPI_Bcast(&n_lc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<int> labels(n_lc), centers(n_lc);
        if (rank == 0) {
            int i = 0;
            for (auto& [lbl, cid] : *coord_ltc) { labels[i] = lbl; centers[i] = cid; i++; }
        }
        MPI_Bcast(labels.data(),  n_lc, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(centers.data(), n_lc, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            out_ltc.clear(); out_ltc.reserve(n_lc);
            for (int i = 0; i < n_lc; i++) out_ltc[labels[i]] = centers[i];
        } else {
            out_ltc.clear();
            for (auto& [lbl, cid] : *coord_ltc) out_ltc[lbl] = cid;
        }
    }
}

// ─── find_latest_checkpoint ──────────────────────────────────────────────────
// Scan <ckpt_base> for subdirectories named "ckpt_s<N>" and return the path of
// the one with the highest N.  Returns "" if the base directory does not exist
// or contains no valid checkpoint subdirectories.
static std::string find_latest_checkpoint(const std::string& ckpt_base)
{
    std::error_code ec;
    if (!std::filesystem::exists(ckpt_base, ec) || ec) return "";

    long long best_s  = -1;
    std::string best_dir;
    for (const auto& entry : std::filesystem::directory_iterator(ckpt_base, ec)) {
        if (ec || !entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        // Expect "ckpt_s<N>" where N is a non-negative integer.
        if (name.size() <= 6 || name.substr(0, 6) != "ckpt_s") continue;
        try {
            long long n = std::stoll(name.substr(6));
            // Validate: metadata.bin must exist and be readable.
            const std::string meta = entry.path().string() + "/metadata.bin";
            if (!std::filesystem::exists(meta)) continue;
            if (n > best_s) { best_s = n; best_dir = entry.path().string(); }
        } catch (...) {}
    }
    return best_dir;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    if (argc < 7) {
        std::cerr
            << "Usage: " << argv[0]
            << " <dataset> <num_partitions>"
            << " <full_threshold>"
            << " <k> <gt_prefix> <output_file>"
            << " [--init-state-dir <dir> --init-partitions <file> [--init-state-step <N>]]\n"
            << "\n"
            << "  full_threshold : min centers to migrate to trigger a rebuild;\n"
            << "                   set >= " << NCENTERS << " to disable\n"
            << "  gt_prefix      : directory with per-step GT files (step<N>.gt100),"
               " or \"\" for static GT only\n"
            << "\n"
            << "  Starting-state options (skip the from-scratch KMeans build and start\n"
            << "  from the output of msturing-cluster-analysis.cpp step <N>):\n"
            << "    --init-state-dir  <dir>   directory holding step_NNNNNN_hnsw.bin,\n"
            << "                              step_NNNNNN_centers.csv, _center_counts.csv,\n"
            << "                              and _labels.csv from cluster analysis\n"
            << "    --init-partitions <file>  precomputed center→shard partition file\n"
            << "                              (one shard id per line; e.g. produced by\n"
            << "                              runbook_partitions_parallel.cpp)\n"
            << "    --init-state-step <N>     cluster-analysis step to load (default 1)\n"
            << "  Both --init-state-dir and --init-partitions are required to enable this\n"
            << "  mode.  It only applies on a fresh start; an existing checkpoint always\n"
            << "  takes precedence and resumes as usual.\n"
            << "\n"
            << "  Rebuild policy: delta always (shadow-delete departing elements,\n"
            << "  insert arriving ones with replace_deleted=true, keep graph topology).\n"
            << "  A full graph reconstruction is forced whenever any shard's tombstone\n"
            << "  ratio reaches " << TOMBSTONE_RATIO_THRESHOLD
            << " (" << static_cast<int>(TOMBSTONE_RATIO_THRESHOLD * 100) << "% unused slots) -- this\n"
            << "  check fires every step and triggers a rebuild on its own even when the\n"
            << "  center-movement threshold is not met.\n";
        return 1;
    }

    const std::string dataset_name   = argv[1];
    const int         num_partitions = std::stoi(argv[2]);
    const int         full_threshold = std::stoi(argv[3]);
    const int         partial_threshold = full_threshold; // partial rebuilds disabled
    const int         k              = std::stoi(argv[4]);
    const std::string gt_prefix      = argv[5];
    const std::string output_file    = argv[6];

    // ── Optional starting-state flags ────────────────────────────────────────
    // When both --init-state-dir and --init-partitions are supplied, Phase 1
    // loads the routing state produced by msturing-cluster-analysis.cpp step
    // <init_state_step> instead of running KMeans from scratch.
    std::string init_state_dir;
    std::string init_partitions_file;
    int         init_state_step = 1;
    for (int ai = 7; ai < argc; ++ai) {
        const std::string a = argv[ai];
        if      (a == "--init-state-dir"  && ai + 1 < argc) init_state_dir       = argv[++ai];
        else if (a == "--init-partitions" && ai + 1 < argc) init_partitions_file = argv[++ai];
        else if (a == "--init-state-step" && ai + 1 < argc) init_state_step      = std::stoi(argv[++ai]);
        else {
            std::cerr << "ERROR: unrecognised or incomplete argument: " << a << "\n";
            return 1;
        }
    }
    const bool use_init_state = !init_state_dir.empty() && !init_partitions_file.empty();
    if (!init_state_dir.empty() ^ !init_partitions_file.empty()) {
        std::cerr << "ERROR: --init-state-dir and --init-partitions must be used together\n";
        return 1;
    }
    // Checkpoint base directory: strip the output file extension, append
    // Checkpoint base directory: stable across runs so that a restarted run
    // automatically resumes from the previous run's checkpoint even when the
    // output file path changes (e.g. due to a new timestamp in results/).
    // Path: checkpoints/<dataset_name>_t<full_threshold>/
    // E.g.  checkpoints/msturing-100M-clustered_t6000/
    const std::string ckpt_base =
        "checkpoints/" + dataset_name + "_t" + std::to_string(full_threshold);
    // Rebuild policy: always attempt delta (shadow-delete + replace_deleted insert).
    // Falls back to full if the tombstone ratio check fires (see TOMBSTONE_RATIO_THRESHOLD).
    const bool        use_delta_rebuild = true;

    // Build the sweep list — identical on all ranks so collective counts match.
    const auto sweep_combos = build_sweep_combos();
    const int  n_combos     = static_cast<int>(sweep_combos.size());

    // ── MPI init ─────────────────────────────────────────────────────────────
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "ERROR: MPI does not support MPI_THREAD_MULTIPLE\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size == 1) {
        std::cerr << "ERROR: requires at least 2 MPI ranks\n";
        MPI_Finalize(); return 1;
    }
    if (world_size != num_partitions + 1) {
        std::cerr << "ERROR: world_size=" << world_size
                  << " must equal num_partitions+1=" << (num_partitions + 1) << "\n";
        MPI_Finalize(); return 1;
    }

    // ── Dataset info ─────────────────────────────────────────────────────────
    const std::string base_file  = DATASETS[dataset_name]["base_file"];
    const std::string query_file = DATASETS[dataset_name]["query_file"];

    auto [nvectors, dim] = get_dataset_info(base_file);
    const FileFormat file_format = getFileFormat(base_file);

    // ── Runbook ──────────────────────────────────────────────────────────────
    const std::string runbook_path = DATASETS[dataset_name]["runbook"];
    std::vector<RunbookStep> steps = parse_runbook(runbook_path, dataset_name);
    if (steps.empty() || steps[0].operation != "insert") {
        if (rank == 0) std::cerr << "ERROR: runbook must begin with an insert step\n";
        MPI_Finalize(); return 1;
    }

    // ── Log + communicator ────────────────────────────────────────────────────
    std::string log_id = "shared_sweep_" + dataset_name + "_" + std::to_string(num_partitions);
    Log logger(log_id);
    Communicator comm;

    // ── Shared routing state (all ranks) ─────────────────────────────────────
    std::unordered_map<int,int>       label_to_center;
    std::unordered_map<int,int>       label_to_shard;    // label → executor rank (1..P); exact even after delta rebuilds
    std::vector<int>                  routing_partitions;
    std::vector<int>                  routing_center_counts;  // center_id → active vector count
    hnswlib::L2Space                  meta_space(dim);
    hnswlib::HierarchicalNSW<float>*  routing_hnsw = nullptr;

    // ── Checkpoint detection ──────────────────────────────────────────────────
    // Rank 0 scans the checkpoint base directory for the highest "ckpt_s<N>"
    // subdirectory that has a valid metadata.bin.  The loop index of the first
    // step to execute on resume (next_s = s_of_checkpoint + 1) is broadcast
    // to all ranks so both coordinator and executor enter the same code path.
    //
    // resume_s == 1  →  no checkpoint found, fresh start.
    // resume_s >  1  →  resume from checkpoint ckpt_base/ckpt_s<resume_s-1>.
    long long ckpt_resume_s_ll = 1LL;
    if (rank == 0) {
        const std::string found = find_latest_checkpoint(ckpt_base);
        if (!found.empty()) {
            std::ifstream meta(found + "/metadata.bin", std::ios::binary);
            if (meta) {
                size_t next_s; int step_num_v;
                meta.read(reinterpret_cast<char*>(&next_s),    sizeof(next_s));
                meta.read(reinterpret_cast<char*>(&step_num_v), sizeof(step_num_v));
                ckpt_resume_s_ll = static_cast<long long>(next_s);
                std::cout << "[Sweep] Found checkpoint " << found
                          << "  --  resuming at loop s=" << next_s
                          << " (last completed runbook step=" << step_num_v << ")\n";
            } else {
                std::cerr << "[Sweep] WARNING: checkpoint found at " << found
                          << " but metadata.bin is unreadable; starting fresh.\n";
            }
        } else {
            std::cout << "[Sweep] No checkpoint found; starting fresh.\n";
        }
    }
    MPI_Bcast(&ckpt_resume_s_ll, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    const size_t resume_s  = static_cast<size_t>(ckpt_resume_s_ll);
    const bool   resuming  = (resume_s > 1);
    // All ranks reconstruct the resume checkpoint path from resume_s.
    // (Checkpoint at loop index s stores next_s = s+1, so ckpt dir = ckpt_s<resume_s-1>.)
    const std::string resume_ckpt = resuming
        ? ckpt_base + "/ckpt_s" + std::to_string(resume_s - 1)
        : "";

    // ══════════════════════════════════════════════════════════════════════════
    //                         PHASE 1 : INITIAL BUILD
    // ══════════════════════════════════════════════════════════════════════════

    if (rank == 0) {
        const RunbookStep& init = steps[0];
        const int init_start = init.start;
        const int init_n     = init.end - init.start;

        Coordinator metaIndex(dim, &comm, &logger);

        if (!resuming && use_init_state) {
            // ── Phase 1: load starting state from cluster analysis ────────────
            // Load meta-HNSW, centroids, counts, label→center, and partitions
            // produced by msturing-cluster-analysis.cpp step <init_state_step>,
            // then distribute the initial batch to executors using that exact
            // assignment (so the physical shard layout matches the loaded state).
            std::cout << "[Sweep] Loading starting state from cluster analysis: "
                      << init_state_dir << " (step " << init_state_step << "),"
                      << " partitions=" << init_partitions_file << "\n";
            metaIndex.loadFromClusterAnalysis(init_state_dir, init_state_step,
                                              init_partitions_file, num_partitions,
                                              EF_SEARCH, init_start);

            // Build per-vector preassigned shard list from the loaded
            // label→center map and partition assignment.  distribute_vectors
            // indexes preassigned_partitions by absolute global id, so the
            // vector must span [0, init_start + init_n).
            const auto& l2c   = metaIndex.getLabelToCenter();
            const auto& parts = metaIndex.getPartitions();
            std::vector<int> preassigned(static_cast<size_t>(init_start) + init_n, 0);
            int missing = 0;
            for (int i = 0; i < init_n; i++) {
                const int label = init_start + i;
                auto it = l2c.find(label);
                if (it == l2c.end()) { missing++; continue; }
                const int cid = it->second;
                if (cid < 0 || cid >= static_cast<int>(parts.size())) { missing++; continue; }
                preassigned[label] = parts[cid];
            }
            if (missing > 0) {
                std::cerr << "ERROR: " << missing << " of " << init_n
                          << " initial vectors have no valid label→center entry in the "
                          << "loaded state; cannot distribute. Check that "
                          << "step_" << init_state_step << "_labels.csv covers ["
                          << init_start << ", " << (init_start + init_n) << ").\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            std::cout << "[Sweep] Distributing initial " << init_n
                      << " vectors (offset " << init_start
                      << ") using loaded assignment\n";
            metaIndex.distribute_vectors(base_file, init_n, false,
                                         NUM_BUILDING_THREADS, &preassigned, init_start);
            comm.broadcast_termination(world_size);
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "[Sweep] Initial build complete (from loaded state)\n";
            comm.broadcast_ef_search(EF_SEARCH, world_size);
        } else if (!resuming) {
            // ── Phase 1: fresh build ──────────────────────────────────────────
            const size_t ss = std::min(static_cast<size_t>(init_n), SAMPLE_SIZE);
            std::vector<float> sample = getSample(base_file, init_n, dim, ss, init_start);
            metaIndex.setSampleData(sample.data(), ss);
            metaIndex.build(NCENTERS, num_partitions, EF_CONSTRUCTION, M_META);
            std::cout << "[Sweep] Distributing initial " << init_n << " vectors (offset " << init_start << ")\n";
            metaIndex.distribute_vectors(base_file, init_n, false, NUM_BUILDING_THREADS, nullptr, init_start);
            comm.broadcast_termination(world_size);
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "[Sweep] Initial build complete\n";
            comm.broadcast_ef_search(EF_SEARCH, world_size);
        } else {
            // ── Phase 1: resume from checkpoint ──────────────────────────────
            // Loads metaHNSW, partitions, center pos/counts, and the
            // authoritative labels_to_centers.bin written during checkpointing.
            std::cout << "[Sweep] Loading coordinator state from "
                      << resume_ckpt << "/coordinator\n";
            metaIndex.load(resume_ckpt + "/coordinator", EF_SEARCH);
        }

        routing_hnsw = metaIndex.getMetaHNSW();

        // Broadcast routing state to all executors (fresh or resume — same call).
        // include_ltc=true so label_to_center is also synced on every rank.
        bcastRoutingState(rank, dim,
                          metaIndex.getMetaHNSW(),
                          &metaIndex.getPartitions(),
                          &metaIndex.getLabelToCenter(),
                          routing_hnsw, routing_partitions, label_to_center,
                          &meta_space, /*include_ltc=*/true);

        // ── Populate label_to_shard ───────────────────────────────────────────
        // label_to_shard[label] = executor_rank tracks exact ownership so deletes
        // can be routed without throwing exceptions in hnswlib.
        if (!resuming) {
            // Fresh start: label_to_center is accurate — derive label_to_shard.
            for (auto& [lbl, cid] : label_to_center)
                if (cid >= 0 && cid < static_cast<int>(routing_partitions.size()))
                    label_to_shard[lbl] = routing_partitions[cid] + 1;
        } else {
            // Resume: load saved label_to_shard (accurate even after delta rebuilds).
            {
                std::ifstream lts(resume_ckpt + "/coordinator/labels_to_shards.bin",
                                  std::ios::binary);
                if (lts) {
                    size_t n = 0; lts.read(reinterpret_cast<char*>(&n), sizeof(n));
                    label_to_shard.reserve(n);
                    for (size_t i = 0; i < n; i++) {
                        int key, val;
                        lts.read(reinterpret_cast<char*>(&key), sizeof(key));
                        lts.read(reinterpret_cast<char*>(&val), sizeof(val));
                        label_to_shard[key] = val;
                    }
                } else {
                    std::cerr << "[Sweep] WARNING: labels_to_shards.bin not found; "
                              << "deriving label_to_shard from label_to_center "
                              << "(may be stale for vectors moved during prior delta rebuilds)\n";
                    for (auto& [lbl, cid] : label_to_center)
                        if (cid >= 0 && cid < static_cast<int>(routing_partitions.size()))
                            label_to_shard[lbl] = routing_partitions[cid] + 1;
                }
            }
            // Broadcast to executors so all ranks share the same label_to_shard.
            {
                int n_lts = static_cast<int>(label_to_shard.size());
                MPI_Bcast(&n_lts, 1, MPI_INT, 0, MPI_COMM_WORLD);
                std::vector<int> lts_keys(n_lts), lts_vals(n_lts);
                { int i = 0; for (auto& [k, v] : label_to_shard) { lts_keys[i] = k; lts_vals[i] = v; ++i; } }
                MPI_Bcast(lts_keys.data(), n_lts, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(lts_vals.data(), n_lts, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }

        // ── Static ground truth ───────────────────────────────────────────────
        std::vector<std::vector<int>> static_gt;
        bool have_static_gt = false;
        auto gt_it = DATASETS[dataset_name].find("gt_file");
        if (gt_it != DATASETS[dataset_name].end() &&
                !gt_it->second.empty() && gt_it->second != "TODO") {
            static_gt      = readGT(gt_it->second, file_format);
            have_static_gt = !static_gt.empty();
        }

        // ── Open CSV ─────────────────────────────────────────────────────────
        // Fresh start: overwrite and write the header.
        // Always open in append mode: safe for both fresh starts (empty file)
        // and resumes (header already present from previous run).
        std::ofstream csv(output_file, std::ios::out | std::ios::app);
        if (!csv.is_open()) {
            std::cerr << "ERROR: cannot open output file: " << output_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!resuming) {
            csv << "step,operation,mode,param"
                << ",range_start,range_end,time_s,throughput,recall@" << k
                << ",theoretical_recall@" << k
                << ",avg_parts_searched"
                << ",centers_moved,elements_moved"
                << ",repart_hnsw_s,repart_bottom_s,repart_kaffpa_s,repart_relabel_s"
                << ",exec_iterate_s,exec_exchange_s,exec_graph_s,dorebuild_wall_s"
                // actual rebuild variant used ("full" or "delta"); empty for non-rebuild rows.
                // may differ from the run-level rebuild_mode if tombstone ratio forced full.
                << ",rebuild_type"
                // delta-rebuild specific: total unreplaced tombstone slots across shards
                // (-1 for full rebuilds; 0+ for delta rebuilds)
                << ",remaining_deleted_slots";
            for (int i = 0; i < num_partitions; i++) csv << ",shard_" << i << "_size";
            csv << "\n";
        }

        // ── Rebuild statistics struct ─────────────────────────────────────────
        struct RebuildStats {
            int         centers_moved             = -1;
            int         elements_moved            = -1;
            double      repart_hnsw_s             = -1.0;
            double      repart_bottom_s           = -1.0;
            double      repart_kaffpa_s           = -1.0;
            double      repart_relabel_s          = -1.0;
            double      exec_iterate_s            = -1.0;
            double      exec_exchange_s           = -1.0;
            double      exec_graph_s              = -1.0;
            double      dorebuild_wall_s          = -1.0;
            // Sum of remaining unreplaced deleted slots across all shards.
            // Populated only for delta rebuilds; -1 for full rebuilds.
            long long   remaining_deleted_slots   = -1;
            // Actual rebuild variant used: "full", "delta", or "" for non-rebuild rows.
            // May differ from the run-level rebuild_mode when the tombstone ratio
            // check overrides a delta rebuild to full.
            std::string rebuild_type              = "";
        };
        static const RebuildStats kNoRebuild{};

        // ── Helper: write one CSV row ─────────────────────────────────────────
        auto write_row = [&](int step_num, const std::string& op,
                             const std::string& mode_str, float param_val,
                             int rs, int re,
                             double time_s, double throughput, double recall,
                             double theoretical_recall,
                             double avg_parts,
                             const RebuildStats& rb,
                             const std::vector<unsigned long long>& sizes)
        {
            csv << step_num << "," << op
                << "," << mode_str << "," << param_val
                << "," << rs << "," << re
                << "," << time_s << "," << throughput
                << "," << recall
                << "," << theoretical_recall
                << "," << avg_parts
                << "," << rb.centers_moved
                << "," << rb.elements_moved
                << "," << rb.repart_hnsw_s
                << "," << rb.repart_bottom_s
                << "," << rb.repart_kaffpa_s
                << "," << rb.repart_relabel_s
                << "," << rb.exec_iterate_s
                << "," << rb.exec_exchange_s
                << "," << rb.exec_graph_s
                << "," << rb.dorebuild_wall_s
                << "," << rb.rebuild_type
                << "," << rb.remaining_deleted_slots;
            for (auto sz : sizes) csv << "," << sz;
            csv << "\n";
            csv.flush();
        };

        // ── Helper: collect shard sizes ───────────────────────────────────────
        auto collect_sizes = [&]() -> std::vector<unsigned long long> {
            unsigned long long dummy = 0;
            std::vector<unsigned long long> sizes(world_size, 0);
            MPI_Gather(&dummy, 1, MPI_UNSIGNED_LONG_LONG,
                       sizes.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            return std::vector<unsigned long long>(sizes.begin() + 1, sizes.end());
        };

        // ── Helper: sync routing after rebuild ────────────────────────────────
        auto sync_routing_after_rebuild = [&]() {
            routing_hnsw       = metaIndex.getMetaHNSW();
            routing_partitions = metaIndex.getPartitions();
            bcastRoutingState(rank, dim,
                              metaIndex.getMetaHNSW(),
                              &metaIndex.getPartitions(),
                              nullptr,
                              routing_hnsw, routing_partitions, label_to_center,
                              &meta_space, false);
        };

        // ── Helper: sync label_to_shard after rebuild (coordinator side) ──────
        // Each executor reports labels of vectors that arrived during the rebuild
        // exchange via Allgatherv.  The coordinator contributes 0 labels and
        // updates its label_to_shard from the received data.
        // Must be called on ALL ranks (matching the executor-side lambda below)
        // after every rebuild + bcastRoutingState sequence.
        auto sync_label_to_shard_after_rebuild_coord = [&]() {
            int my_n = 0;
            std::vector<int> all_ns(world_size);
            MPI_Allgather(&my_n, 1, MPI_INT, all_ns.data(), 1, MPI_INT, MPI_COMM_WORLD);
            std::vector<int> displs(world_size, 0);
            for (int r = 1; r < world_size; r++) displs[r] = displs[r-1] + all_ns[r-1];
            const int total_moved = displs[world_size-1] + all_ns[world_size-1];
            std::vector<int> all_moved(total_moved);
            int dummy = 0;
            MPI_Allgatherv(&dummy, 0, MPI_INT,
                           all_moved.data(), all_ns.data(), displs.data(),
                           MPI_INT, MPI_COMM_WORLD);
            for (int r = 0; r < world_size; r++)
                for (int i = displs[r]; i < displs[r] + all_ns[r]; i++)
                    label_to_shard[all_moved[i]] = r;
        };

        // Initial insert row (written on fresh start only; on resume the row
        // is already present in the CSV from the previous run).
        if (!resuming) {
            auto sizes = collect_sizes();
            write_row(init.step_num, "insert", "", -1.0f,
                      init.start, init.end, 0.0, 0.0, -1.0, -1.0, -1.0,
                      kNoRebuild, sizes);
        }

        // ══════════════════════════════════════════════════════════════════════
        //                    COORDINATOR STREAMING LOOP
        // ══════════════════════════════════════════════════════════════════════
        // On resume, last_ckpt_dir is initialised to the checkpoint we loaded so
        // it is removed after the next checkpoint is successfully written.
        // On a fresh start it remains empty until the first checkpoint is taken.
        std::string last_ckpt_dir = resume_ckpt;
        int my_ckpt_write_ok = 1;  // set in checkpoint block; used by Allreduce

        for (size_t s = resume_s; s < steps.size(); s++) {
            const RunbookStep& step = steps[s];

            int bcast_buf[3] = { step.step_num, step.start, step.end };
            int op_code = (step.operation == "insert") ? 0
                        : (step.operation == "delete") ? 1
                        : 2;
            MPI_Bcast(&op_code,  1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(bcast_buf, 3, MPI_INT, 0, MPI_COMM_WORLD);

            // ── INSERT ────────────────────────────────────────────────────────
            if (step.operation == "insert") {
                const int n_insert = step.end - step.start;
                std::cout << "[Sweep] Step " << step.step_num
                          << "  INSERT [" << step.start << ", " << step.end << ")\n";

                std::vector<float> batch = readVecs(base_file, dim, n_insert, step.start);

                std::vector<std::vector<int>>   send_ids(world_size);
                std::vector<std::vector<float>> send_vecs(world_size);
                std::vector<std::pair<int,int>>  my_assignments;

                #pragma omp parallel
                {
                    std::vector<std::vector<int>>   local_ids(world_size);
                    std::vector<std::vector<float>> local_vecs(world_size);
                    std::vector<std::pair<int,int>> local_assign;

                    #pragma omp for nowait schedule(static)
                    for (int i = 0; i < n_insert; i++) {
                        if (i % world_size != rank) continue;
                        const int label = step.start + i;
                        float*    vec   = batch.data() + static_cast<size_t>(i) * dim;
                        auto pq  = routing_hnsw->searchKnn(vec, 1);
                        int cid  = static_cast<int>(pq.top().second);
                        int tgt  = routing_partitions[cid] + 1;
                        local_ids[tgt].push_back(label);
                        local_vecs[tgt].insert(local_vecs[tgt].end(), vec, vec + dim);
                        local_assign.push_back({label, cid});
                    }

                    #pragma omp critical
                    {
                        for (int r = 0; r < world_size; r++) {
                            send_ids[r].insert(send_ids[r].end(),
                                               local_ids[r].begin(), local_ids[r].end());
                            send_vecs[r].insert(send_vecs[r].end(),
                                                local_vecs[r].begin(), local_vecs[r].end());
                        }
                        my_assignments.insert(my_assignments.end(),
                                              local_assign.begin(), local_assign.end());
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);
                const double t0 = MPI_Wtime();

                std::vector<std::vector<int>>   recv_ids;
                std::vector<std::vector<float>> recv_vecs;
                AllToAllV(send_ids,  recv_ids,  MPI_INT,   world_size, MPI_COMM_WORLD);
                AllToAllV(send_vecs, recv_vecs, MPI_FLOAT, world_size, MPI_COMM_WORLD);

                // Allgatherv: sync label_to_center.
                {
                    const int my_n = static_cast<int>(my_assignments.size());
                    std::vector<int> all_ns(world_size);
                    MPI_Allgather(&my_n, 1, MPI_INT, all_ns.data(), 1, MPI_INT, MPI_COMM_WORLD);
                    std::vector<int> displs(world_size, 0);
                    for (int r = 1; r < world_size; r++)
                        displs[r] = displs[r-1] + all_ns[r-1];
                    const int total = displs[world_size-1] + all_ns[world_size-1];
                    std::vector<int> all_labels(total), all_centers(total);
                    std::vector<int> my_labels(my_n), my_centers(my_n);
                    for (int i = 0; i < my_n; i++) {
                        my_labels[i]  = my_assignments[i].first;
                        my_centers[i] = my_assignments[i].second;
                    }
                    MPI_Allgatherv(my_labels.data(),  my_n, MPI_INT,
                                   all_labels.data(),  all_ns.data(), displs.data(),
                                   MPI_INT, MPI_COMM_WORLD);
                    MPI_Allgatherv(my_centers.data(), my_n, MPI_INT,
                                   all_centers.data(), all_ns.data(), displs.data(),
                                   MPI_INT, MPI_COMM_WORLD);
                    for (int i = 0; i < total; i++)
                        label_to_center[all_labels[i]] = all_centers[i];
                    // Mirror into label_to_shard (routing_partitions is valid here).
                    for (int i = 0; i < total; i++) {
                        const int cid = all_centers[i];
                        if (cid >= 0 && cid < static_cast<int>(routing_partitions.size()))
                            label_to_shard[all_labels[i]] = routing_partitions[cid] + 1;
                    }
                }

                const double t1 = MPI_Wtime();
                double max_t = 0.0;
                {
                    double elapsed = t1 - t0;
                    MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                }

                {
                    std::vector<int> cids(n_insert);
                    for (int i = 0; i < n_insert; i++)
                        cids[i] = label_to_center[step.start + i];
                    metaIndex.updateCentersForInsertBatch(batch, cids);
                }

                // Rebuild check.  checkNeedRebuild() reports whether the
                // center-movement threshold is met; the tombstone-ratio check
                // below can independently force a full rebuild even when it is not.
                int rb_type = metaIndex.checkNeedRebuild(
                    full_threshold, partial_threshold, EF_CONSTRUCTION, M_META);

                // Tombstone ratio check — now runs every step so it can trigger a
                // rebuild on its own.  Collect the max tombstone ratio across all
                // executor shards (coordinator contributes 0.0).  Must stay
                // symmetric with the executor side below.
                double max_ratio = 0.0;
                {
                    double coord_ratio = 0.0;
                    MPI_Reduce(&coord_ratio, &max_ratio, 1, MPI_DOUBLE,
                               MPI_MAX, 0, MPI_COMM_WORLD);
                }
                const bool tombstone_forces = (max_ratio >= TOMBSTONE_RATIO_THRESHOLD);

                // Rebuild if the center-movement threshold is met OR a shard's
                // tombstone ratio crossed the limit.  A tombstone-forced rebuild
                // is always full; otherwise delta when enabled.
                int do_rebuild = (rb_type > 0 || tombstone_forces) ? 1 : 0;
                MPI_Bcast(&do_rebuild, 1, MPI_INT, 0, MPI_COMM_WORLD);

                RebuildStats rb_stats;
                if (do_rebuild) {
                    const bool actual_delta = use_delta_rebuild && !tombstone_forces;
                    if (tombstone_forces)
                        std::cout << "[Sweep] Step " << step.step_num
                                  << "  tombstone ratio " << max_ratio
                                  << " >= " << TOMBSTONE_RATIO_THRESHOLD
                                  << " -- forcing full rebuild"
                                  << (rb_type > 0 ? "\n"
                                                  : " (center-movement threshold not met)\n");

                    // Dispatch.  When checkNeedRebuild() reported a rebuild it has
                    // cached the repartition, so use the normal delta/full path.
                    // When the rebuild is forced purely by the tombstone ratio
                    // (rb_type == 0) there is no cached repartition, so compute and
                    // dispatch a full rebuild explicitly (same on-wire protocol as
                    // doRebuildSimple, so executors see a FULL_REBUILD_REQUEST).
                    const double t_rb0 = MPI_Wtime();
                    if (rb_type > 0) {
                        if (actual_delta) metaIndex.doRebuildDelta(world_size);
                        else              metaIndex.doRebuildSimple(world_size);
                    } else {
                        metaIndex.doForceFullRebuild(world_size, EF_CONSTRUCTION, M_META);
                    }
                    rb_stats.dorebuild_wall_s = MPI_Wtime() - t_rb0;

                    // Repartition stats are valid after dispatch for both paths.
                    rb_stats.rebuild_type     = actual_delta ? "delta" : "full";
                    rb_stats.centers_moved    = metaIndex.getCachedCentersMoved();
                    rb_stats.elements_moved   = metaIndex.getCachedElementsMoved();
                    rb_stats.repart_hnsw_s    = metaIndex.getCachedRepartHnswS();
                    rb_stats.repart_bottom_s  = metaIndex.getCachedRepartBottomS();
                    rb_stats.repart_kaffpa_s  = metaIndex.getCachedRepartKaffpaS();
                    rb_stats.repart_relabel_s = metaIndex.getCachedRepartRelabelS();

                    sync_routing_after_rebuild();

                    // Collect executor per-phase timings via MPI_Reduce(MAX).
                    double exec_send[3] = {0.0, 0.0, 0.0};
                    double exec_recv[3] = {0.0, 0.0, 0.0};
                    MPI_Reduce(exec_send, exec_recv, 3, MPI_DOUBLE, MPI_MAX,
                               0, MPI_COMM_WORLD);
                    rb_stats.exec_iterate_s  = exec_recv[0];
                    rb_stats.exec_exchange_s = exec_recv[1];
                    rb_stats.exec_graph_s    = exec_recv[2];

                    // Delta rebuild only: collect total remaining deleted slots.
                    // The executor side gates the matching reduce on hdr.type ==
                    // INPLACE_REBUILD_REQUEST, sent only for delta rebuilds, so a
                    // full rebuild (forced or threshold-driven) skips it on both sides.
                    if (actual_delta) {
                        long long del_send = 0LL;
                        long long del_recv = 0LL;
                        MPI_Reduce(&del_send, &del_recv, 1, MPI_LONG_LONG_INT,
                                   MPI_SUM, 0, MPI_COMM_WORLD);
                        rb_stats.remaining_deleted_slots = del_recv;
                    }

                    // Sync label_to_shard: each executor reports its arrived labels.
                    sync_label_to_shard_after_rebuild_coord();

                    const double total_rb_s = rb_stats.repart_hnsw_s + rb_stats.repart_bottom_s
                                           + rb_stats.repart_kaffpa_s + rb_stats.repart_relabel_s
                                           + rb_stats.dorebuild_wall_s;
                    std::cout << "[Sweep] Step " << step.step_num
                              << "  rebuild_type=" << rb_stats.rebuild_type
                              << "  type=" << rb_type
                              << "  centers_moved=" << rb_stats.centers_moved
                              << "  elements_moved=" << rb_stats.elements_moved
                              << "  total_rebuild_s=" << total_rb_s;
                    if (actual_delta)
                        std::cout << "  remaining_deleted_slots="
                                  << rb_stats.remaining_deleted_slots;
                    std::cout << "\n";
                }

                const double throughput = (max_t > 0.0)
                    ? static_cast<double>(n_insert) / max_t : 0.0;
                auto sizes = collect_sizes();
                write_row(step.step_num, "insert", "", -1.0f,
                          step.start, step.end, max_t, throughput, -1.0, -1.0, -1.0,
                          kNoRebuild, sizes);
                if (do_rebuild) {
                    const double total_rb_s = rb_stats.repart_hnsw_s + rb_stats.repart_bottom_s
                                           + rb_stats.repart_kaffpa_s + rb_stats.repart_relabel_s
                                           + rb_stats.dorebuild_wall_s;
                    write_row(step.step_num, "rebuild", "", -1.0f, -1, -1,
                              total_rb_s, -1.0, -1.0, -1.0, -1.0, rb_stats, sizes);
                }

            // ── DELETE ────────────────────────────────────────────────────────
            } else if (step.operation == "delete") {
                const int n_delete = step.end - step.start;
                std::cout << "[Sweep] Step " << step.step_num
                          << "  DELETE [" << step.start << ", " << step.end << ")\n";

                std::vector<int> del_center_ids(n_delete, -1);
                for (int i = 0; i < n_delete; i++) {
                    auto it = label_to_center.find(step.start + i);
                    if (it != label_to_center.end()) del_center_ids[i] = it->second;
                }

                MPI_Barrier(MPI_COMM_WORLD);
                const double t0 = MPI_Wtime();

                for (int i = 0; i < n_delete; i++) {
                    const int cid = del_center_ids[i];
                    if (cid == -1) continue;
                    // Coordinator has no shard — only erases from routing maps.
                    label_to_center.erase(step.start + i);
                    label_to_shard.erase(step.start + i);
                }

                const double t1 = MPI_Wtime();
                double max_t = 0.0;
                {
                    double elapsed = t1 - t0;
                    MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                }

                {
                    std::vector<float> del_vecs =
                        readVecs(base_file, dim, n_delete, step.start);
                    std::vector<float> valid_vecs;
                    std::vector<int>   valid_cids;
                    for (int i = 0; i < n_delete; i++) {
                        if (del_center_ids[i] == -1) continue;
                        const float* v = del_vecs.data() + static_cast<size_t>(i) * dim;
                        valid_vecs.insert(valid_vecs.end(), v, v + dim);
                        valid_cids.push_back(del_center_ids[i]);
                    }
                    if (!valid_cids.empty())
                        metaIndex.updateCentersForDeleteBatch(valid_vecs, valid_cids);
                }

                int rb_type = metaIndex.checkNeedRebuild(
                    full_threshold, partial_threshold, EF_CONSTRUCTION, M_META);

                // Tombstone ratio check — runs every step so it can trigger a
                // rebuild on its own (symmetric with the executor side below).
                double max_ratio = 0.0;
                {
                    double coord_ratio = 0.0;
                    MPI_Reduce(&coord_ratio, &max_ratio, 1, MPI_DOUBLE,
                               MPI_MAX, 0, MPI_COMM_WORLD);
                }
                const bool tombstone_forces = (max_ratio >= TOMBSTONE_RATIO_THRESHOLD);

                int do_rebuild = (rb_type > 0 || tombstone_forces) ? 1 : 0;
                MPI_Bcast(&do_rebuild, 1, MPI_INT, 0, MPI_COMM_WORLD);

                RebuildStats rb_stats;
                if (do_rebuild) {
                    const bool actual_delta = use_delta_rebuild && !tombstone_forces;
                    if (tombstone_forces)
                        std::cout << "[Sweep] Step " << step.step_num
                                  << "  tombstone ratio " << max_ratio
                                  << " >= " << TOMBSTONE_RATIO_THRESHOLD
                                  << " -- forcing full rebuild"
                                  << (rb_type > 0 ? "\n"
                                                  : " (center-movement threshold not met)\n");

                    // Dispatch.  Threshold-driven rebuilds reuse the repartition
                    // cached by checkNeedRebuild(); a purely tombstone-forced
                    // rebuild (rb_type == 0) has no cache, so compute and dispatch
                    // a full rebuild explicitly.
                    const double t_rb0 = MPI_Wtime();
                    if (rb_type > 0) {
                        if (actual_delta) metaIndex.doRebuildDelta(world_size);
                        else              metaIndex.doRebuildSimple(world_size);
                    } else {
                        metaIndex.doForceFullRebuild(world_size, EF_CONSTRUCTION, M_META);
                    }
                    rb_stats.dorebuild_wall_s = MPI_Wtime() - t_rb0;

                    rb_stats.rebuild_type     = actual_delta ? "delta" : "full";
                    rb_stats.centers_moved    = metaIndex.getCachedCentersMoved();
                    rb_stats.elements_moved   = metaIndex.getCachedElementsMoved();
                    rb_stats.repart_hnsw_s    = metaIndex.getCachedRepartHnswS();
                    rb_stats.repart_bottom_s  = metaIndex.getCachedRepartBottomS();
                    rb_stats.repart_kaffpa_s  = metaIndex.getCachedRepartKaffpaS();
                    rb_stats.repart_relabel_s = metaIndex.getCachedRepartRelabelS();

                    sync_routing_after_rebuild();

                    // Collect executor per-phase timings via MPI_Reduce(MAX).
                    double exec_send[3] = {0.0, 0.0, 0.0};
                    double exec_recv[3] = {0.0, 0.0, 0.0};
                    MPI_Reduce(exec_send, exec_recv, 3, MPI_DOUBLE, MPI_MAX,
                               0, MPI_COMM_WORLD);
                    rb_stats.exec_iterate_s  = exec_recv[0];
                    rb_stats.exec_exchange_s = exec_recv[1];
                    rb_stats.exec_graph_s    = exec_recv[2];

                    // Delta rebuild only: collect total remaining deleted slots.
                    // The executor side gates the matching reduce on hdr.type ==
                    // INPLACE_REBUILD_REQUEST, so a full rebuild skips it on both sides.
                    if (actual_delta) {
                        long long del_send = 0LL;
                        long long del_recv = 0LL;
                        MPI_Reduce(&del_send, &del_recv, 1, MPI_LONG_LONG_INT,
                                   MPI_SUM, 0, MPI_COMM_WORLD);
                        rb_stats.remaining_deleted_slots = del_recv;
                    }

                    // Sync label_to_shard: each executor reports its arrived labels.
                    sync_label_to_shard_after_rebuild_coord();

                    const double total_rb_s = rb_stats.repart_hnsw_s + rb_stats.repart_bottom_s
                                           + rb_stats.repart_kaffpa_s + rb_stats.repart_relabel_s
                                           + rb_stats.dorebuild_wall_s;
                    std::cout << "[Sweep] Step " << step.step_num
                              << "  rebuild_type=" << rb_stats.rebuild_type
                              << "  type=" << rb_type
                              << "  centers_moved=" << rb_stats.centers_moved
                              << "  elements_moved=" << rb_stats.elements_moved
                              << "  total_rebuild_s=" << total_rb_s;
                    if (actual_delta)
                        std::cout << "  remaining_deleted_slots="
                                  << rb_stats.remaining_deleted_slots;
                    std::cout << "\n";
                }

                const double throughput = (max_t > 0.0)
                    ? static_cast<double>(n_delete) / max_t : 0.0;
                auto sizes = collect_sizes();
                write_row(step.step_num, "delete", "", -1.0f,
                          step.start, step.end, max_t, throughput, -1.0, -1.0, -1.0,
                          kNoRebuild, sizes);
                if (do_rebuild) {
                    const double total_rb_s = rb_stats.repart_hnsw_s + rb_stats.repart_bottom_s
                                           + rb_stats.repart_kaffpa_s + rb_stats.repart_relabel_s
                                           + rb_stats.dorebuild_wall_s;
                    write_row(step.step_num, "rebuild", "", -1.0f, -1, -1,
                              total_rb_s, -1.0, -1.0, -1.0, -1.0, rb_stats, sizes);
                }

            // ── SEARCH (sweep) ────────────────────────────────────────────────
            } else if (step.operation == "search") {
                std::cout << "[Sweep] Step " << step.step_num
                          << "  SEARCH (" << n_combos << " combos)\n";

                // Load queries (not timed).
                std::vector<float> queries = readVecs(query_file, dim);
                const size_t nq    = queries.size() / dim;
                const size_t chunk = (nq + world_size - 1) / world_size;
                const size_t my_qs = static_cast<size_t>(rank) * chunk;
                const size_t my_qe = std::min(my_qs + chunk, nq);

                // Load GT once for the whole sweep over this search step.
                std::vector<std::vector<int>> gt;
                bool have_gt = false;
                if (!gt_prefix.empty()) {
                    std::string per_step =
                        gt_prefix + "/step" + std::to_string(step.step_num) + ".gt100";
                    if (std::filesystem::exists(per_step)) {
                        gt      = readGT(per_step, file_format);
                        have_gt = !gt.empty();
                    }
                }
                if (!have_gt && have_static_gt) { gt = static_gt; have_gt = true; }

                // Broadcast current center_counts to all ranks so every rank uses
                // the same per-cluster size prior in routeQuery (RecallTarget mode).
                {
                    routing_center_counts = metaIndex.getCenterCounts();
                    int n = static_cast<int>(routing_center_counts.size());
                    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(routing_center_counts.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
                }

                // Pin the routing meta-HNSW search ef for this step.  Done at
                // search time on every rank because rebuilds and bcastRoutingState
                // reset routing_hnsw's ef to EF_SEARCH; this guarantees both the
                // GT->partition lookup and routeQuery use EF_ROUTING (matching the
                // Python theoretical pipeline's router ef).
                routing_hnsw->setEf(EF_ROUTING);

                // ── Theoretical-recall setup ─────────────────────────────
                // For each query in this rank's slice, resolve its top-K GT
                // neighbours to partition IDs via a fresh meta-HNSW lookup
                // (mirrors compute_theoretical_recall_updated.py).  Each rank
                // computes its own slice — no broadcast needed because
                // routing_hnsw and routing_partitions are already replicated.
                // Runs outside the per-combo timing window, so search
                // throughput is unaffected.
                //
                // gt_partitions[q-my_qs] is a per-neighbour vector (length kk,
                // -1 for unresolved ids) — NOT a set.  This gives the
                // neighbor-frequency theoretical recall used by the Python
                // _recall(), which is a strict upper bound on actual recall.
                std::vector<std::vector<int>> gt_partitions(my_qe - my_qs);
                if (have_gt) {
                    BaseMmap base_view(base_file, file_format);
                    #pragma omp parallel
                    {
                        std::vector<float> tmp_vec;
                        #pragma omp for schedule(static)
                        for (size_t q = my_qs; q < my_qe; q++) {
                            if (q >= gt.size()) continue;
                            const auto& gt_q = gt[q];
                            const int   kk   = std::min(static_cast<int>(gt_q.size()), k);
                            auto&       glist = gt_partitions[q - my_qs];
                            glist.assign(kk, -1);
                            for (int i = 0; i < kk; i++) {
                                const int gid = gt_q[i];
                                if (gid < 0 || static_cast<uint32_t>(gid) >= base_view.n())
                                    continue;
                                base_view.readVector(static_cast<uint32_t>(gid), tmp_vec);
                                auto pq = routing_hnsw->searchKnn(tmp_vec.data(), 1);
                                if (pq.empty()) continue;
                                const int cid = static_cast<int>(pq.top().second);
                                if (cid < 0 || cid >= static_cast<int>(routing_partitions.size()))
                                    continue;
                                glist[i] = routing_partitions[cid];
                            }
                        }
                    }
                }

                // ── Combo loop (results buffered; collect_sizes() called once after) ──
                // Declared before the loop because ComboResult must live that long.
                struct ComboResult {
                    std::string mode_str;
                    float       param;
                    double      time_s;
                    double      qps;
                    double      recall;
                    double      avg_parts;
                    // Theoretical recall: fraction of partitions holding the
                    // top-K GT neighbours that the router would also probe
                    // for this mode/param.  GT→partition mapping uses a fresh
                    // routing_hnsw->searchKnn(gt_vec, 1) lookup per GT id,
                    // matching compute_theoretical_recall_updated.py exactly.
                    // -1 if no GT was loaded for this step.
                    double      theoretical_recall;
                };
                std::vector<ComboResult> combo_results;
                combo_results.reserve(n_combos);

                // Precompute once: number of routing centers per partition.
                // Passed into routeQuery so RecallTarget mode doesn't recompute
                // this per-query.  Matches np.bincount(partitions) in the Python.
                std::vector<int> routing_part_size(static_cast<size_t>(num_partitions), 0);
                for (int p : routing_partitions) ++routing_part_size[static_cast<size_t>(p)];

                for (int ci = 0; ci < n_combos; ci++) {
                    const auto& [sweep_mode, sweep_param] = sweep_combos[ci];
                    const std::string sweep_mode_str = mode_to_string(sweep_mode);

                    std::vector<std::vector<uint32_t>> send_qids(world_size);
                    std::vector<std::vector<float>>    send_qvecs(world_size);
                    long long my_total_parts = 0;
                    // Theoretical-recall accumulator over this rank's slice
                    // (computed alongside routing; pre-timing).
                    double    my_theo_sum     = 0.0;
                    long long my_theo_counted = 0;

                    #pragma omp parallel
                    {
                        std::vector<std::vector<uint32_t>> local_qids(world_size);
                        std::vector<std::vector<float>>    local_qvecs(world_size);
                        long long thread_parts      = 0;
                        double    thread_theo_sum   = 0.0;
                        long long thread_theo_count = 0;

                        #pragma omp for nowait schedule(static)
                        for (size_t q = my_qs; q < my_qe; q++) {
                            float* Q = queries.data() + q * dim;
                            std::set<int> targets = routeQuery(
                                Q, routing_hnsw, routing_partitions,
                                num_partitions, sweep_mode, sweep_param,
                                routing_center_counts, routing_part_size);
                            thread_parts += static_cast<long long>(targets.size());
                            for (int tgt : targets) {
                                local_qids[tgt].push_back(static_cast<uint32_t>(q));
                                local_qvecs[tgt].insert(local_qvecs[tgt].end(), Q, Q + dim);
                            }

                            // Theoretical recall (neighbor-frequency, upper-
                            // bounds actual recall): for each top-K GT
                            // neighbour, check whether its owning partition
                            // is in the visited set; divide by k.  targets
                            // stores partition_id + 1; subtract 1 to recover.
                            const auto& glist = gt_partitions[q - my_qs];
                            if (!glist.empty()) {
                                int inter = 0;
                                for (int pid : glist) {
                                    if (pid < 0) continue;
                                    if (targets.count(pid + 1)) ++inter;
                                }
                                thread_theo_sum   += static_cast<double>(inter)
                                                   / static_cast<double>(k);
                                thread_theo_count += 1;
                            }
                        }

                        #pragma omp critical
                        {
                            for (int r = 0; r < world_size; r++) {
                                send_qids[r].insert(send_qids[r].end(),
                                                    local_qids[r].begin(), local_qids[r].end());
                                send_qvecs[r].insert(send_qvecs[r].end(),
                                                     local_qvecs[r].begin(), local_qvecs[r].end());
                            }
                            my_total_parts  += thread_parts;
                            my_theo_sum     += thread_theo_sum;
                            my_theo_counted += thread_theo_count;
                        }
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    const double t0_s = MPI_Wtime();

                    std::vector<std::vector<uint32_t>> recv_qids;
                    std::vector<std::vector<float>>    recv_qvecs;
                    AllToAllV(send_qids,  recv_qids,  MPI_UINT32_T, world_size, MPI_COMM_WORLD);
                    AllToAllV(send_qvecs, recv_qvecs, MPI_FLOAT,    world_size, MPI_COMM_WORLD);

                    // Coordinator has no shard — empty Phase-2 send buffers.
                    std::vector<std::vector<uint32_t>> snd_rqids(world_size);
                    std::vector<std::vector<uint32_t>> snd_rids(world_size);
                    std::vector<std::vector<float>>    snd_rdists(world_size);
                    std::vector<std::vector<uint32_t>> rcv_rqids;
                    std::vector<std::vector<uint32_t>> rcv_rids;
                    std::vector<std::vector<float>>    rcv_rdists;
                    AllToAllV(snd_rqids,  rcv_rqids,  MPI_UINT32_T, world_size, MPI_COMM_WORLD);
                    AllToAllV(snd_rids,   rcv_rids,   MPI_UINT32_T, world_size, MPI_COMM_WORLD);
                    AllToAllV(snd_rdists, rcv_rdists,  MPI_FLOAT,    world_size, MPI_COMM_WORLD);

                    const double t1_s = MPI_Wtime();
                    double max_t = 0.0;
                    {
                        double elapsed = t1_s - t0_s;
                        MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                    }

                    // Fused SUM reduce: (total_parts, theo_sum, theo_counted).
                    // Encoded as double[3] — long long counts fit exactly up
                    // to 2^53, far above any realistic nq * world_size.
                    double local_sum[3]  = {
                        static_cast<double>(my_total_parts),
                        my_theo_sum,
                        static_cast<double>(my_theo_counted),
                    };
                    double global_sum[3] = {0.0, 0.0, 0.0};
                    MPI_Reduce(local_sum, global_sum, 3, MPI_DOUBLE,
                               MPI_SUM, 0, MPI_COMM_WORLD);
                    const double avg_parts =
                        (nq > 0) ? global_sum[0] / static_cast<double>(nq) : -1.0;
                    const double theoretical_recall =
                        (global_sum[2] > 0.0) ? global_sum[1] / global_sum[2] : -1.0;

                    using KNNVec = std::vector<std::pair<float, uint32_t>>;
                    std::vector<KNNVec> neighbors(my_qe - my_qs);
                    for (int src = 0; src < world_size; src++) {
                        const size_t nres = rcv_rqids[src].size();
                        for (size_t j = 0; j < nres; j++) {
                            const uint32_t qid   = rcv_rqids[src][j];
                            const size_t q_local = qid - my_qs;
                            if (q_local >= neighbors.size()) continue;
                            for (int nn = 0; nn < k; nn++) {
                                const size_t off = j * k + nn;
                                if (off >= rcv_rids[src].size()) break;
                                neighbors[q_local].push_back(
                                    {rcv_rdists[src][off], rcv_rids[src][off]});
                            }
                        }
                    }
                    for (auto& nv : neighbors) {
                        std::sort(nv.begin(), nv.end());
                        nv.erase(std::unique(nv.begin(), nv.end(),
                            [](const auto& a, const auto& b){ return a.second == b.second; }),
                            nv.end());
                        if (static_cast<int>(nv.size()) > k) nv.resize(k);
                    }

                    uint64_t my_hits = 0;
                    if (have_gt) {
                        for (size_t q = my_qs; q < my_qe; q++) {
                            if (q >= gt.size()) break;
                            const auto& gt_q  = gt[q];
                            const auto& res_q = neighbors[q - my_qs];
                            std::set<uint32_t> res_set;
                            for (auto& [d, lbl] : res_q) res_set.insert(lbl);
                            for (int j = 0; j < k && j < static_cast<int>(gt_q.size()); j++)
                                if (res_set.count(static_cast<uint32_t>(gt_q[j])))
                                    my_hits++;
                        }
                    }
                    uint64_t total_hits = 0;
                    MPI_Reduce(&my_hits, &total_hits, 1, MPI_UINT64_T,
                               MPI_SUM, 0, MPI_COMM_WORLD);

                    const double recall = (have_gt && nq > 0)
                        ? static_cast<double>(total_hits) / (static_cast<double>(nq) * k)
                        : -1.0;
                    const double qps = (max_t > 0.0) ? static_cast<double>(nq) / max_t : 0.0;

                    std::cout << "[Sweep] Step " << step.step_num
                              << "  mode=" << sweep_mode_str
                              << "  param=" << sweep_param
                              << "  recall@" << k << "=" << recall
                              << "  theo_recall@" << k << "=" << theoretical_recall
                              << "  qps=" << qps
                              << "  avg_parts=" << avg_parts << "\n";

                    combo_results.push_back({sweep_mode_str, sweep_param,
                                             max_t, qps, recall, avg_parts,
                                             theoretical_recall});
                }

                // No rebuild after search.
                int rb_type = 0;
                MPI_Bcast(&rb_type, 1, MPI_INT, 0, MPI_COMM_WORLD);

                // Collect shard sizes once for all combo rows in this step.
                auto sizes = collect_sizes();
                for (auto& cr : combo_results) {
                    write_row(step.step_num, "search",
                              cr.mode_str, cr.param,
                              0, static_cast<int>(nq),
                              cr.time_s, cr.qps, cr.recall,
                              cr.theoretical_recall,
                              cr.avg_parts,
                              kNoRebuild, sizes);
                }

            } else {
                std::cerr << "[Sweep] Unknown operation '" << step.operation
                          << "' in step " << step.step_num << "; skipping\n";
                int rb_type = 0;
                MPI_Bcast(&rb_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
                unsigned long long dummy = 0;
                std::vector<unsigned long long> sizes_tmp(world_size);
                MPI_Gather(&dummy, 1, MPI_UNSIGNED_LONG_LONG,
                           sizes_tmp.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            }

            // ── CHECKPOINT ────────────────────────────────────────────────────
            // Taken at the end of every CHECKPOINT_INTERVAL-th step so the
            // binary can be restarted from a recent point after a failure.
            // Protocol:
            //   1. Write new checkpoint files (coordinator + executor sides).
            //   2. MPI_Barrier — all ranks finish writing before any deletion.
            //   3. Rank 0 removes the *previous* checkpoint directory.
            //
            // The new checkpoint is completely on disk before the old one is
            // touched, so the system always has at least one valid checkpoint.
            if (s % CHECKPOINT_INTERVAL == 0) {
                const std::string ckpt_dir =
                    ckpt_base + "/ckpt_s" + std::to_string(s);
                const std::string coord_dir = ckpt_dir + "/coordinator";

                std::cout << "[Sweep] Step " << step.step_num
                          << "  checkpoint -> " << ckpt_dir << "\n";

                // ── 1a. Write coordinator state ───────────────────────────────
                std::filesystem::create_directories(coord_dir);

                // Save meta-HNSW, partitions, center positions, center counts.
                // (Coordinator::save() also writes labels_to_centers.bin but
                // that file is stale — it is only populated at build time and
                // never updated during streaming.  We overwrite it below.)
                metaIndex.save(coord_dir);

                // Overwrite labels_to_centers.bin with the authoritative map
                // that is kept live via Allgatherv on every INSERT/DELETE step.
                {
                    std::ofstream ltc(coord_dir + "/labels_to_centers.bin",
                                      std::ios::binary | std::ios::trunc);
                    if (!ltc) {
                        std::cerr << "[Sweep] ERROR: cannot write "
                                  << coord_dir << "/labels_to_centers.bin\n";
                    } else {
                        const std::size_t n = label_to_center.size();
                        ltc.write(reinterpret_cast<const char*>(&n), sizeof(n));
                        for (const auto& [key, val] : label_to_center) {
                            ltc.write(reinterpret_cast<const char*>(&key), sizeof(key));
                            ltc.write(reinterpret_cast<const char*>(&val), sizeof(val));
                        }
                    }
                }

                // Save label_to_shard for accurate resume.
                // Mirrors labels_to_centers.bin but stores executor ranks instead
                // of center IDs, so resume does not depend on deriving ownership
                // from a potentially stale label_to_center.
                {
                    std::ofstream lts(coord_dir + "/labels_to_shards.bin",
                                      std::ios::binary | std::ios::trunc);
                    if (!lts) {
                        std::cerr << "[Sweep] ERROR: cannot write "
                                  << coord_dir << "/labels_to_shards.bin\n";
                    } else {
                        const std::size_t n = label_to_shard.size();
                        lts.write(reinterpret_cast<const char*>(&n), sizeof(n));
                        for (const auto& [key, val] : label_to_shard) {
                            lts.write(reinterpret_cast<const char*>(&key), sizeof(key));
                            lts.write(reinterpret_cast<const char*>(&val), sizeof(val));
                        }
                    }
                }

                // Write metadata: loop index of the next step to execute on
                // resume, and the corresponding runbook step_num for sanity.
                {
                    std::ofstream meta(ckpt_dir + "/metadata.bin",
                                       std::ios::binary | std::ios::trunc);
                    const size_t next_s    = s + 1;
                    const int    step_num_v = step.step_num;
                    meta.write(reinterpret_cast<const char*>(&next_s),    sizeof(next_s));
                    meta.write(reinterpret_cast<const char*>(&step_num_v), sizeof(step_num_v));
                }

                // ── 1b. Executors write their shards (happens concurrently) ──
                // (Executor side mirrors this block; see executor streaming loop.)

                // Verify coordinator writes succeeded (metadata.bin is last written).
                {
                    std::error_code ec;
                    const auto sz = std::filesystem::file_size(
                        ckpt_dir + "/metadata.bin", ec);
                    my_ckpt_write_ok = (!ec && sz > 0) ? 1 : 0;
                }
                if (!my_ckpt_write_ok)
                    std::cerr << "[Sweep] WARNING: coordinator checkpoint write failed "
                              << "at " << ckpt_dir << "\n";

                // ── 2. Barrier + all-ranks write validation ───────────────────
                // Each rank reports whether its own write succeeded (1/0).
                // MPI_Allreduce(MIN) gives 1 only if ALL ranks wrote successfully.
                // If any rank failed (e.g. disk full), deletion is skipped so the
                // previous checkpoint is kept as a valid fallback.
                MPI_Barrier(MPI_COMM_WORLD);
                int all_ckpt_writes_ok = 0;
                MPI_Allreduce(&my_ckpt_write_ok, &all_ckpt_writes_ok,
                              1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

                // ── 3. Remove the previous checkpoint (rank 0 only) ──────────
                if (all_ckpt_writes_ok) {
                    if (!last_ckpt_dir.empty() &&
                            std::filesystem::exists(last_ckpt_dir)) {
                        std::error_code ec;
                        std::filesystem::remove_all(last_ckpt_dir, ec);
                        if (ec)
                            std::cerr << "[Sweep] WARNING: failed to remove old checkpoint "
                                      << last_ckpt_dir << ": " << ec.message() << "\n";
                        else
                            std::cout << "[Sweep] Removed old checkpoint "
                                      << last_ckpt_dir << "\n";
                    }
                } else {
                    std::cout << "[Sweep] WARNING: new checkpoint incomplete "
                              << "(disk full on some node?); keeping old checkpoint "
                              << last_ckpt_dir << " as fallback.\n";
                }
                last_ckpt_dir = ckpt_dir;
            }

        } // end coordinator streaming loop

        csv.close();
        std::cout << "[Sweep] Done. Results written to " << output_file << "\n";

    // ══════════════════════════════════════════════════════════════════════════
    //                        EXECUTOR (rank > 0)
    // ══════════════════════════════════════════════════════════════════════════
    } else {

        Executor subIndex(rank, dim, comm, &logger);

        if (!resuming) {
            // Phase 1: receive initial vectors and build sub-HNSW.
            {
                bool done = false;
                while (!done) {
                    MessageHeader hdr;
                    comm.recv_header(hdr, 0);
                    if (hdr.type == END_OF_COMMUNICATION) { done = true; }
                    else { assert(hdr.type == VECTOR_SEND); subIndex.receiveData(hdr.size); }
                }
            }
            subIndex.build(EF_CONSTRUCTION, M_SUB, NUM_BUILDING_THREADS);
            MPI_Barrier(MPI_COMM_WORLD);

            {
                MessageHeader hdr;
                comm.recv_header(hdr, 0);
                if (hdr.type == SET_EF_SEARCH) subIndex.setEfSearch(hdr.size);
            }
        } else {
            // Phase 1: resume — load this executor's shard from checkpoint.
            std::cout << "[Sweep Executor " << rank
                      << "] Loading shard from " << resume_ckpt << "\n";
            subIndex.load(resume_ckpt + "/shard", EF_SEARCH);
        }

        // Receive routing state broadcast from coordinator (fresh or resume).
        bcastRoutingState(rank, dim, nullptr, nullptr, nullptr,
                          routing_hnsw, routing_partitions, label_to_center,
                          &meta_space, /*include_ltc=*/true);

        // ── Populate label_to_shard ───────────────────────────────────────────
        if (!resuming) {
            // Fresh start: label_to_center is accurate — derive label_to_shard.
            for (auto& [lbl, cid] : label_to_center)
                if (cid >= 0 && cid < static_cast<int>(routing_partitions.size()))
                    label_to_shard[lbl] = routing_partitions[cid] + 1;
        } else {
            // Resume: receive label_to_shard broadcast from coordinator.
            int n_lts = 0;
            MPI_Bcast(&n_lts, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::vector<int> lts_keys(n_lts), lts_vals(n_lts);
            MPI_Bcast(lts_keys.data(), n_lts, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(lts_vals.data(), n_lts, MPI_INT, 0, MPI_COMM_WORLD);
            label_to_shard.reserve(n_lts);
            for (int j = 0; j < n_lts; j++) label_to_shard[lts_keys[j]] = lts_vals[j];
        }

        // Participate in initial-row size gather (fresh start only; on resume
        // the gather was already done in the original run).
        if (!resuming) {
            unsigned long long my_size = subIndex.getElementCount();
            MPI_Gather(&my_size, 1, MPI_UNSIGNED_LONG_LONG,
                       nullptr,  1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        }

        // ── Helper: sync label_to_shard after rebuild (executor side) ───────────
        // Contributes this executor's arrived labels via Allgatherv so all ranks
        // update label_to_shard correctly.  Must match the coordinator-side lambda.
        auto sync_label_to_shard_after_rebuild_exec = [&]() {
            const auto& arrived = subIndex.getLastRebuildMovedLabels();
            const int my_n = static_cast<int>(arrived.size());
            std::vector<int> all_ns(world_size);
            MPI_Allgather(&my_n, 1, MPI_INT, all_ns.data(), 1, MPI_INT, MPI_COMM_WORLD);
            std::vector<int> displs(world_size, 0);
            for (int r = 1; r < world_size; r++) displs[r] = displs[r-1] + all_ns[r-1];
            const int total_moved = displs[world_size-1] + all_ns[world_size-1];
            std::vector<int> all_moved(total_moved);
            int dummy = 0;
            MPI_Allgatherv(arrived.empty() ? &dummy : arrived.data(), my_n, MPI_INT,
                           all_moved.data(), all_ns.data(), displs.data(),
                           MPI_INT, MPI_COMM_WORLD);
            for (int r = 0; r < world_size; r++)
                for (int i = displs[r]; i < displs[r] + all_ns[r]; i++)
                    label_to_shard[all_moved[i]] = r;
        };

        // ══════════════════════════════════════════════════════════════════════
        //                    EXECUTOR STREAMING LOOP
        // ══════════════════════════════════════════════════════════════════════
        // Mirrors the coordinator's tracking variable (see coordinator loop).
        std::string last_ckpt_dir = resume_ckpt;

        for (size_t s = resume_s; s < steps.size(); s++) {
            int op_code, bcast_buf[3];
            MPI_Bcast(&op_code,  1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(bcast_buf, 3, MPI_INT, 0, MPI_COMM_WORLD);
            const int range_start = bcast_buf[1];
            const int range_end   = bcast_buf[2];

            // ── INSERT ────────────────────────────────────────────────────────
            if (op_code == 0) {
                const int n_insert = range_end - range_start;

                std::vector<float> batch = readVecs(base_file, dim, n_insert, range_start);

                std::vector<std::vector<int>>   send_ids(world_size);
                std::vector<std::vector<float>> send_vecs(world_size);
                std::vector<std::pair<int,int>>  my_assignments;

                #pragma omp parallel
                {
                    std::vector<std::vector<int>>   local_ids(world_size);
                    std::vector<std::vector<float>> local_vecs(world_size);
                    std::vector<std::pair<int,int>> local_assign;

                    #pragma omp for nowait schedule(static)
                    for (int i = 0; i < n_insert; i++) {
                        if (i % world_size != rank) continue;
                        const int label = range_start + i;
                        float*    vec   = batch.data() + static_cast<size_t>(i) * dim;
                        auto pq  = routing_hnsw->searchKnn(vec, 1);
                        int cid  = static_cast<int>(pq.top().second);
                        int tgt  = routing_partitions[cid] + 1;
                        local_ids[tgt].push_back(label);
                        local_vecs[tgt].insert(local_vecs[tgt].end(), vec, vec + dim);
                        local_assign.push_back({label, cid});
                    }

                    #pragma omp critical
                    {
                        for (int r = 0; r < world_size; r++) {
                            send_ids[r].insert(send_ids[r].end(),
                                               local_ids[r].begin(), local_ids[r].end());
                            send_vecs[r].insert(send_vecs[r].end(),
                                                local_vecs[r].begin(), local_vecs[r].end());
                        }
                        my_assignments.insert(my_assignments.end(),
                                              local_assign.begin(), local_assign.end());
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);
                const double t0_ex = MPI_Wtime();

                std::vector<std::vector<int>>   recv_ids;
                std::vector<std::vector<float>> recv_vecs;
                AllToAllV(send_ids,  recv_ids,  MPI_INT,   world_size, MPI_COMM_WORLD);
                AllToAllV(send_vecs, recv_vecs, MPI_FLOAT, world_size, MPI_COMM_WORLD);

                {
                    std::vector<float> flat_vecs;
                    std::vector<int>   flat_labels;
                    for (int src = 0; src < world_size; src++) {
                        for (size_t j = 0; j < recv_ids[src].size(); j++) {
                            flat_labels.push_back(recv_ids[src][j]);
                            flat_vecs.insert(flat_vecs.end(),
                                             recv_vecs[src].begin() + static_cast<ptrdiff_t>(j * dim),
                                             recv_vecs[src].begin() + static_cast<ptrdiff_t>((j+1) * dim));
                        }
                    }
                    if (!flat_labels.empty())
                        subIndex.insertLocalBatch(flat_vecs, flat_labels);
                }

                {
                    const int my_n = static_cast<int>(my_assignments.size());
                    std::vector<int> all_ns(world_size);
                    MPI_Allgather(&my_n, 1, MPI_INT, all_ns.data(), 1, MPI_INT, MPI_COMM_WORLD);
                    std::vector<int> displs(world_size, 0);
                    for (int r = 1; r < world_size; r++) displs[r] = displs[r-1] + all_ns[r-1];
                    const int total = displs[world_size-1] + all_ns[world_size-1];
                    std::vector<int> all_labels(total), all_centers(total);
                    std::vector<int> my_labels(my_n), my_centers(my_n);
                    for (int i = 0; i < my_n; i++) {
                        my_labels[i]  = my_assignments[i].first;
                        my_centers[i] = my_assignments[i].second;
                    }
                    MPI_Allgatherv(my_labels.data(),  my_n, MPI_INT,
                                   all_labels.data(),  all_ns.data(), displs.data(),
                                   MPI_INT, MPI_COMM_WORLD);
                    MPI_Allgatherv(my_centers.data(), my_n, MPI_INT,
                                   all_centers.data(), all_ns.data(), displs.data(),
                                   MPI_INT, MPI_COMM_WORLD);
                    for (int i = 0; i < total; i++)
                        label_to_center[all_labels[i]] = all_centers[i];
                    // Mirror into label_to_shard (routing_partitions is valid here).
                    for (int i = 0; i < total; i++) {
                        const int cid = all_centers[i];
                        if (cid >= 0 && cid < static_cast<int>(routing_partitions.size()))
                            label_to_shard[all_labels[i]] = routing_partitions[cid] + 1;
                    }
                }

                {
                    double elapsed = MPI_Wtime() - t0_ex;
                    double max_t   = 0.0;
                    MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                }

                // Tombstone ratio check — contribute this shard's ratio to the
                // MAX reduce every step (symmetric with the coordinator) so it can
                // trigger a forced full rebuild on its own.  Must be ordered before
                // the do_rebuild broadcast.
                {
                    double my_ratio = subIndex.getTombstoneRatio();
                    double dummy_max = 0.0;
                    MPI_Reduce(&my_ratio, &dummy_max, 1, MPI_DOUBLE,
                               MPI_MAX, 0, MPI_COMM_WORLD);
                }
                int do_rebuild = 0;
                MPI_Bcast(&do_rebuild, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (do_rebuild) {
                    MessageHeader hdr;
                    comm.recv_header(hdr, 0);
                    if (hdr.type == INPLACE_REBUILD_REQUEST) {
                        subIndex.reBuildDelta(static_cast<int>(hdr.size), NCENTERS,
                                              world_size, NUM_BUILDING_THREADS);
                    } else {
                        subIndex.reBuild(static_cast<int>(hdr.size), NCENTERS,
                                         world_size, EF_CONSTRUCTION, M_SUB,
                                         NUM_BUILDING_THREADS);
                    }
                    // Sync updated routing state.
                    bcastRoutingState(rank, dim, nullptr, nullptr, nullptr,
                                      routing_hnsw, routing_partitions, label_to_center,
                                      &meta_space, false);
                    // Contribute per-phase timings to coordinator's MPI_Reduce(MAX).
                    double exec_send[3] = {
                        subIndex.getLastRebuildIterateS(),
                        subIndex.getLastRebuildExchangeS(),
                        subIndex.getLastRebuildGraphS()
                    };
                    double exec_recv[3]; // unused on non-root ranks
                    MPI_Reduce(exec_send, exec_recv, 3, MPI_DOUBLE, MPI_MAX,
                               0, MPI_COMM_WORLD);
                    // Delta rebuild only: contribute remaining deleted slots to
                    // the coordinator's SUM reduce.
                    if (hdr.type == INPLACE_REBUILD_REQUEST) {
                        long long del_send = static_cast<long long>(
                            subIndex.getLastRebuildRemainingDeleted());
                        long long del_recv = 0LL; // unused on non-root ranks
                        MPI_Reduce(&del_send, &del_recv, 1, MPI_LONG_LONG_INT,
                                   MPI_SUM, 0, MPI_COMM_WORLD);
                    }
                    // Sync label_to_shard: report arrived labels to all ranks.
                    sync_label_to_shard_after_rebuild_exec();
                }

                {
                    unsigned long long my_size = subIndex.getElementCount();
                    MPI_Gather(&my_size, 1, MPI_UNSIGNED_LONG_LONG,
                               nullptr,  1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
                }

            // ── DELETE ────────────────────────────────────────────────────────
            } else if (op_code == 1) {
                const int n_delete = range_end - range_start;

                std::vector<int> del_center_ids(n_delete, -1);
                for (int i = 0; i < n_delete; i++) {
                    auto it = label_to_center.find(range_start + i);
                    if (it != label_to_center.end()) del_center_ids[i] = it->second;
                }

                MPI_Barrier(MPI_COMM_WORLD);
                const double t0_del = MPI_Wtime();

                std::vector<int> my_delete_labels;
                for (int i = 0; i < n_delete; i++) {
                    const int label = range_start + i;
                    const int cid   = del_center_ids[i];
                    if (cid == -1) continue;
                    label_to_center.erase(label);
                    auto it = label_to_shard.find(label);
                    if (it != label_to_shard.end()) {
                        // label_to_shard is accurate even after delta rebuilds:
                        // only the owning executor calls markDelete.
                        if (it->second == rank)
                            my_delete_labels.push_back(label);
                        label_to_shard.erase(it);
                    } else {
                        // Fallback: label_to_shard entry missing (shouldn't happen
                        // in steady state; could occur with a pre-fix checkpoint).
                        my_delete_labels.push_back(label);
                    }
                }
                subIndex.markDeleteLocalBatch(my_delete_labels);

                {
                    double elapsed = MPI_Wtime() - t0_del;
                    double max_t   = 0.0;
                    MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                }

                // Tombstone ratio check — contribute this shard's ratio to the
                // MAX reduce every step (symmetric with the coordinator) so it can
                // trigger a forced full rebuild on its own.  Must be ordered before
                // the do_rebuild broadcast.
                {
                    double my_ratio = subIndex.getTombstoneRatio();
                    double dummy_max = 0.0;
                    MPI_Reduce(&my_ratio, &dummy_max, 1, MPI_DOUBLE,
                               MPI_MAX, 0, MPI_COMM_WORLD);
                }
                int do_rebuild = 0;
                MPI_Bcast(&do_rebuild, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (do_rebuild) {
                    MessageHeader hdr;
                    comm.recv_header(hdr, 0);
                    if (hdr.type == INPLACE_REBUILD_REQUEST) {
                        subIndex.reBuildDelta(static_cast<int>(hdr.size), NCENTERS,
                                              world_size, NUM_BUILDING_THREADS);
                    } else {
                        subIndex.reBuild(static_cast<int>(hdr.size), NCENTERS,
                                         world_size, EF_CONSTRUCTION, M_SUB,
                                         NUM_BUILDING_THREADS);
                    }
                    // Sync updated routing state.
                    bcastRoutingState(rank, dim, nullptr, nullptr, nullptr,
                                      routing_hnsw, routing_partitions, label_to_center,
                                      &meta_space, false);
                    // Contribute per-phase timings to coordinator's MPI_Reduce(MAX).
                    double exec_send[3] = {
                        subIndex.getLastRebuildIterateS(),
                        subIndex.getLastRebuildExchangeS(),
                        subIndex.getLastRebuildGraphS()
                    };
                    double exec_recv[3]; // unused on non-root ranks
                    MPI_Reduce(exec_send, exec_recv, 3, MPI_DOUBLE, MPI_MAX,
                               0, MPI_COMM_WORLD);
                    // Delta rebuild only: contribute remaining deleted slots to
                    // the coordinator's SUM reduce.
                    if (hdr.type == INPLACE_REBUILD_REQUEST) {
                        long long del_send = static_cast<long long>(
                            subIndex.getLastRebuildRemainingDeleted());
                        long long del_recv = 0LL; // unused on non-root ranks
                        MPI_Reduce(&del_send, &del_recv, 1, MPI_LONG_LONG_INT,
                                   MPI_SUM, 0, MPI_COMM_WORLD);
                    }
                    // Sync label_to_shard: report arrived labels to all ranks.
                    sync_label_to_shard_after_rebuild_exec();
                }

                {
                    unsigned long long my_size = subIndex.getElementCount();
                    MPI_Gather(&my_size, 1, MPI_UNSIGNED_LONG_LONG,
                               nullptr,  1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
                }

            // ── SEARCH (executor side of sweep loop) ──────────────────────────
            } else if (op_code == 2) {
                std::vector<float> queries = readVecs(query_file, dim);
                const size_t nq    = queries.size() / dim;
                const size_t chunk = (nq + world_size - 1) / world_size;
                const size_t my_qs = static_cast<size_t>(rank) * chunk;
                const size_t my_qe = std::min(my_qs + chunk, nq);

                // Load GT for recall computation (matches coordinator's load).
                std::vector<std::vector<int>> gt;
                bool have_gt = false;
                if (!gt_prefix.empty()) {
                    int step_num_ex = bcast_buf[0];
                    std::string per_step =
                        gt_prefix + "/step" + std::to_string(step_num_ex) + ".gt100";
                    if (std::filesystem::exists(per_step)) {
                        gt      = readGT(per_step, file_format);
                        have_gt = !gt.empty();
                    }
                }

                // Receive center_counts broadcast from coordinator.
                {
                    int n = 0;
                    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    routing_center_counts.resize(n);
                    MPI_Bcast(routing_center_counts.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
                }

                // Pin the routing meta-HNSW search ef for this step (mirrors
                // coordinator).  Reset on every rebuild/bcastRoutingState, so it
                // must be set here each search step on every rank.
                routing_hnsw->setEf(EF_ROUTING);

                // ── Theoretical-recall setup (mirrors coordinator) ───────
                // Per-neighbour partition list (length kk, -1 for unresolved);
                // neighbor-frequency recall, divides by k.
                std::vector<std::vector<int>> gt_partitions(my_qe - my_qs);
                if (have_gt) {
                    BaseMmap base_view(base_file, file_format);
                    #pragma omp parallel
                    {
                        std::vector<float> tmp_vec;
                        #pragma omp for schedule(static)
                        for (size_t q = my_qs; q < my_qe; q++) {
                            if (q >= gt.size()) continue;
                            const auto& gt_q = gt[q];
                            const int   kk   = std::min(static_cast<int>(gt_q.size()), k);
                            auto&       glist = gt_partitions[q - my_qs];
                            glist.assign(kk, -1);
                            for (int i = 0; i < kk; i++) {
                                const int gid = gt_q[i];
                                if (gid < 0 || static_cast<uint32_t>(gid) >= base_view.n())
                                    continue;
                                base_view.readVector(static_cast<uint32_t>(gid), tmp_vec);
                                auto pq = routing_hnsw->searchKnn(tmp_vec.data(), 1);
                                if (pq.empty()) continue;
                                const int cid = static_cast<int>(pq.top().second);
                                if (cid < 0 || cid >= static_cast<int>(routing_partitions.size()))
                                    continue;
                                glist[i] = routing_partitions[cid];
                            }
                        }
                    }
                }

                // Precompute once: number of routing centers per partition (mirrors coordinator).
                std::vector<int> routing_part_size(static_cast<size_t>(num_partitions), 0);
                for (int p : routing_partitions) ++routing_part_size[static_cast<size_t>(p)];

                // Run once per combo — must match coordinator's combo loop exactly.
                for (int ci = 0; ci < n_combos; ci++) {
                    const auto& [sweep_mode_ex, sweep_param_ex] = sweep_combos[ci];

                    // Route this executor's query chunk.
                    std::vector<std::vector<uint32_t>> send_qids(world_size);
                    std::vector<std::vector<float>>    send_qvecs(world_size);
                    long long my_total_parts = 0;
                    // Theoretical-recall accumulator (mirrors coordinator).
                    double    my_theo_sum     = 0.0;
                    long long my_theo_counted = 0;

                    #pragma omp parallel
                    {
                        std::vector<std::vector<uint32_t>> local_qids(world_size);
                        std::vector<std::vector<float>>    local_qvecs(world_size);
                        long long thread_parts      = 0;
                        double    thread_theo_sum   = 0.0;
                        long long thread_theo_count = 0;

                        #pragma omp for nowait schedule(static)
                        for (size_t q = my_qs; q < my_qe; q++) {
                            float* Q = queries.data() + q * dim;
                            std::set<int> targets = routeQuery(
                                Q, routing_hnsw, routing_partitions,
                                num_partitions, sweep_mode_ex, sweep_param_ex,
                                routing_center_counts, routing_part_size);
                            thread_parts += static_cast<long long>(targets.size());
                            for (int tgt : targets) {
                                local_qids[tgt].push_back(static_cast<uint32_t>(q));
                                local_qvecs[tgt].insert(local_qvecs[tgt].end(), Q, Q + dim);
                            }

                            const auto& glist = gt_partitions[q - my_qs];
                            if (!glist.empty()) {
                                int inter = 0;
                                for (int pid : glist) {
                                    if (pid < 0) continue;
                                    if (targets.count(pid + 1)) ++inter;
                                }
                                thread_theo_sum   += static_cast<double>(inter)
                                                   / static_cast<double>(k);
                                thread_theo_count += 1;
                            }
                        }

                        #pragma omp critical
                        {
                            for (int r = 0; r < world_size; r++) {
                                send_qids[r].insert(send_qids[r].end(),
                                                    local_qids[r].begin(), local_qids[r].end());
                                send_qvecs[r].insert(send_qvecs[r].end(),
                                                     local_qvecs[r].begin(), local_qvecs[r].end());
                            }
                            my_total_parts  += thread_parts;
                            my_theo_sum     += thread_theo_sum;
                            my_theo_counted += thread_theo_count;
                        }
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    const double t0_srch = MPI_Wtime();

                    std::vector<std::vector<uint32_t>> recv_qids;
                    std::vector<std::vector<float>>    recv_qvecs;
                    AllToAllV(send_qids,  recv_qids,  MPI_UINT32_T, world_size, MPI_COMM_WORLD);
                    AllToAllV(send_qvecs, recv_qvecs, MPI_FLOAT,    world_size, MPI_COMM_WORLD);

                    // Search received queries and prepare return buffers.
                    std::vector<std::vector<uint32_t>> snd_rqids(world_size);
                    std::vector<std::vector<uint32_t>> snd_rids(world_size);
                    std::vector<std::vector<float>>    snd_rdists(world_size);

                    for (int src = 0; src < world_size; src++) {
                        if (recv_qids[src].empty()) continue;
                        const size_t nrecv     = recv_qids[src].size();
                        std::vector<float> flat_qvecs(recv_qvecs[src]);
                        auto results = subIndex.searchLocalBatch(flat_qvecs, nrecv, k);

                        for (size_t j = 0; j < nrecv; j++) {
                            const uint32_t qid   = recv_qids[src][j];
                            const int      owner = static_cast<int>(qid / chunk);
                            const auto&    res   = results[j];
                            snd_rqids[owner].push_back(qid);
                            for (int nn = 0; nn < k; nn++) {
                                if (nn < static_cast<int>(res.size())) {
                                    snd_rids[owner].push_back(
                                        static_cast<uint32_t>(res[nn].second));
                                    snd_rdists[owner].push_back(res[nn].first);
                                } else {
                                    snd_rids[owner].push_back(0);
                                    snd_rdists[owner].push_back(
                                        std::numeric_limits<float>::max());
                                }
                            }
                        }
                    }

                    std::vector<std::vector<uint32_t>> rcv_rqids;
                    std::vector<std::vector<uint32_t>> rcv_rids;
                    std::vector<std::vector<float>>    rcv_rdists;
                    AllToAllV(snd_rqids,  rcv_rqids,  MPI_UINT32_T, world_size, MPI_COMM_WORLD);
                    AllToAllV(snd_rids,   rcv_rids,   MPI_UINT32_T, world_size, MPI_COMM_WORLD);
                    AllToAllV(snd_rdists, rcv_rdists,  MPI_FLOAT,    world_size, MPI_COMM_WORLD);

                    {
                        double elapsed = MPI_Wtime() - t0_srch;
                        double max_t   = 0.0;
                        MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                    }

                    {
                        // Mirror coordinator's fused SUM reduce:
                        // (total_parts, theo_sum, theo_counted) as double[3].
                        double local_sum[3]  = {
                            static_cast<double>(my_total_parts),
                            my_theo_sum,
                            static_cast<double>(my_theo_counted),
                        };
                        double dummy_recv[3] = {0.0, 0.0, 0.0}; // unused on non-root
                        MPI_Reduce(local_sum, dummy_recv, 3, MPI_DOUBLE,
                                   MPI_SUM, 0, MPI_COMM_WORLD);
                    }

                    // Merge results and compute partial hits.
                    using KNNVec = std::vector<std::pair<float, uint32_t>>;
                    std::vector<KNNVec> neighbors(my_qe - my_qs);
                    for (int src = 0; src < world_size; src++) {
                        const size_t nres = rcv_rqids[src].size();
                        for (size_t j = 0; j < nres; j++) {
                            const uint32_t qid    = rcv_rqids[src][j];
                            const size_t q_local  = qid - my_qs;
                            if (q_local >= neighbors.size()) continue;
                            for (int nn = 0; nn < k; nn++) {
                                const size_t off = j * k + nn;
                                if (off >= rcv_rids[src].size()) break;
                                neighbors[q_local].push_back(
                                    {rcv_rdists[src][off], rcv_rids[src][off]});
                            }
                        }
                    }
                    for (auto& nv : neighbors) {
                        std::sort(nv.begin(), nv.end());
                        nv.erase(std::unique(nv.begin(), nv.end(),
                            [](const auto& a, const auto& b){ return a.second == b.second; }),
                            nv.end());
                        if (static_cast<int>(nv.size()) > k) nv.resize(k);
                    }

                    uint64_t my_hits = 0;
                    if (have_gt) {
                        for (size_t q = my_qs; q < my_qe; q++) {
                            if (q >= gt.size()) break;
                            const auto& gt_q  = gt[q];
                            const auto& res_q = neighbors[q - my_qs];
                            std::set<uint32_t> res_set;
                            for (auto& [d, lbl] : res_q) res_set.insert(lbl);
                            for (int j = 0; j < k && j < static_cast<int>(gt_q.size()); j++)
                                if (res_set.count(static_cast<uint32_t>(gt_q[j])))
                                    my_hits++;
                        }
                    }
                    uint64_t dummy_hits = 0;
                    MPI_Reduce(&my_hits, &dummy_hits, 1, MPI_UINT64_T,
                               MPI_SUM, 0, MPI_COMM_WORLD);

                } // end executor combo loop

                // No rebuild after search.
                int rb_type = 0;
                MPI_Bcast(&rb_type, 1, MPI_INT, 0, MPI_COMM_WORLD);

                // Participate in shard size gather.
                {
                    unsigned long long my_size = subIndex.getElementCount();
                    MPI_Gather(&my_size, 1, MPI_UNSIGNED_LONG_LONG,
                               nullptr,  1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
                }

            } else {
                int rb_type = 0;
                MPI_Bcast(&rb_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
                unsigned long long my_size = subIndex.getElementCount();
                MPI_Gather(&my_size, 1, MPI_UNSIGNED_LONG_LONG,
                           nullptr,  1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            }

            // ── CHECKPOINT ────────────────────────────────────────────────────
            // Symmetric with the coordinator block above.  Every executor saves
            // its sub-HNSW shard, then joins the barrier so rank 0 knows all
            // writes are complete before it removes the previous checkpoint.
            if (s % CHECKPOINT_INTERVAL == 0) {
                const std::string ckpt_dir =
                    ckpt_base + "/ckpt_s" + std::to_string(s);

                // Ensure the checkpoint directory exists (create_directories is
                // idempotent; the coordinator may have created it first via NFS,
                // or we may arrive first — either way is fine).
                std::filesystem::create_directories(ckpt_dir);

                // Save this executor's sub-HNSW shard.
                // Executor::save(prefix) appends "_<rank>.bin" to prefix and
                // returns the full path written.
                const std::string shard_path = subIndex.save(ckpt_dir + "/shard");

                // Verify the shard file was written successfully.
                std::error_code shard_ec;
                const auto shard_sz = std::filesystem::file_size(shard_path, shard_ec);
                int my_ckpt_write_ok = (!shard_ec && shard_sz > 0) ? 1 : 0;
                if (!my_ckpt_write_ok)
                    std::cerr << "[Sweep Executor " << rank
                              << "] WARNING: shard write failed or empty: "
                              << shard_path << " (disk full?)\n";

                // ── Barrier + all-ranks write validation ──────────────────────
                // Matches the coordinator's barrier exactly.
                MPI_Barrier(MPI_COMM_WORLD);
                int all_ckpt_writes_ok = 0;
                MPI_Allreduce(&my_ckpt_write_ok, &all_ckpt_writes_ok,
                              1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

                // ── Delete old checkpoint (this rank's local copy) ────────────
                // Only delete if ALL ranks confirmed successful writes.
                // ~/surge may be local to each node rather than NFS-shared, so
                // every rank removes the old checkpoint independently.
                // On NFS this is idempotent (remove_all on a missing path is a no-op).
                if (all_ckpt_writes_ok && !last_ckpt_dir.empty()) {
                    std::error_code ec;
                    std::filesystem::remove_all(last_ckpt_dir, ec);
                }
                last_ckpt_dir = ckpt_dir;
            }

        } // end executor streaming loop

        std::cout << "[Sweep Executor " << rank << "] Done.\n";
    }

    if (rank > 0 && routing_hnsw != nullptr) delete routing_hnsw;

    MPI_Finalize();
    return 0;
}
