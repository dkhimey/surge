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
// ─── Sweep parameter grids ───────────────────────────────────────────────────
//  BranchingFactor : {1, 2, 5, 10, 15, 20, 25, 30}
//  NProbe          : {1, 2, 3, 4, 5, 6, 7, 8, 9}
//  RecallTarget    : {0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99}
//
// ─── Output CSV columns ──────────────────────────────────────────────────────
//  All rows:
//    step, operation, mode, param,
//    range_start, range_end, time_s, throughput, recall@<k>, avg_parts_searched
//  Rebuild rows (operation="rebuild") additionally populate:
//    centers_moved, elements_moved,
//    repart_hnsw_s, repart_bottom_s, repart_kaffpa_s, repart_relabel_s,
//    exec_iterate_s, exec_exchange_s, exec_graph_s, dorebuild_wall_s
//  (non-rebuild rows carry -1 in those rebuild-specific columns)
//  All rows end with shard_0_size, …, shard_<P-1>_size
//
//  For non-search rows, mode="" and param=-1.

#include <algorithm>
#include <atomic>
#include <cassert>
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

#include "index.h"

// ─── Build hyper-parameters ──────────────────────────────────────────────────
static constexpr int    NCENTERS             = 10000;
static constexpr int    EF_CONSTRUCTION      = 200;
static constexpr int    M_META               = 16;
static constexpr int    M_SUB                = 16;
static constexpr int    NUM_BUILDING_THREADS = 32;
static constexpr int    EF_SEARCH            = 200;
static constexpr size_t SAMPLE_SIZE          = 100000;

// ─── Sweep parameter grids (visible to all ranks so loop counts agree) ───────
static const std::vector<int>   BRANCHING_FACTOR_PARAMS = {1, 2, 5, 10, 15, 20, 25, 30};
static const std::vector<int>   NPROBE_PARAMS           = {1, 2, 3, 4, 5, 6, 7, 8, 9};
static const std::vector<float> TARGET_PARAMS           = {.6f, .7f, .75f, .8f, .85f,
                                                            .9f, .95f, .97f, .98f, .99f};

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
    float                            param)
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

        std::vector<double> part_probs(static_cast<size_t>(num_partitions), 0.0);
        for (size_t r = 0; r < centers.size(); r++) {
            const float d   = centers[r].first;
            const int   pid = partitions[static_cast<int>(centers[r].second)];
            const double w  = 1.0 / std::pow(static_cast<double>(d) + 1e-5, 3.0);
            part_probs[static_cast<size_t>(pid)] += w / static_cast<double>(r + 1);
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

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    if (argc != 7) {
        std::cerr
            << "Usage: " << argv[0]
            << " <dataset> <num_partitions>"
            << " <full_threshold>"
            << " <k> <gt_prefix> <output_file>\n"
            << "\n"
            << "  full_threshold : min centers to migrate to trigger a full rebuild;\n"
            << "                   set >= " << NCENTERS << " to disable\n"
            << "  gt_prefix      : directory with per-step GT files (step<N>.gt100),"
               " or \"\" for static GT only\n";
        return 1;
    }

    const std::string dataset_name   = argv[1];
    const int         num_partitions = std::stoi(argv[2]);
    const int         full_threshold = std::stoi(argv[3]);
    const int         partial_threshold = full_threshold; // partial rebuilds disabled
    const int         k              = std::stoi(argv[4]);
    const std::string gt_prefix      = argv[5];
    const std::string output_file    = argv[6];

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
    if (steps.empty() || steps[0].operation != "insert" || steps[0].start != 0) {
        if (rank == 0) std::cerr << "ERROR: runbook must begin with insert at start=0\n";
        MPI_Finalize(); return 1;
    }

    // ── Log + communicator ────────────────────────────────────────────────────
    std::string log_id = "shared_sweep_" + dataset_name + "_" + std::to_string(num_partitions);
    Log logger(log_id);
    Communicator comm;

    // ── Shared routing state (all ranks) ─────────────────────────────────────
    std::unordered_map<int,int>       label_to_center;
    std::vector<int>                  routing_partitions;
    hnswlib::L2Space                  meta_space(dim);
    hnswlib::HierarchicalNSW<float>*  routing_hnsw = nullptr;

    // ══════════════════════════════════════════════════════════════════════════
    //                         PHASE 1 : INITIAL BUILD
    // ══════════════════════════════════════════════════════════════════════════

    if (rank == 0) {
        const RunbookStep& init = steps[0];
        const int init_n = init.end;

        const size_t ss = std::min(static_cast<size_t>(init_n), SAMPLE_SIZE);
        std::vector<float> sample = getSample(base_file, init_n, dim, ss);

        Coordinator metaIndex(dim, &comm, &logger);
        metaIndex.setSampleData(sample.data(), ss);
        metaIndex.build(NCENTERS, num_partitions, EF_CONSTRUCTION, M_META);

        std::cout << "[Sweep] Distributing initial " << init_n << " vectors\n";
        metaIndex.distribute_vectors(base_file, init_n, false, NUM_BUILDING_THREADS);
        comm.broadcast_termination(world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "[Sweep] Initial build complete\n";

        comm.broadcast_ef_search(EF_SEARCH, world_size);

        routing_hnsw = metaIndex.getMetaHNSW();

        bcastRoutingState(rank, dim,
                          metaIndex.getMetaHNSW(),
                          &metaIndex.getPartitions(),
                          &metaIndex.getLabelToCenter(),
                          routing_hnsw, routing_partitions, label_to_center,
                          &meta_space, /*include_ltc=*/true);

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
        std::ofstream csv(output_file, std::ios::out);
        if (!csv.is_open()) {
            std::cerr << "ERROR: cannot open output file: " << output_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        csv << "step,operation,mode,param"
            << ",range_start,range_end,time_s,throughput,recall@" << k
            << ",avg_parts_searched"
            << ",centers_moved,elements_moved"
            << ",repart_hnsw_s,repart_bottom_s,repart_kaffpa_s,repart_relabel_s"
            << ",exec_iterate_s,exec_exchange_s,exec_graph_s,dorebuild_wall_s";
        for (int i = 0; i < num_partitions; i++) csv << ",shard_" << i << "_size";
        csv << "\n";

        // ── Rebuild statistics struct ─────────────────────────────────────────
        struct RebuildStats {
            int    centers_moved    = -1;
            int    elements_moved   = -1;
            double repart_hnsw_s    = -1.0;
            double repart_bottom_s  = -1.0;
            double repart_kaffpa_s  = -1.0;
            double repart_relabel_s = -1.0;
            double exec_iterate_s   = -1.0;
            double exec_exchange_s  = -1.0;
            double exec_graph_s     = -1.0;
            double dorebuild_wall_s = -1.0;
        };
        static const RebuildStats kNoRebuild{};

        // ── Helper: write one CSV row ─────────────────────────────────────────
        auto write_row = [&](int step_num, const std::string& op,
                             const std::string& mode_str, float param_val,
                             int rs, int re,
                             double time_s, double throughput, double recall,
                             double avg_parts,
                             const RebuildStats& rb,
                             const std::vector<unsigned long long>& sizes)
        {
            csv << step_num << "," << op
                << "," << mode_str << "," << param_val
                << "," << rs << "," << re
                << "," << time_s << "," << throughput
                << "," << recall
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
                << "," << rb.dorebuild_wall_s;
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

        // Initial insert row (build not timed as a streaming operation).
        {
            auto sizes = collect_sizes();
            write_row(init.step_num, "insert", "", -1.0f,
                      init.start, init.end, 0.0, 0.0, -1.0, -1.0,
                      kNoRebuild, sizes);
        }

        // ══════════════════════════════════════════════════════════════════════
        //                    COORDINATOR STREAMING LOOP
        // ══════════════════════════════════════════════════════════════════════
        for (size_t s = 1; s < steps.size(); s++) {
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

                // Rebuild check.
                int rb_type = metaIndex.checkNeedRebuild(
                    full_threshold, partial_threshold, EF_CONSTRUCTION, M_META);
                MPI_Bcast(&rb_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
                RebuildStats rb_stats;
                if (rb_type > 0) {
                    rb_stats.centers_moved    = metaIndex.getCachedCentersMoved();
                    rb_stats.elements_moved   = metaIndex.getCachedElementsMoved();
                    rb_stats.repart_hnsw_s    = metaIndex.getCachedRepartHnswS();
                    rb_stats.repart_bottom_s  = metaIndex.getCachedRepartBottomS();
                    rb_stats.repart_kaffpa_s  = metaIndex.getCachedRepartKaffpaS();
                    rb_stats.repart_relabel_s = metaIndex.getCachedRepartRelabelS();

                    const double t_rb0 = MPI_Wtime();
                    metaIndex.doRebuildSimple(world_size);
                    rb_stats.dorebuild_wall_s = MPI_Wtime() - t_rb0;

                    sync_routing_after_rebuild();

                    double exec_send[3] = {0.0, 0.0, 0.0};
                    double exec_recv[3] = {0.0, 0.0, 0.0};
                    MPI_Reduce(exec_send, exec_recv, 3, MPI_DOUBLE, MPI_MAX,
                               0, MPI_COMM_WORLD);
                    rb_stats.exec_iterate_s  = exec_recv[0];
                    rb_stats.exec_exchange_s = exec_recv[1];
                    rb_stats.exec_graph_s    = exec_recv[2];

                    const double total_rb_s = rb_stats.repart_hnsw_s + rb_stats.repart_bottom_s
                                           + rb_stats.repart_kaffpa_s + rb_stats.repart_relabel_s
                                           + rb_stats.dorebuild_wall_s;
                    std::cout << "[Sweep] Step " << step.step_num
                              << "  rebuild type=" << rb_type
                              << "  centers_moved=" << rb_stats.centers_moved
                              << "  elements_moved=" << rb_stats.elements_moved
                              << "  total_rebuild_s=" << total_rb_s << "\n";
                }

                const double throughput = (max_t > 0.0)
                    ? static_cast<double>(n_insert) / max_t : 0.0;
                auto sizes = collect_sizes();
                write_row(step.step_num, "insert", "", -1.0f,
                          step.start, step.end, max_t, throughput, -1.0, -1.0,
                          kNoRebuild, sizes);
                if (rb_type > 0) {
                    const double total_rb_s = rb_stats.repart_hnsw_s + rb_stats.repart_bottom_s
                                           + rb_stats.repart_kaffpa_s + rb_stats.repart_relabel_s
                                           + rb_stats.dorebuild_wall_s;
                    write_row(step.step_num, "rebuild", "", -1.0f, -1, -1,
                              total_rb_s, -1.0, -1.0, -1.0, rb_stats, sizes);
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
                    // Coordinator has no shard — only erases from label_to_center.
                    label_to_center.erase(step.start + i);
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
                MPI_Bcast(&rb_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
                RebuildStats rb_stats;
                if (rb_type > 0) {
                    rb_stats.centers_moved    = metaIndex.getCachedCentersMoved();
                    rb_stats.elements_moved   = metaIndex.getCachedElementsMoved();
                    rb_stats.repart_hnsw_s    = metaIndex.getCachedRepartHnswS();
                    rb_stats.repart_bottom_s  = metaIndex.getCachedRepartBottomS();
                    rb_stats.repart_kaffpa_s  = metaIndex.getCachedRepartKaffpaS();
                    rb_stats.repart_relabel_s = metaIndex.getCachedRepartRelabelS();

                    const double t_rb0 = MPI_Wtime();
                    metaIndex.doRebuildSimple(world_size);
                    rb_stats.dorebuild_wall_s = MPI_Wtime() - t_rb0;

                    sync_routing_after_rebuild();

                    double exec_send[3] = {0.0, 0.0, 0.0};
                    double exec_recv[3] = {0.0, 0.0, 0.0};
                    MPI_Reduce(exec_send, exec_recv, 3, MPI_DOUBLE, MPI_MAX,
                               0, MPI_COMM_WORLD);
                    rb_stats.exec_iterate_s  = exec_recv[0];
                    rb_stats.exec_exchange_s = exec_recv[1];
                    rb_stats.exec_graph_s    = exec_recv[2];

                    const double total_rb_s = rb_stats.repart_hnsw_s + rb_stats.repart_bottom_s
                                           + rb_stats.repart_kaffpa_s + rb_stats.repart_relabel_s
                                           + rb_stats.dorebuild_wall_s;
                    std::cout << "[Sweep] Step " << step.step_num
                              << "  rebuild type=" << rb_type
                              << "  centers_moved=" << rb_stats.centers_moved
                              << "  elements_moved=" << rb_stats.elements_moved
                              << "  total_rebuild_s=" << total_rb_s << "\n";
                }

                const double throughput = (max_t > 0.0)
                    ? static_cast<double>(n_delete) / max_t : 0.0;
                auto sizes = collect_sizes();
                write_row(step.step_num, "delete", "", -1.0f,
                          step.start, step.end, max_t, throughput, -1.0, -1.0,
                          kNoRebuild, sizes);
                if (rb_type > 0) {
                    const double total_rb_s = rb_stats.repart_hnsw_s + rb_stats.repart_bottom_s
                                           + rb_stats.repart_kaffpa_s + rb_stats.repart_relabel_s
                                           + rb_stats.dorebuild_wall_s;
                    write_row(step.step_num, "rebuild", "", -1.0f, -1, -1,
                              total_rb_s, -1.0, -1.0, -1.0, rb_stats, sizes);
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

                // ── Combo loop (results buffered; collect_sizes() called once after) ──
                // Declared before the loop because ComboResult must live that long.
                struct ComboResult {
                    std::string mode_str;
                    float       param;
                    double      time_s;
                    double      qps;
                    double      recall;
                    double      avg_parts;
                };
                std::vector<ComboResult> combo_results;
                combo_results.reserve(n_combos);

                for (int ci = 0; ci < n_combos; ci++) {
                    const auto& [sweep_mode, sweep_param] = sweep_combos[ci];
                    const std::string sweep_mode_str = mode_to_string(sweep_mode);

                    std::vector<std::vector<uint32_t>> send_qids(world_size);
                    std::vector<std::vector<float>>    send_qvecs(world_size);
                    long long my_total_parts = 0;

                    #pragma omp parallel
                    {
                        std::vector<std::vector<uint32_t>> local_qids(world_size);
                        std::vector<std::vector<float>>    local_qvecs(world_size);
                        long long thread_parts = 0;

                        #pragma omp for nowait schedule(static)
                        for (size_t q = my_qs; q < my_qe; q++) {
                            float* Q = queries.data() + q * dim;
                            std::set<int> targets = routeQuery(
                                Q, routing_hnsw, routing_partitions,
                                num_partitions, sweep_mode, sweep_param);
                            thread_parts += static_cast<long long>(targets.size());
                            for (int tgt : targets) {
                                local_qids[tgt].push_back(static_cast<uint32_t>(q));
                                local_qvecs[tgt].insert(local_qvecs[tgt].end(), Q, Q + dim);
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
                            my_total_parts += thread_parts;
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

                    long long global_parts = 0;
                    MPI_Reduce(&my_total_parts, &global_parts, 1, MPI_LONG_LONG,
                               MPI_SUM, 0, MPI_COMM_WORLD);
                    const double avg_parts =
                        (nq > 0) ? static_cast<double>(global_parts) / static_cast<double>(nq)
                                 : -1.0;

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
                              << "  qps=" << qps
                              << "  avg_parts=" << avg_parts << "\n";

                    combo_results.push_back({sweep_mode_str, sweep_param,
                                             max_t, qps, recall, avg_parts});
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
                              cr.time_s, cr.qps, cr.recall, cr.avg_parts,
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
        } // end coordinator streaming loop

        csv.close();
        std::cout << "[Sweep] Done. Results written to " << output_file << "\n";

    // ══════════════════════════════════════════════════════════════════════════
    //                        EXECUTOR (rank > 0)
    // ══════════════════════════════════════════════════════════════════════════
    } else {

        Executor subIndex(rank, dim, comm, &logger);

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

        bcastRoutingState(rank, dim, nullptr, nullptr, nullptr,
                          routing_hnsw, routing_partitions, label_to_center,
                          &meta_space, /*include_ltc=*/true);

        // Participate in initial-row size gather.
        {
            unsigned long long my_size = subIndex.getElementCount();
            MPI_Gather(&my_size, 1, MPI_UNSIGNED_LONG_LONG,
                       nullptr,  1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        }

        // ══════════════════════════════════════════════════════════════════════
        //                    EXECUTOR STREAMING LOOP
        // ══════════════════════════════════════════════════════════════════════
        for (size_t s = 1; s < steps.size(); s++) {
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
                }

                {
                    double elapsed = MPI_Wtime() - t0_ex;
                    double max_t   = 0.0;
                    MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                }

                int rb_type = 0;
                MPI_Bcast(&rb_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (rb_type > 0) {
                    MessageHeader hdr;
                    comm.recv_header(hdr, 0);
                    subIndex.reBuild(static_cast<int>(hdr.size), NCENTERS,
                                     world_size, EF_CONSTRUCTION, M_SUB,
                                     NUM_BUILDING_THREADS);
                    bcastRoutingState(rank, dim, nullptr, nullptr, nullptr,
                                      routing_hnsw, routing_partitions, label_to_center,
                                      &meta_space, false);
                    double exec_send[3] = {
                        subIndex.getLastRebuildIterateS(),
                        subIndex.getLastRebuildExchangeS(),
                        subIndex.getLastRebuildGraphS()
                    };
                    double exec_recv[3];
                    MPI_Reduce(exec_send, exec_recv, 3, MPI_DOUBLE, MPI_MAX,
                               0, MPI_COMM_WORLD);
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
                    if (routing_partitions[cid] + 1 == rank)
                        my_delete_labels.push_back(label);
                    label_to_center.erase(label);
                }
                subIndex.markDeleteLocalBatch(my_delete_labels);

                {
                    double elapsed = MPI_Wtime() - t0_del;
                    double max_t   = 0.0;
                    MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                }

                int rb_type = 0;
                MPI_Bcast(&rb_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (rb_type > 0) {
                    MessageHeader hdr;
                    comm.recv_header(hdr, 0);
                    subIndex.reBuild(static_cast<int>(hdr.size), NCENTERS,
                                     world_size, EF_CONSTRUCTION, M_SUB,
                                     NUM_BUILDING_THREADS);
                    bcastRoutingState(rank, dim, nullptr, nullptr, nullptr,
                                      routing_hnsw, routing_partitions, label_to_center,
                                      &meta_space, false);
                    double exec_send[3] = {
                        subIndex.getLastRebuildIterateS(),
                        subIndex.getLastRebuildExchangeS(),
                        subIndex.getLastRebuildGraphS()
                    };
                    double exec_recv[3];
                    MPI_Reduce(exec_send, exec_recv, 3, MPI_DOUBLE, MPI_MAX,
                               0, MPI_COMM_WORLD);
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

                // Run once per combo — must match coordinator's combo loop exactly.
                for (int ci = 0; ci < n_combos; ci++) {
                    const auto& [sweep_mode_ex, sweep_param_ex] = sweep_combos[ci];

                    // Route this executor's query chunk.
                    std::vector<std::vector<uint32_t>> send_qids(world_size);
                    std::vector<std::vector<float>>    send_qvecs(world_size);
                    long long my_total_parts = 0;

                    #pragma omp parallel
                    {
                        std::vector<std::vector<uint32_t>> local_qids(world_size);
                        std::vector<std::vector<float>>    local_qvecs(world_size);
                        long long thread_parts = 0;

                        #pragma omp for nowait schedule(static)
                        for (size_t q = my_qs; q < my_qe; q++) {
                            float* Q = queries.data() + q * dim;
                            std::set<int> targets = routeQuery(
                                Q, routing_hnsw, routing_partitions,
                                num_partitions, sweep_mode_ex, sweep_param_ex);
                            thread_parts += static_cast<long long>(targets.size());
                            for (int tgt : targets) {
                                local_qids[tgt].push_back(static_cast<uint32_t>(q));
                                local_qvecs[tgt].insert(local_qvecs[tgt].end(), Q, Q + dim);
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
                            my_total_parts += thread_parts;
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
                        long long dummy_recv = 0;
                        MPI_Reduce(&my_total_parts, &dummy_recv, 1, MPI_LONG_LONG,
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
        } // end executor streaming loop

        std::cout << "[Sweep Executor " << rank << "] Done.\n";
    }

    if (rank > 0 && routing_hnsw != nullptr) delete routing_hnsw;

    MPI_Finalize();
    return 0;
}
