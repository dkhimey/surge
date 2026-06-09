// shared_static_experiment.cpp
//
// Static-index throughput benchmark that replaces static_routing_qps_single.cpp.
//
// Key improvements over static_routing_qps_single.cpp:
//  - All MPI ranks hold a local copy of the routing state (meta-HNSW +
//    partitions) and participate in query dispatch via AllToAllV, matching the
//    setup of shared_batch_experiment_sweep.cpp and gp-ann's distributed bench.
//    In the old experiment the coordinator dispatched queries serially over the
//    MPI message-passing protocol (handle_query); the coordinator became the
//    bottleneck and executor parallelism was under-utilised.
//  - Timing follows gp-ann's distributed_bench.cpp: every rank measures its
//    own wall time, then MPI_Reduce(MAX) gives the true end-to-end time.
//  - Sweeps all three routing modes and their full parameter grids, identical
//    to the grids in shared_batch_experiment_sweep.cpp.
//  - Loads a pre-built static index; no insert / delete / rebuild logic.
//
// ─── Usage ───────────────────────────────────────────────────────────────────
//   mpirun -np <P+1> ./shared_static_experiment \
//       <dataset> <num_partitions> <k> <output_file>
//
// ─── Sweep parameter grids ───────────────────────────────────────────────────
//   BranchingFactor : {1, 2, 5, 10, 15, 20, 25, 30}
//   NProbe          : {1, 2, 3, 4, 5, 6, 7, 8, 9}
//   RecallTarget    : {0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99}
//
// ─── Output CSV columns ──────────────────────────────────────────────────────
//   mode, param, recall@<k>, qps, avg_parts_searched

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "index.h"

// ─── Hyper-parameters ────────────────────────────────────────────────────────
static constexpr int EF_SEARCH = 200;
static constexpr int NUM_RUNS  = 10;  // timed passes per (mode, param) combo

// ─── Sweep grids (must match shared_batch_experiment_sweep.cpp) ───────────────
static const std::vector<int>   BRANCHING_FACTOR_PARAMS = {1, 2, 5, 10, 15, 20, 25, 30};
static const std::vector<int>   NPROBE_PARAMS           = {1, 2, 3, 4, 5, 6, 7, 8, 9};
static const std::vector<float> TARGET_PARAMS           = {.6f, .7f, .75f, .8f, .85f,
                                                            .9f, .95f, .97f, .98f, .99f};

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

// ─── routeQuery ──────────────────────────────────────────────────────────────
// Identical to shared_batch_experiment_sweep.cpp — kept here so this file
// compiles standalone without a shared header.
static std::set<int> routeQuery(
    float*                           vec,
    hnswlib::HierarchicalNSW<float>* hnsw,
    const std::vector<int>&          partitions,
    int                              num_partitions,
    RoutingMode                      mode,
    float                            param,
    const std::vector<int>&          center_counts)  // |rho(c)|: #vectors each centroid owns
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
        int    nprobe = static_cast<int>(param);
        size_t cur_k  = static_cast<size_t>(nprobe);
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
        // Fetch a deep candidate list of the closest centroids, then trim its tail
        // so that the kept prefix is expected to contain `recall_target` of the
        // query's true neighbours.
        //
        // Distance is used ONLY to order the candidates (nearest first).  The cut
        // point is driven by the requested recall: each centroid is expected to hold
        // a share of the query's neighbours proportional to the number of vectors it
        // owns, |rho(c)| (center_counts).  Walking nearest-first, we accumulate that
        // captured-neighbour fraction and stop once it reaches recall_target —
        // exactly the quantity the oracle accumulates from the ground-truth
        // partition distribution, estimated here by per-centroid point mass.
        float recall_target = std::clamp(param, 0.0f, 1.0f);
        const size_t ncenters = hnsw->getCurrentElementCount();
        const size_t knn      = std::min<size_t>(40, ncenters);
        auto centers = hnsw->searchKnnCloserFirst(vec, knn);   // ascending by distance
        if (centers.empty()) return target_ranks;

        // Total point mass over the candidate list = denominator of the recall est.
        double total_mass = 0.0;
        for (const auto& c : centers)
            total_mass += static_cast<double>(center_counts[static_cast<int>(c.second)]);
        if (total_mass <= 0.0) {
            target_ranks.insert(partitions[static_cast<int>(centers[0].second)] + 1);
            return target_ranks;
        }

        // Keep centroids (and their partitions) nearest-first until the cumulative
        // captured-neighbour fraction reaches the requested recall; trim the rest.
        double captured = 0.0;
        for (size_t r = 0; r < centers.size(); r++) {
            const int center_id = static_cast<int>(centers[r].second);
            target_ranks.insert(partitions[center_id] + 1);
            captured += static_cast<double>(center_counts[center_id]) / total_mass;
            if (captured >= recall_target) break;
        }
    }

    return target_ranks;
}

// ─── AllToAllV helper ─────────────────────────────────────────────────────────
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

// ─── bcastRoutingState ────────────────────────────────────────────────────────
// Broadcasts the coordinator's meta-HNSW and partition assignment to every
// other rank so they can route queries locally.
static void bcastRoutingState(
    int                              rank,
    int                              dim,
    hnswlib::HierarchicalNSW<float>* coord_meta,   // non-null only on rank 0
    const std::vector<int>*          coord_parts,  // non-null only on rank 0
    const std::vector<int>*          coord_counts, // non-null only on rank 0
    hnswlib::HierarchicalNSW<float>*& out_meta,
    std::vector<int>&                 out_parts,
    std::vector<int>&                 out_counts,
    hnswlib::SpaceInterface<float>*   space)
{
    // 1. Partitions vector
    {
        int n = (rank == 0) ? static_cast<int>(coord_parts->size()) : 0;
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) out_parts.resize(n);
        MPI_Bcast(
            (rank == 0) ? const_cast<int*>(coord_parts->data()) : out_parts.data(),
            n, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) out_parts = *coord_parts;
    }
    // 1b. Per-centroid point counts |rho(c)| (same indexing as partitions)
    {
        int n = (rank == 0) ? static_cast<int>(coord_counts->size()) : 0;
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) out_counts.resize(n);
        MPI_Bcast(
            (rank == 0) ? const_cast<int*>(coord_counts->data()) : out_counts.data(),
            n, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) out_counts = *coord_counts;
    }
    // 2. Meta-HNSW serialised bytes
    {
        int hnsw_size = 0;
        std::vector<char> buf;
        if (rank == 0) {
            coord_meta->saveIndex("tmp_shared_static_bcast.bin");
            std::ifstream f("tmp_shared_static_bcast.bin", std::ios::binary);
            buf.assign(std::istreambuf_iterator<char>(f), {});
            hnsw_size = static_cast<int>(buf.size());
        }
        MPI_Bcast(&hnsw_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) buf.resize(hnsw_size);
        MPI_Bcast(buf.data(), hnsw_size, MPI_BYTE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            const std::string tmp = "tmp_shared_static_r" + std::to_string(rank) + ".bin";
            { std::ofstream f(tmp, std::ios::binary); f.write(buf.data(), hnsw_size); }
            delete out_meta;
            out_meta = new hnswlib::HierarchicalNSW<float>(space, tmp);
            out_meta->setEf(EF_SEARCH);
        }
    }
}

// ─── runSearchPass ────────────────────────────────────────────────────────────
// One full pass over all queries for a single (mode, param) combo.
//
// All ranks participate:
//   Phase 1 (routing): each rank routes its query chunk → send_qids / send_qvecs
//   Phase 2 (dispatch): AllToAllV sends (qid, qvec) to target executor ranks
//   Phase 3 (search):  executor ranks call searchLocalBatch on received queries;
//                      coordinator sends empty buffers (it has no shard)
//   Phase 4 (return):  AllToAllV returns (qid, result_ids, result_dists)
//   Phase 5 (merge):   each rank merges its results and (optionally) scores recall
//
// Returns per-query part counts (for avg_parts_searched) and hit count.
// Timing is the caller's responsibility (barrier + MPI_Wtime).
static void runSearchPass(
    int                              rank,
    int                              world_size,
    int                              num_partitions,
    int                              k,
    size_t                           nq,
    size_t                           my_qs,
    size_t                           my_qe,
    const std::vector<float>&        queries,
    int                              dim,
    hnswlib::HierarchicalNSW<float>* routing_hnsw,
    const std::vector<int>&          routing_partitions,
    const std::vector<int>&          routing_counts,   // |rho(c)| per centroid
    RoutingMode                      mode,
    float                            param,
    Executor*                        sub_index,        // null on coordinator
    const std::vector<std::vector<int>>* gt,           // null → skip recall
    // outputs:
    long long&  out_total_parts,
    uint64_t&   out_hits,
    // ── per-rank diagnostics (to separate routing overhead from load skew) ──
    double&     out_route_time,    // Phase-1 local routing compute wall time (s)
    double&     out_search_time,   // Phase-3 local shard-search wall time (s)
    long long&  out_search_load)   // # queries this rank searched (column sum)
{
    out_route_time  = 0.0;
    out_search_time = 0.0;
    out_search_load = 0;
    // ── Phase 1: local routing ────────────────────────────────────────────────
    std::vector<std::vector<uint32_t>> send_qids(world_size);
    std::vector<std::vector<float>>    send_qvecs(world_size);
    long long my_total_parts = 0;

    const double t_route0 = MPI_Wtime();
    #pragma omp parallel
    {
        std::vector<std::vector<uint32_t>> local_qids(world_size);
        std::vector<std::vector<float>>    local_qvecs(world_size);
        long long thread_parts = 0;

        #pragma omp for nowait schedule(static)
        for (size_t q = my_qs; q < my_qe; q++) {
            float* Q = const_cast<float*>(queries.data()) + q * dim;
            std::set<int> targets = routeQuery(
                Q, routing_hnsw, routing_partitions,
                num_partitions, mode, param, routing_counts);
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
    out_route_time = MPI_Wtime() - t_route0;

    // ── Phase 2: dispatch queries to executors ────────────────────────────────
    std::vector<std::vector<uint32_t>> recv_qids;
    std::vector<std::vector<float>>    recv_qvecs;
    AllToAllV(send_qids,  recv_qids,  MPI_UINT32_T, world_size, MPI_COMM_WORLD);
    AllToAllV(send_qvecs, recv_qvecs, MPI_FLOAT,    world_size, MPI_COMM_WORLD);

    // ── Phase 3: search + prepare return buffers ──────────────────────────────
    const size_t chunk = (nq + world_size - 1) / world_size;

    std::vector<std::vector<uint32_t>> snd_rqids(world_size);
    std::vector<std::vector<uint32_t>> snd_rids(world_size);
    std::vector<std::vector<float>>    snd_rdists(world_size);

    const double t_search0 = MPI_Wtime();
    long long my_search_load = 0;
    if (sub_index != nullptr) {
        // Executor: search all received query batches.
        for (int src = 0; src < world_size; src++) {
            if (recv_qids[src].empty()) continue;
            const size_t nrecv = recv_qids[src].size();
            my_search_load += static_cast<long long>(nrecv);
            auto results = sub_index->searchLocalBatch(recv_qvecs[src], nrecv, k);

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
    }
    out_search_time = MPI_Wtime() - t_search0;
    out_search_load = my_search_load;
    // Coordinator: snd_r* buffers remain empty — it holds no shard.

    // ── Phase 4: return results to query owners ───────────────────────────────
    std::vector<std::vector<uint32_t>> rcv_rqids;
    std::vector<std::vector<uint32_t>> rcv_rids;
    std::vector<std::vector<float>>    rcv_rdists;
    AllToAllV(snd_rqids,  rcv_rqids,  MPI_UINT32_T, world_size, MPI_COMM_WORLD);
    AllToAllV(snd_rids,   rcv_rids,   MPI_UINT32_T, world_size, MPI_COMM_WORLD);
    AllToAllV(snd_rdists, rcv_rdists,  MPI_FLOAT,    world_size, MPI_COMM_WORLD);

    // ── Phase 5: merge + recall ───────────────────────────────────────────────
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
    if (gt != nullptr) {
        for (size_t q = my_qs; q < my_qe; q++) {
            if (q >= gt->size()) break;
            const auto& gt_q  = (*gt)[q];
            const auto& res_q = neighbors[q - my_qs];
            std::set<uint32_t> res_set;
            for (auto& [d, lbl] : res_q) res_set.insert(lbl);
            for (int j = 0; j < k && j < static_cast<int>(gt_q.size()); j++)
                if (res_set.count(static_cast<uint32_t>(gt_q[j])))
                    my_hits++;
        }
    }

    out_total_parts = my_total_parts;
    out_hits        = my_hits;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <num_partitions> <k> <output_file>\n";
        return 1;
    }

    const std::string dataset_name   = argv[1];
    const int         num_partitions = std::stoi(argv[2]);
    const int         k              = std::stoi(argv[3]);
    const std::string output_file    = argv[4];

    const auto sweep_combos = build_sweep_combos();
    const int  n_combos     = static_cast<int>(sweep_combos.size());

    // ── MPI init ──────────────────────────────────────────────────────────────
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

    // ── Dataset info ──────────────────────────────────────────────────────────
    const std::string base_file  = DATASETS[dataset_name]["base_file"];
    const std::string query_file = DATASETS[dataset_name]["query_file"];
    const FileFormat  file_format = getFileFormat(base_file);

    auto [nvectors, dim] = get_dataset_info(base_file);

    // ── Index directory convention (matches static_routing_qps_single.cpp) ────
    const std::string meta_dir = dataset_name + "_" + std::to_string(num_partitions);

    std::string log_id = "shared_static_" + dataset_name + "_" + std::to_string(num_partitions);
    Log logger(log_id);
    Communicator comm;

    // ── Shared routing state (all ranks) ─────────────────────────────────────
    std::vector<int>                  routing_partitions;
    std::vector<int>                  routing_counts;     // |rho(c)| per centroid
    hnswlib::L2Space                  meta_space(dim);
    hnswlib::HierarchicalNSW<float>*  routing_hnsw = nullptr;

    // ── Load index ────────────────────────────────────────────────────────────
    Coordinator* meta_index = nullptr;
    Executor*    sub_index  = nullptr;

    // All ranks already know nvectors and dim from get_dataset_info above.
    // No coordinator→executor broadcast is needed before loading — each rank
    // loads its own shard directly from disk and passes dim to its constructor.

    if (rank == 0) {
        // Coordinator: load meta-index, then broadcast routing state.
        std::cout << "[Static] Loading coordinator index from " << meta_dir << "\n";
        meta_index = new Coordinator(dim, &comm, &logger);
        meta_index->load(meta_dir, EF_SEARCH);

        routing_hnsw = meta_index->getMetaHNSW();
        bcastRoutingState(rank, dim,
                          meta_index->getMetaHNSW(),
                          &meta_index->getPartitions(),
                          &meta_index->getCenterCounts(),
                          routing_hnsw, routing_partitions, routing_counts,
                          &meta_space);

        std::cout << "[Static] Coordinator routing state broadcast complete\n";
    } else {
        // Executor: load shard, then receive routing state broadcast.
        const std::string filename_prefix =
            meta_dir + "/executor_" + std::to_string(rank)
            + "_" + dataset_name + "_" + std::to_string(num_partitions);

        std::cout << "[Static] Executor " << rank
                  << " loading shard from " << filename_prefix << "\n";
        sub_index = new Executor(rank, dim, comm, &logger);
        sub_index->load(filename_prefix, EF_SEARCH);

        std::cout << "[Static] Executor " << rank
                  << " loaded " << sub_index->getElementCount()
                  << " vectors in local graph\n";

        bcastRoutingState(rank, dim, nullptr, nullptr, nullptr,
                          routing_hnsw, routing_partitions, routing_counts,
                          &meta_space);

        std::cout << "[Static] Executor " << rank << " ready\n";
    }

    // ── Load queries (all ranks) ───────────────────────────────────────────────
    std::vector<float> queries = readVecs(query_file, dim);
    const size_t nq    = queries.size() / dim;
    const size_t chunk = (nq + world_size - 1) / world_size;
    const size_t my_qs = static_cast<size_t>(rank) * chunk;
    const size_t my_qe = std::min(my_qs + chunk, nq);

    // ── Ground truth (all ranks load for recall computation) ──────────────────
    std::vector<std::vector<int>> gt;
    bool have_gt = false;
    {
        auto gt_it = DATASETS[dataset_name].find("gt_file");
        if (gt_it != DATASETS[dataset_name].end() &&
                !gt_it->second.empty() && gt_it->second != "TODO") {
            gt      = readGT(gt_it->second, file_format);
            have_gt = !gt.empty();
        }
    }

    if (rank == 0)
        std::cout << "[Static] Loaded " << nq << " queries"
                  << (have_gt ? " with ground truth" : " (no ground truth)")
                  << "\n";

    // ── Open CSV (rank 0 only) ─────────────────────────────────────────────────
    std::ofstream csv;
    if (rank == 0) {
        csv.open(output_file, std::ios::out);
        if (!csv.is_open()) {
            std::cerr << "ERROR: cannot open output file: " << output_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        csv << "mode,param,recall@" << k << ",qps,avg_parts_searched\n";
    }

    // ── Open per-rank diagnostics CSV (rank 0 only) ───────────────────────────
    // One row per combo; per-executor cells are "load:route_t:search_t".
    std::ofstream diag;
    if (rank == 0) {
        diag.open(output_file + std::string(".diag.csv"), std::ios::out);
        if (diag.is_open()) {
            diag << "mode,param,recall,qps";
            for (int e = 1; e < world_size; e++) diag << ",exec" << e;
            diag << "\n";
        }
    }

    // Hot-node tracking (rank 0): per-mode cumulative per-executor query load,
    // and how often each executor is THE hottest, summed over that mode's combos.
    std::unordered_map<std::string, std::vector<long long>> cum_load;   // mode → load[world_size]
    std::unordered_map<std::string, std::vector<int>>       hot_count;  // mode → times-hottest[world_size]

    // ══════════════════════════════════════════════════════════════════════════
    //  Sweep: for each (mode, param) combo:
    //    1. Warmup pass   – measures recall, not timed
    //    2. NUM_RUNS timed passes – all queries, barrier-to-barrier,
    //                               MPI_Reduce(MAX) for true end-to-end time
    // ══════════════════════════════════════════════════════════════════════════
    for (int ci = 0; ci < n_combos; ci++) {
        const auto& [mode, param]      = sweep_combos[ci];
        const std::string mode_str     = mode_to_string(mode);

        if (rank == 0)
            std::cout << "[Static] combo " << (ci+1) << "/" << n_combos
                      << "  mode=" << mode_str << "  param=" << param << "\n";

        // ── Warmup pass (measures recall, not timed) ──────────────────────────
        long long warmup_parts = 0;
        uint64_t  warmup_hits  = 0;
        double    dbg_rt = 0.0, dbg_st = 0.0;
        long long dbg_load = 0;
        runSearchPass(rank, world_size, num_partitions, k,
                      nq, my_qs, my_qe,
                      queries, dim,
                      routing_hnsw, routing_partitions, routing_counts,
                      mode, param,
                      sub_index,
                      have_gt ? &gt : nullptr,
                      warmup_parts, warmup_hits,
                      dbg_rt, dbg_st, dbg_load);

        uint64_t total_hits = 0;
        MPI_Reduce(&warmup_hits, &total_hits, 1, MPI_UINT64_T,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        long long global_warmup_parts = 0;
        MPI_Reduce(&warmup_parts, &global_warmup_parts, 1, MPI_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);

        const double recall = (have_gt && nq > 0)
            ? static_cast<double>(total_hits) / (static_cast<double>(nq) * k)
            : -1.0;
        const double avg_parts_warmup =
            (nq > 0) ? static_cast<double>(global_warmup_parts) / static_cast<double>(nq)
                     : -1.0;

        if (rank == 0)
            std::cout << "  [warmup] recall@" << k << "=" << recall
                      << "  avg_parts=" << avg_parts_warmup << "\n";

        // ── Timed passes ──────────────────────────────────────────────────────
        // Each run is timed independently.  QPS is computed from the minimum
        // elapsed time (max-across-ranks per run, then min across runs).
        long long timed_parts_acc = 0;
        double    min_elapsed     = std::numeric_limits<double>::max();

        // Per-rank diagnostics for the run that yields the QPS (min elapsed):
        //   best_route[r] = Phase-1 routing time on rank r
        //   best_search[r]= Phase-3 shard-search time on rank r
        //   best_load[r]  = # queries rank r searched (column sum)
        std::vector<double>    best_route (rank == 0 ? world_size : 0, 0.0);
        std::vector<double>    best_search(rank == 0 ? world_size : 0, 0.0);
        std::vector<long long> best_load  (rank == 0 ? world_size : 0, 0);

        for (int r = 0; r < NUM_RUNS; r++) {
            long long pass_parts = 0;
            uint64_t  pass_hits  = 0;
            double    rt = 0.0, st = 0.0;
            long long load = 0;

            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();

            runSearchPass(rank, world_size, num_partitions, k,
                          nq, my_qs, my_qe,
                          queries, dim,
                          routing_hnsw, routing_partitions, routing_counts,
                          mode, param,
                          sub_index,
                          /*gt=*/nullptr,   // skip recall in timed passes
                          pass_parts, pass_hits,
                          rt, st, load);

            const double t1 = MPI_Wtime();

            double run_elapsed = t1 - t0;
            double run_max     = 0.0;
            MPI_Reduce(&run_elapsed, &run_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            // Gather this run's per-rank breakdown to rank 0.
            std::vector<double>    g_route (rank == 0 ? world_size : 0);
            std::vector<double>    g_search(rank == 0 ? world_size : 0);
            std::vector<long long> g_load  (rank == 0 ? world_size : 0);
            MPI_Gather(&rt,   1, MPI_DOUBLE,    g_route.data(),  1, MPI_DOUBLE,    0, MPI_COMM_WORLD);
            MPI_Gather(&st,   1, MPI_DOUBLE,    g_search.data(), 1, MPI_DOUBLE,    0, MPI_COMM_WORLD);
            MPI_Gather(&load, 1, MPI_LONG_LONG, g_load.data(),   1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

            if (rank == 0 && run_max < min_elapsed) {
                min_elapsed = run_max;
                best_route  = g_route;
                best_search = g_search;
                best_load   = g_load;
            }

            timed_parts_acc += pass_parts;
        }

        long long global_timed_parts = 0;
        MPI_Reduce(&timed_parts_acc, &global_timed_parts, 1, MPI_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const double qps = (min_elapsed > 0.0)
                ? static_cast<double>(nq) / min_elapsed
                : 0.0;
            const double avg_parts_timed =
                (nq > 0 && NUM_RUNS > 0)
                ? static_cast<double>(global_timed_parts)
                    / (static_cast<double>(nq) * NUM_RUNS)
                : -1.0;

            std::cout << "  [timed]  qps=" << qps
                      << "  avg_parts=" << avg_parts_timed << "\n";

            csv << mode_str << "," << param
                << "," << recall
                << "," << qps
                << "," << avg_parts_timed
                << "\n";
            csv.flush();

            // ── Diagnostics over executor ranks (rank >= 1) from the QPS run ──
            // load skew  → max/mean of per-rank queries-searched (column sums)
            // route_time → Phase-1 per-rank routing compute (uniform if no skew)
            // search_time→ Phase-3 per-rank shard search (tracks load skew)
            long long load_sum = 0, load_max = 0;
            double    rt_sum = 0.0,  rt_max = 0.0;
            double    st_sum = 0.0,  st_max = 0.0;
            int       nexec  = 0;
            for (int e = 1; e < world_size; e++) {     // skip coordinator (rank 0)
                nexec++;
                load_sum += best_load[e];
                load_max  = std::max(load_max, best_load[e]);
                rt_sum   += best_route[e];
                rt_max    = std::max(rt_max, best_route[e]);
                st_sum   += best_search[e];
                st_max    = std::max(st_max, best_search[e]);
            }
            const double load_mean = nexec ? double(load_sum) / nexec : 0.0;
            const double rt_mean   = nexec ? rt_sum / nexec : 0.0;
            const double st_mean   = nexec ? st_sum / nexec : 0.0;
            const double load_skew = load_mean > 0 ? load_max / load_mean : 0.0;
            const double st_skew   = st_mean   > 0 ? st_max   / st_mean   : 0.0;

            // ── Hot node: which executor carries the peak load this combo ──
            int hot_exec = -1;
            long long hot_val = -1;
            for (int e = 1; e < world_size; e++) {
                if (best_load[e] > hot_val) { hot_val = best_load[e]; hot_exec = e; }
            }
            // Accumulate per-mode totals to test whether the SAME node is hot
            // across modes (popularity skew) vs mode-specific.
            auto& cl = cum_load[mode_str];
            auto& hc = hot_count[mode_str];
            if (cl.empty()) cl.assign(world_size, 0);
            if (hc.empty()) hc.assign(world_size, 0);
            for (int e = 1; e < world_size; e++) cl[e] += best_load[e];
            if (hot_exec >= 0) hc[hot_exec]++;

            std::cout << "  [diag]   load_skew(max/mean)=" << load_skew
                      << "  search_skew=" << st_skew
                      << "  route_t(mean/max)=" << rt_mean << "/" << rt_max
                      << "  search_t(mean/max)=" << st_mean << "/" << st_max << "\n";
            std::cout << "  [hot]    exec" << hot_exec
                      << " load=" << hot_val
                      << " (" << (load_mean > 0 ? hot_val / load_mean : 0.0) << "x mean)"
                      << "  per-exec(load/mean):";
            for (int e = 1; e < world_size; e++)
                std::cout << " e" << e << "=" << (load_mean > 0 ? best_load[e] / load_mean : 0.0);
            std::cout << "\n";

            // Per-executor breakdown to a companion CSV (output_file + ".diag.csv").
            diag << mode_str << "," << param << "," << recall << "," << qps;
            for (int e = 1; e < world_size; e++) {
                diag << "," << best_load[e]
                     << ":" << best_route[e]
                     << ":" << best_search[e];
            }
            diag << "\n";
            diag.flush();
        }
    }

    // ── Hot-node summary across the whole sweep (rank 0) ───────────────────────
    // For each mode, print per-executor cumulative load normalised by the mean,
    // and which executor was hottest most often.  If the same executor tops every
    // mode, the skew is a property of the partitioning (popularity), and the
    // recall-target router merely amplifies it.
    if (rank == 0) {
        std::cout << "\n[Static] ===== HOT-NODE SUMMARY (per-executor load / mean) =====\n";
        for (auto& [mode_str, cl] : cum_load) {
            long long tot = 0;
            for (int e = 1; e < world_size; e++) tot += cl[e];
            const double mean = (world_size > 1) ? double(tot) / (world_size - 1) : 0.0;
            int hot_e = -1; double hot_mult = 0.0;
            std::cout << "  " << mode_str << ":";
            for (int e = 1; e < world_size; e++) {
                const double mult = mean > 0 ? cl[e] / mean : 0.0;
                std::cout << " e" << e << "=" << mult;
                if (mult > hot_mult) { hot_mult = mult; hot_e = e; }
            }
            std::cout << "  | hottest=exec" << hot_e << " (" << hot_mult << "x)";
            const auto& hc = hot_count[mode_str];
            int champ = -1, champ_n = -1, mode_combos = 0;
            for (int e = 1; e < world_size; e++) {
                if (e < static_cast<int>(hc.size())) {
                    mode_combos += hc[e];
                    if (hc[e] > champ_n) { champ_n = hc[e]; champ = e; }
                }
            }
            std::cout << "  topped " << champ_n << "/" << mode_combos
                      << " combos by exec" << champ << "\n";
        }
        std::cout << "[Static] =========================================================\n";
    }

    if (rank == 0) {
        csv.close();
        if (diag.is_open()) diag.close();
        std::cout << "[Static] Done. Results written to " << output_file << "\n";
    }

    delete meta_index;
    delete sub_index;
    if (rank > 0 && routing_hnsw != nullptr) delete routing_hnsw;

    MPI_Finalize();
    return 0;
}
