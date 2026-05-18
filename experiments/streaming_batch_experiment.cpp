// streaming_batch_experiment.cpp
//
// Runs the NeurIPS streaming benchmark runbook against a distributed SURGE
// index.  Each runbook step is either:
//   insert  [start, end)  – vectors with those global indices are inserted
//   delete  [start, end)  – vectors with those global indices are deleted
//   search               – run the query set, measure recall and QPS
//
// The first step is always an insert: the index is built fresh from those
// vectors.  All subsequent insert/delete steps use the live insert/delete
// API.  After every update step the coordinator checks whether maintenance
// (full or partial rebuild) is needed, then collects per-shard element counts.
//
// ─── Assumptions ────────────────────────────────────────────────────────────
//  1. The first runbook step is always operation='insert' with start=0.
//  2. Vector labels equal their global index in the base file.
//  3. Ground truth is the static file at DATASETS[dataset]["gt_file"].
//     This gives approximate recall against the full dataset rather than the
//     current active set; per-step GT would be needed for exact recall.
//  4. Build hyper-parameters are hardcoded (NCENTERS, EF_CONSTRUCTION, …),
//     matching the conventions of the other experiments in this repo.
//  5. Delete steps use handle_delete() one label at a time.  handle_deletes()
//     (batch) has an incomplete vector-feedback path in index.cpp and is not
//     used here.
//  6. Shard sizes are collected via a lightweight SIZE_REQUEST/response round-
//     trip after every step.  The reported count is the number of live (non-
//     soft-deleted) elements in each executor's sub-HNSW.
//
// ─── Usage ──────────────────────────────────────────────────────────────────
//  mpirun -np <num_partitions+1> ./streaming_batch_experiment \
//      <dataset> <num_partitions> <full_threshold> <partial_threshold> \
//      <query_mode> <query_param> <k> <output_file> \
//      <num_coord_threads> <window_size>
//
//  query_mode  : BranchingFactor | NProbe | RecallTarget
//  query_param : branching factor (int), nprobe (int), or recall target (float)
//  full_threshold / partial_threshold : minimum vectors that must migrate to
//      trigger a full / partial rebuild.  Set both to 0 to disable maintenance.
//
// ─── Output CSV columns ─────────────────────────────────────────────────────
//  step, operation, range_start, range_end,
//  time_s, throughput,
//  recall@<k>,
//  shard_0_size, shard_1_size, …, shard_<P-1>_size
//
//  time_s     = wall time for the primary operation (inserts, deletes, or
//               queries).  Maintenance / rebuild time is NOT included so that
//               the raw operation rate is visible.
//  throughput = n_ops / time_s   (inserts/s, deletes/s, or queries/s)
//  recall     = mean recall@k over all queries (-1 if no ground truth)
//  shard sizes are the live (non-soft-deleted) element counts collected after
//  both the operation and any maintenance rebuild have completed.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include "index.h"

// ─── Build hyper-parameters (hardcoded, matching other experiments) ───────────
static constexpr int    NCENTERS             = 10000;
static constexpr int    EF_CONSTRUCTION      = 200;
static constexpr int    M_META               = 16;
static constexpr int    M_SUB                = 16;
static constexpr int    NUM_BUILDING_THREADS = 32;
static constexpr int    EF_SEARCH            = 200;
static constexpr size_t SAMPLE_SIZE          = 100000;

// ─── Runbook step ─────────────────────────────────────────────────────────────
struct RunbookStep {
    int         step_num  = -1;
    std::string operation;        // "insert" | "delete" | "search"
    int         start     = -1;
    int         end       = -1;   // exclusive upper bound
};

// Parse the NeurIPS YAML runbook via yaml-cpp.
// Returns steps sorted by their integer key; non-numeric keys (e.g. max_pts)
// are silently skipped.
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
            std::all_of(key.begin(), key.end(), [](unsigned char c){ return std::isdigit(c); });
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

static RoutingMode parse_routing_mode(const std::string& s) {
    if (s == "BranchingFactor") return RoutingMode::BranchingFactor;
    if (s == "NProbe")          return RoutingMode::NProbe;
    if (s == "RecallTarget")    return RoutingMode::RecallTarget;
    throw std::invalid_argument(
        "Unknown query_mode '" + s + "'. Use BranchingFactor, NProbe, or RecallTarget.");
}

// Windowed semaphore for coordinator OMP task dispatch during search steps.
struct CountingSemaphore {
    explicit CountingSemaphore(int n) : count_(n) {}
    void acquire() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{ return count_ > 0; });
        --count_;
    }
    void release() {
        std::lock_guard<std::mutex> lk(mu_);
        ++count_;
        cv_.notify_one();
    }
private:
    std::mutex              mu_;
    std::condition_variable cv_;
    int                     count_;
};

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {

    if (argc != 11) {
        std::cerr
            << "Usage: " << argv[0]
            << " <dataset> <num_partitions>"
            << " <full_threshold> <partial_threshold>"
            << " <query_mode> <query_param> <k>"
            << " <output_file>"
            << " <num_coord_threads> <window_size>\n"
            << "\n"
            << "  query_mode  : BranchingFactor | NProbe | RecallTarget\n"
            << "  query_param : branching factor (int), nprobe (int), or"
               " recall target (float in [0,1])\n"
            << "  full_threshold / partial_threshold : 0 disables that level\n";
        return 1;
    }

    const std::string dataset_name     = argv[1];
    const int         num_partitions   = std::stoi(argv[2]);
    const int         full_threshold   = std::stoi(argv[3]);
    const int         partial_threshold = std::stoi(argv[4]);
    const std::string query_mode_str   = argv[5];
    const float       query_param      = std::stof(argv[6]);
    const int         k                = std::stoi(argv[7]);
    const std::string output_file      = argv[8];
    const int         num_coord_threads = std::stoi(argv[9]);
    const int         window_size      = std::stoi(argv[10]);

    const RoutingMode query_mode = parse_routing_mode(query_mode_str);

    // ── MPI init ──────────────────────────────────────────────────────────────
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "ERROR: MPI does not support MPI_THREAD_MULTIPLE\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int node, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size == 1) {
        std::cerr << "Not Implemented: Single Node SURGE\n";
        MPI_Finalize();
        return 1;
    }
    if (world_size != num_partitions + 1) {
        std::cerr << "ERROR: world_size=" << world_size
                  << " must equal num_partitions+1=" << (num_partitions + 1) << "\n";
        MPI_Finalize();
        return 1;
    }

    // ── Logger & communicator ─────────────────────────────────────────────────
    std::string log_id = "streaming_" + dataset_name + "_" + std::to_string(num_partitions);
    Log         logger(log_id);
    Communicator comm;

    // ── Dataset dimensions (all ranks need these) ─────────────────────────────
    const std::string base_file = DATASETS[dataset_name]["base_file"];
    std::pair<int,int> data_info = get_dataset_info(base_file);
    const int nvectors = data_info.first;
    const int dim      = data_info.second;

    // ═══════════════════════════════════════════════════════════════════════════
    //                          COORDINATOR  (rank 0)
    // ═══════════════════════════════════════════════════════════════════════════
    if (node == 0) {

        comm.broadcast_dataset_info(nvectors, dim, world_size);

        // ── Parse runbook ─────────────────────────────────────────────────────
        const std::string runbook_path = DATASETS[dataset_name]["runbook"];
        std::vector<RunbookStep> steps = parse_runbook(runbook_path, dataset_name);
        if (steps.empty()) {
            std::cerr << "ERROR: runbook is empty for dataset '" << dataset_name << "'\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (steps[0].operation != "insert") {
            std::cerr << "ERROR: first runbook step must be 'insert' (got '"
                      << steps[0].operation << "')\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (steps[0].start != 0) {
            std::cerr << "ERROR: first insert step must have start=0 (got "
                      << steps[0].start << "); distribute_vectors reads from offset 0\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "[Coordinator] Runbook: " << steps.size() << " steps\n";

        // ── Helper: collect per-shard live element counts ─────────────────────
        // Sends SIZE_REQUEST to every executor, collects the SIZE_REQUEST echo
        // (executor packs the count in header.size and sends back).
        // Called after every step, once all operation and rebuild work is done.
        auto collect_shard_sizes = [&]() -> std::vector<size_t> {
            MessageHeader req;
            req.type = SIZE_REQUEST;
            req.size = 0;
            req.tag  = 0;
            for (int i = 1; i < world_size; ++i)
                MPI_Send(&req, sizeof(MessageHeader), MPI_BYTE, i, 0, MPI_COMM_WORLD);

            std::vector<size_t> sizes(num_partitions);
            for (int i = 1; i < world_size; ++i) {
                MessageHeader resp;
                MPI_Recv(&resp, sizeof(MessageHeader), MPI_BYTE, i, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sizes[i - 1] = resp.size;
            }
            return sizes;
        };

        // ── Phase 1: initial build ────────────────────────────────────────────
        const RunbookStep& init_step = steps[0];
        const int init_n = init_step.end;   // start==0, so end == count

        const size_t sample_size = std::min((size_t)init_n, SAMPLE_SIZE);
        std::vector<float> sample = getSample(base_file, init_n, dim, sample_size);

        Coordinator metaIndex(dim, &comm, &logger);
        metaIndex.setSampleData(sample.data(), sample_size);
        metaIndex.build(NCENTERS, num_partitions, EF_CONSTRUCTION, M_META);

        std::cout << "[Coordinator] Distributing initial " << init_n << " vectors\n";
        metaIndex.distribute_vectors(base_file, init_n, /*log_partitions=*/false,
                                     NUM_BUILDING_THREADS);

        comm.broadcast_termination(world_size);   // executors → build sub-HNSWs
        MPI_Barrier(MPI_COMM_WORLD);              // wait until all builds finish
        std::cout << "[Coordinator] Initial build complete\n";

        comm.broadcast_ef_search(EF_SEARCH, world_size);

        // ── Load queries and (static) ground truth ────────────────────────────
        const std::string query_file = DATASETS[dataset_name]["query_file"];
        FileFormat format = getFileFormat(base_file);

        std::vector<float> queries = readVecs(query_file, dim);
        const size_t num_queries   = queries.size() / dim;
        std::cout << "[Coordinator] Loaded " << num_queries << " queries\n";

        std::vector<std::vector<int>> ground_truth;
        bool have_gt = false;
        auto gt_it = DATASETS[dataset_name].find("gt_file");
        if (gt_it != DATASETS[dataset_name].end() &&
                !gt_it->second.empty() && gt_it->second != "TODO") {
            ground_truth = readGT(gt_it->second, format);
            have_gt = !ground_truth.empty();
        }
        if (have_gt)
            std::cout << "[Coordinator] Loaded ground truth\n";
        else
            std::cout << "[Coordinator] No ground truth available; recall = -1\n";

        // ── Open output CSV ───────────────────────────────────────────────────
        std::ofstream outfile(output_file, std::ios::out);
        if (!outfile.is_open()) {
            std::cerr << "ERROR: cannot open output file: " << output_file << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Dynamic header: one shard_N_size column per executor.
        outfile << "step,operation,range_start,range_end,time_s,throughput,recall@" << k;
        for (int i = 0; i < num_partitions; ++i)
            outfile << ",shard_" << i << "_size";
        outfile << "\n";
        outfile.flush();

        // Helper: write one CSV row.
        auto write_row = [&](int step_num,
                             const std::string& op,
                             int range_start, int range_end,
                             double time_s, double throughput,
                             double recall,
                             const std::vector<size_t>& shard_sizes)
        {
            outfile << step_num << "," << op
                    << "," << range_start << "," << range_end
                    << "," << time_s << "," << throughput
                    << "," << recall;
            for (size_t sz : shard_sizes) outfile << "," << sz;
            outfile << "\n";
            outfile.flush();
        };

        // Log initial build step (collect sizes after build barrier).
        {
            auto sizes = collect_shard_sizes();
            // time/throughput for the build itself are not instrumented here
            // (they are not part of the streaming workload).
            write_row(init_step.step_num, "insert",
                      init_step.start, init_step.end,
                      0.0, 0.0, -1.0, sizes);
        }

        // Semaphore limiting concurrent in-flight queries.
        CountingSemaphore sem(window_size);

        // ── Phase 2: streaming steps ──────────────────────────────────────────
        for (size_t s = 1; s < steps.size(); ++s) {
            const RunbookStep& step = steps[s];
            std::cout << "[Coordinator] Step " << step.step_num
                      << " op=" << step.operation << "\n";

            // ── INSERT ────────────────────────────────────────────────────────
            if (step.operation == "insert") {
                const int n_insert = step.end - step.start;
                std::cout << "[Coordinator]   inserting " << n_insert
                          << " vectors [" << step.start << ", " << step.end << ")\n";

                std::vector<float> insert_vecs =
                    readVecs(base_file, dim, n_insert, step.start);
                std::vector<int> labels(n_insert);
                std::iota(labels.begin(), labels.end(), step.start);

                // Time the insert operation only.
                const double t0 = MPI_Wtime();
                metaIndex.handle_inserts(insert_vecs, labels);
                const double t1 = MPI_Wtime();

                // Maintenance (rebuild if needed) — not counted in time_s.
                const int rb = metaIndex.reBuild(world_size, EF_CONSTRUCTION, M_META,
                                                  full_threshold, partial_threshold);
                if (rb > 0)
                    std::cout << "[Coordinator]   rebuild performed (type=" << rb << ")\n";

                const double time_s    = t1 - t0;
                const double throughput = (time_s > 0.0)
                                          ? static_cast<double>(n_insert) / time_s
                                          : 0.0;

                auto sizes = collect_shard_sizes();
                write_row(step.step_num, "insert",
                          step.start, step.end,
                          time_s, throughput, -1.0, sizes);

            // ── DELETE ────────────────────────────────────────────────────────
            } else if (step.operation == "delete") {
                const int n_delete = step.end - step.start;
                std::cout << "[Coordinator]   deleting " << n_delete
                          << " vectors [" << step.start << ", " << step.end << ")\n";

                // Single-delete loop (handle_deletes has an incomplete
                // vector-feedback path and is not used here).
                const double t0 = MPI_Wtime();
                for (int label = step.start; label < step.end; ++label)
                    metaIndex.handle_delete(label, world_size);
                const double t1 = MPI_Wtime();

                const int rb = metaIndex.reBuild(world_size, EF_CONSTRUCTION, M_META,
                                                  full_threshold, partial_threshold);
                if (rb > 0)
                    std::cout << "[Coordinator]   rebuild performed (type=" << rb << ")\n";

                const double time_s     = t1 - t0;
                const double throughput = (time_s > 0.0)
                                          ? static_cast<double>(n_delete) / time_s
                                          : 0.0;

                auto sizes = collect_shard_sizes();
                write_row(step.step_num, "delete",
                          step.start, step.end,
                          time_s, throughput, -1.0, sizes);

            // ── SEARCH ────────────────────────────────────────────────────────
            } else if (step.operation == "search") {
                std::cout << "[Coordinator]   running search over "
                          << num_queries << " queries\n";

                std::vector<std::atomic<double>> per_query_recall(num_queries);
                for (auto& v : per_query_recall) v.store(-1.0);

                const double t0 = MPI_Wtime();

                #pragma omp parallel num_threads(num_coord_threads)
                #pragma omp single
                {
                    for (int i = 0; i < (int)num_queries; ++i) {
                        sem.acquire();
                        float* qvec = queries.data() + (size_t)i * dim;

                        #pragma omp task firstprivate(i, qvec)
                        {
                            std::vector<int> results = metaIndex.handle_query(
                                qvec, i, k, query_mode, query_param);
                            sem.release();

                            if (have_gt && i < (int)ground_truth.size()) {
                                const auto& gt = ground_truth[i];
                                int hits = 0;
                                for (int j = 0; j < k && j < (int)gt.size(); ++j)
                                    for (int res_id : results)
                                        if (gt[j] == res_id) ++hits;
                                per_query_recall[i].store(
                                    static_cast<double>(hits) / static_cast<double>(k));
                            }
                        }
                    }
                    #pragma omp taskwait
                }

                const double t1  = MPI_Wtime();
                const double time_s     = t1 - t0;
                const double throughput = (time_s > 0.0)
                                          ? static_cast<double>(num_queries) / time_s
                                          : 0.0;

                double recall = -1.0;
                if (have_gt) {
                    double sum = 0.0;
                    for (auto& v : per_query_recall) sum += v.load();
                    recall = sum / static_cast<double>(num_queries);
                }

                std::cout << "[Coordinator] Step " << step.step_num
                          << "  recall@" << k << "=" << recall
                          << "  qps=" << throughput << "\n";

                // Search steps span the full query set; use 0..num_queries as range.
                auto sizes = collect_shard_sizes();
                write_row(step.step_num, "search",
                          0, static_cast<int>(num_queries),
                          time_s, throughput, recall, sizes);

            } else {
                std::cerr << "[Coordinator] Unknown operation '" << step.operation
                          << "' in step " << step.step_num << "; skipping\n";
            }
        }

        // Signal executors to exit their streaming loops.
        comm.broadcast_termination(world_size, 1);
        std::cout << "[Coordinator] Done. Results written to " << output_file << "\n";

    // ═══════════════════════════════════════════════════════════════════════════
    //                          EXECUTOR  (rank > 0)
    // ═══════════════════════════════════════════════════════════════════════════
    } else {

        int nvectors_recv, dim_recv;
        comm.recv_dataset_info(nvectors_recv, dim_recv);
        std::cout << "[Executor " << node << "] dataset info: n=" << nvectors_recv
                  << " d=" << dim_recv << "\n";

        Executor subIndex(node, dim, comm, &logger);

        // ── Phase 1: receive initial vectors then build sub-HNSW ─────────────
        {
            bool done = false;
            while (!done) {
                MessageHeader hdr;
                comm.recv_header(hdr, 0);
                if (hdr.type == END_OF_COMMUNICATION) {
                    done = true;
                } else {
                    assert(hdr.type == VECTOR_SEND);
                    subIndex.receiveData(hdr.size);
                }
            }
        }
        subIndex.build(EF_CONSTRUCTION, M_SUB, NUM_BUILDING_THREADS);
        std::cout << "[Executor " << node << "] Initial build complete\n";

        MPI_Barrier(MPI_COMM_WORLD);   // sync with coordinator

        // ── Phase 2: streaming event loop ─────────────────────────────────────
        //
        // One receiver thread pulls MessageHeader frames from the coordinator
        // (all on MPI tag 0).  Worker threads (OMP) dequeue and process them.
        //
        // Message types and their handlers:
        //   SET_EF_SEARCH            – handled inline in receiver thread
        //   SIZE_REQUEST             – handled inline in receiver thread;
        //                             replies with live element count in header.size
        //   QUERY_SEND               – search(header.size /*=k*/, header.tag)
        //   QUERY_BATCH_SEND         – batch_search(header.size, k, header.tag)
        //   INSERT_BATCH_SEND        – insert_batch(header.size, header.tag)
        //   DELETE_SEND              – delete_vector(header.size /*=label*/, header.tag)
        //   FULL_REBUILD_REQUEST     – reBuild(header.size, …)
        //   PARTIAL_REBUILD_REQUEST  – partialReBuild(header.size, …)
        //   END_OF_COMMUNICATION     – exit

        std::queue<MessageHeader> work_queue;
        std::mutex                queue_mu;
        std::condition_variable   queue_cv;
        std::atomic<bool>         end_flag{false};
        std::atomic<int>          num_processed{0};

        const int num_worker_threads = 32;

        std::thread receiver([&]() {
            while (true) {
                MessageHeader hdr;
                comm.recv_header(hdr, 0);

                if (hdr.type == END_OF_COMMUNICATION) {
                    end_flag = true;
                    queue_cv.notify_all();
                    break;
                }

                // Lightweight messages handled directly in the receiver thread
                // to avoid queuing delays and to guarantee in-order response.

                if (hdr.type == SET_EF_SEARCH) {
                    subIndex.setEfSearch(hdr.size);
                    continue;
                }

                if (hdr.type == SIZE_REQUEST) {
                    // Reply with the live (non-deleted) element count.
                    // getElementCount() takes a shared lock — safe to call here
                    // because SIZE_REQUEST only arrives after the previous step's
                    // work (and any rebuild) has fully completed.
                    MessageHeader resp;
                    resp.type = SIZE_REQUEST;   // echo type so coordinator can assert
                    resp.size = subIndex.getElementCount();
                    resp.tag  = 0;
                    MPI_Send(&resp, sizeof(MessageHeader), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                    continue;
                }

                {
                    std::lock_guard<std::mutex> lk(queue_mu);
                    work_queue.push(hdr);
                }
                queue_cv.notify_one();
            }
        });

        #pragma omp parallel num_threads(num_worker_threads - 1)
        {
            while (true) {
                MessageHeader hdr;
                {
                    std::unique_lock<std::mutex> lk(queue_mu);
                    queue_cv.wait(lk, [&]{
                        return !work_queue.empty() || end_flag.load();
                    });
                    if (end_flag && work_queue.empty()) break;
                    hdr = work_queue.front();
                    work_queue.pop();
                }

                if (hdr.type == QUERY_SEND) {
                    // header.size == num_neighbors (k) as packed by handle_query
                    subIndex.search(hdr.size, hdr.tag);
                    ++num_processed;

                } else if (hdr.type == QUERY_BATCH_SEND) {
                    // header.size == number of queries routed to this shard
                    subIndex.batch_search(hdr.size, static_cast<size_t>(k), hdr.tag);
                    num_processed += static_cast<int>(hdr.size);

                } else if (hdr.type == INSERT_BATCH_SEND) {
                    // header.size == number of vectors in this shard's batch
                    subIndex.insert_batch(hdr.size, hdr.tag);

                } else if (hdr.type == DELETE_SEND) {
                    // header.size == label (packed by Communicator::send_delete)
                    subIndex.delete_vector(static_cast<size_t>(hdr.size), hdr.tag);

                } else if (hdr.type == FULL_REBUILD_REQUEST) {
                    subIndex.reBuild(static_cast<int>(hdr.size),
                                     NCENTERS, world_size,
                                     EF_CONSTRUCTION, M_SUB, NUM_BUILDING_THREADS);

                } else if (hdr.type == PARTIAL_REBUILD_REQUEST) {
                    subIndex.partialReBuild(static_cast<int>(hdr.size),
                                            NCENTERS, world_size,
                                            EF_CONSTRUCTION, NUM_BUILDING_THREADS);

                } else {
                    std::cerr << "[Executor " << node
                              << "] Unexpected message type " << hdr.type << "; ignoring\n";
                }
            }
        }

        receiver.join();
        std::cout << "[Executor " << node << "] Done. Processed "
                  << num_processed.load() << " queries\n";
    }

    MPI_Finalize();
    return 0;
}
