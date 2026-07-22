// runbook_partitions_parallel.cpp
//
// KaHIP partitioner for the theoretical-recall simulation. For each
// step_NNNNNN_base_layer.csv produced by runbook_centers, partitions the base
// layer into <num_partitions> blocks and writes step_NNNNNN_partitions.csv,
// matching each step to the previous one to minimize relabeling (drift phi).
//
// Usage:  ./bin/runbook_partitions_parallel <input_dir> [output_dir] <num_partitions>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "utils.h"

extern "C" {
#include "kaHIP_interface.h"
}

namespace fs = std::filesystem;

// CSV helpers

static std::vector<int> load_counts(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path.string());
    std::vector<int> counts;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) counts.push_back(std::stoi(line));
    return counts;
}

static fs::path counts_path(const fs::path& in_dir, int step) {
    std::ostringstream oss;
    oss << "step_" << std::setw(6) << std::setfill('0') << step << "_center_counts.csv";
    return in_dir / oss.str();
}

// Step number extraction

static int step_num_base(const fs::path& p) {
    // step_000002_base_layer.csv -> 2
    auto name = p.stem().string(); // "step_000002_base_layer"
    return std::stoi(name.substr(6, name.find('_', 6) - 6)); // skip "step_", read up to first '_'
}

// Build CSR graph and partition with KaFFPa

static void process_csv(
    const fs::path& edges_csv,
    const fs::path& counts_csv,
    bool   use_edge_weights,
    bool   use_node_weights,
    int    nblocks,
    double imbalance,
    int    mode,
    int    seed,
    int*   out_part,
    int    num_nodes)
{
    // Read edges and compute normalized weights
    std::ifstream ef(edges_csv);
    if (!ef) throw std::runtime_error("Cannot open " + edges_csv.string());

    std::vector<int>    src, dst, wgt;
    std::vector<double> inv_dist;
    std::vector<int>    degree(num_nodes, 0);
    double w_max = 0.0;

    {
        std::string line;
        while (std::getline(ef, line)) {
            if (line.empty()) continue;
            const char* p = line.c_str();
            char* end;
            int    s = static_cast<int>(std::strtol(p,       &end, 10)); p = end + 1;
            int    t = static_cast<int>(std::strtol(p,       &end, 10)); p = end + 1;
            double d = std::strtod(p, nullptr);
            double w = 1.0 / d;
            if (w > w_max) w_max = w;
            src.push_back(s);
            dst.push_back(t);
            inv_dist.push_back(w);
            degree[s]++;
            degree[t]++;
        }
    }

    const int ne = static_cast<int>(src.size());
    wgt.resize(ne);
    for (int i = 0; i < ne; ++i)
        wgt[i] = std::max(1, static_cast<int>(std::llround(inv_dist[i] / w_max * 10000.0)));
    inv_dist.clear();

    // Build compressed sparse row graph
    std::vector<int> xadj(num_nodes + 1, 0);
    for (int u = 0; u < num_nodes; ++u)
        xadj[u + 1] = xadj[u] + degree[u];

    std::vector<int> adjncy(xadj[num_nodes]), adjcwgt(xadj[num_nodes]);
    std::vector<int> cursor(xadj.begin(), xadj.begin() + num_nodes);

    for (int i = 0; i < ne; ++i) {
        int s = src[i], t = dst[i], w = wgt[i];
        adjncy[cursor[s]] = t; adjcwgt[cursor[s]] = w; ++cursor[s];
        adjncy[cursor[t]] = s; adjcwgt[cursor[t]] = w; ++cursor[t];
    }

    auto counts = load_counts(counts_csv);
    std::vector<int> vwgt(num_nodes, 1);
    if (use_node_weights)
        for (int u = 0; u < num_nodes; ++u)
            vwgt[u] = counts[u];

    if (!use_edge_weights)
        std::fill(adjcwgt.begin(), adjcwgt.end(), 1);

    int edgecut = 0;
    kaffpa(&num_nodes, vwgt.data(), xadj.data(), adjcwgt.data(), adjncy.data(),
           &nblocks, &imbalance, /*suppress_output=*/true, seed, mode,
           &edgecut, out_part);
}

// Partition label matching

static std::pair<std::vector<int>, int> match_partitions(
    const int*              new_blocks,
    const std::vector<int>& prev_blocks,
    int num_nodes,
    int nblocks)
{
    std::vector<std::vector<int>> cost(nblocks, std::vector<int>(nblocks, 0));
    for (int i = 0; i < num_nodes; ++i)
        cost[new_blocks[i]][prev_blocks[i]]++;

    std::vector<int> relabel_map;
    maximum_matching(cost, relabel_map);

    std::vector<int> relabeled(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        relabeled[i] = relabel_map[new_blocks[i]];

    int moved = 0;
    for (int i = 0; i < num_nodes; ++i)
        if (relabeled[i] != prev_blocks[i]) ++moved;
    std::cout << "Nodes that changed partition: " << moved
              << " / " << num_nodes << "\n";

    return {relabeled, moved};
}

// Save partition to file

static void save_blocks(const fs::path& path, const std::vector<int>& blocks, int moved_nodes) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path.string());
    for (int b : blocks) f << b << "\n";
    f << moved_nodes << "\n";
}

// Main entry point

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " input_dir [output_dir] num_partitions"
              << " [--no-edge-weights] [--no-node-weights]\n";
    std::exit(1);
}

int main(int argc, char* argv[]) {
    if (argc < 3) usage(argv[0]);

    std::vector<std::string> pos_args;
    bool use_edge_weights = true;
    bool use_node_weights = true;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--no-edge-weights") use_edge_weights = false;
        else if (a == "--no-node-weights") use_node_weights = false;
        else                               pos_args.push_back(a);
    }

    if (pos_args.size() < 2 || pos_args.size() > 3) usage(argv[0]);

    fs::path in_dir  = pos_args[0];
    fs::path out_dir = (pos_args.size() == 3) ? fs::path(pos_args[1]) : in_dir;
    int      nblocks = std::stoi(pos_args.back());

    if (!fs::is_directory(in_dir)) {
        std::cerr << "Error: " << in_dir << " is not a directory\n";
        return 1;
    }
    fs::create_directories(out_dir);

    // Collect and sort input files
    std::vector<fs::path> files;
    std::regex base_re("step_\\d+_base_layer\\.csv");
    for (auto& entry : fs::directory_iterator(in_dir))
        if (entry.is_regular_file() &&
            std::regex_match(entry.path().filename().string(), base_re))
            files.push_back(entry.path());

    if (files.empty()) {
        std::cerr << "No base_layer_step_*.csv files found in " << in_dir << "\n";
        return 1;
    }
    std::sort(files.begin(), files.end(),
              [](const fs::path& a, const fs::path& b) {
                  return step_num_base(a) < step_num_base(b);
              });

    const int    n_steps   = static_cast<int>(files.size());
    const int    mode      = 2;
    const double imbalance = 0.03;
    const int    seed      = 0;

    // Load node count from first step
    const int num_nodes = static_cast<int>(
        load_counts(counts_path(in_dir, step_num_base(files[0]))).size());
    std::cout << num_nodes << " nodes per step, " << n_steps << " steps\n";

    // Allocate shared buffer for partition results
    const size_t buf_bytes = static_cast<size_t>(n_steps) * num_nodes * sizeof(int);
    int* shared_buf = static_cast<int*>(
        mmap(nullptr, buf_bytes, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0));

    if (shared_buf == MAP_FAILED) {
        std::perror("mmap");
        return 1;
    }

    // Parallel partitioning with worker pool
    const int max_workers = static_cast<int>(std::thread::hardware_concurrency());
    std::cout << "Partitioning with pool of " << max_workers << " workers...\n";

    int dispatched = 0;
    std::map<pid_t, int> pid_to_step;

    auto kill_all_children = [&]() {
        for (auto& [pid, _] : pid_to_step)
            kill(pid, SIGTERM);
        while (!pid_to_step.empty()) {
            waitpid(-1, nullptr, 0);
            pid_to_step.erase(pid_to_step.begin());
        }
    };

    while (dispatched < n_steps || !pid_to_step.empty()) {
        while (dispatched < n_steps &&
               static_cast<int>(pid_to_step.size()) < max_workers) {
            int idx  = dispatched++;
            int step = step_num_base(files[idx]);

            pid_t pid = fork();
            if (pid < 0) {
                std::perror("fork");
                kill_all_children();
                munmap(shared_buf, buf_bytes);
                return 1;
            }

            if (pid == 0) {
                try {
                    process_csv(files[idx], counts_path(in_dir, step),
                                use_edge_weights, use_node_weights,
                                nblocks, imbalance, mode, seed,
                                shared_buf + idx * num_nodes,
                                num_nodes);
                    _exit(0);
                } catch (const std::exception& ex) {
                    std::cerr << "[step " << step << "] " << ex.what() << "\n";
                    _exit(1);
                }
            }

            pid_to_step[pid] = idx;
            std::cout << "  Dispatched step " << step << " (pid " << pid << ")\n";
        }

        int status;
        pid_t done = waitpid(-1, &status, 0);

        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            int failed = pid_to_step.count(done) ? pid_to_step[done] : -1;
            std::cerr << "Worker for step " << failed
                      << " (pid " << done << ") failed — aborting.\n";
            pid_to_step.erase(done);
            kill_all_children();
            munmap(shared_buf, buf_bytes);
            return 1;
        }

        int idx = pid_to_step[done];
        pid_to_step.erase(done);
        std::cout << "  Finished step " << step_num_base(files[idx])
                  << " (pid " << done << ")\n";
    }

    // Label matching and output
    std::cout << "All partitions done. Running label matching...\n";
    std::vector<int> prev_blocks;

    for (int i = 0; i < n_steps; ++i) {
        int* raw = shared_buf + i * num_nodes;

        std::vector<int> blocks;
        int moved_nodes = 0;

        if (prev_blocks.empty()) {
            blocks.assign(raw, raw + num_nodes);
        } else {
            std::tie(blocks, moved_nodes) = match_partitions(raw, prev_blocks, num_nodes, nblocks);
        }

        std::string out_name = files[i].filename().string();
        // Replace base_layer with partitions in filename
        auto pos = out_name.find("base_layer");
        if (pos != std::string::npos) out_name.replace(pos, 10, "partitions");
        save_blocks(out_dir / out_name, blocks, moved_nodes);

        std::cout << "Saved step " << step_num_base(files[i])
                  << " (moved " << moved_nodes << ")\n";
        prev_blocks = std::move(blocks);
    }

    munmap(shared_buf, buf_bytes);
    return 0;
}