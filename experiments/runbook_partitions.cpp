#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.h"

// KaHIP C interface — assumed available via your include path
extern "C" {
#include "kaHIP_interface.h"
}

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// CSV helpers
// ---------------------------------------------------------------------------

static std::vector<int> load_counts(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path.string());

    std::vector<int> counts;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) counts.push_back(std::stoi(line));
    return counts;
}

// ---------------------------------------------------------------------------
// Step number extraction from filenames
// ---------------------------------------------------------------------------

static int step_num_base(const fs::path& p) {
    // step_000002_base_layer.csv -> 2
    auto name = p.stem().string(); // "step_000002_base_layer"
    return std::stoi(name.substr(6, name.find('_', 6) - 6)); // skip "step_", read up to first '_'
}

// ---------------------------------------------------------------------------
// Build KaFFPa CSR graph and run partitioner
// ---------------------------------------------------------------------------

static std::vector<int> process_csv(
    const fs::path& edges_csv,
    const fs::path& counts_csv,
    bool   use_edge_weights,
    bool   use_node_weights,
    int    nblocks,
    double imbalance,
    int    mode,
    int    seed)
{
    // -----------------------------------------------------------------------
    // Pass 1 — read raw triples, find w_max, accumulate per-node degree.
    //
    // We store (source, target, weight) as three parallel flat arrays rather
    // than a vector of structs so that the data we actually need for CSR
    // construction is laid out without padding or indirection.
    // -----------------------------------------------------------------------
    std::ifstream ef(edges_csv);
    if (!ef) throw std::runtime_error("Cannot open " + edges_csv.string());

    std::vector<int> src, dst;
    std::vector<int> wgt;  // final integer weights, filled after w_max known
    std::vector<double> inv_dist;

    int    num_nodes = 0;
    double w_max     = 0.0;

    {
        std::string line;
        while (std::getline(ef, line)) {
            if (line.empty()) continue;

            // Fast manual parse — avoids istringstream overhead per line
            const char* p = line.c_str();
            char* end;
            int   s = static_cast<int>(std::strtol(p,   &end, 10)); p = end + 1;
            int   t = static_cast<int>(std::strtol(p,   &end, 10)); p = end + 1;
            double d = std::strtod(p, nullptr);

            double w = 1.0 / d;
            if (w > w_max) w_max = w;

            src.push_back(s);
            dst.push_back(t);
            inv_dist.push_back(w);

            num_nodes = std::max(num_nodes, std::max(s, t) + 1);
        }
    }

    const int nedges_directed = static_cast<int>(src.size()); // undirected × 2 later

    // Compute integer weights and degree counts in one pass
    wgt.resize(nedges_directed);
    std::vector<int> degree(num_nodes, 0);

    for (int i = 0; i < nedges_directed; ++i) {
        wgt[i] = std::max(1, static_cast<int>(std::llround(inv_dist[i] / w_max * 10000.0)));
        degree[src[i]]++;
        degree[dst[i]]++;   // undirected: both directions
    }
    inv_dist.clear();       // no longer needed — free the memory

    // -----------------------------------------------------------------------
    // Pass 2 — build CSR directly into final KaHIP arrays.
    //
    // xadj is computed from degree counts via prefix-sum; adjncy/adjcwgt are
    // filled by walking the edge list once and advancing per-node cursors.
    // No adjacency-list-of-vectors is created at any point.
    // -----------------------------------------------------------------------
    std::vector<int> xadj(num_nodes + 1, 0);
    for (int u = 0; u < num_nodes; ++u)
        xadj[u + 1] = xadj[u] + degree[u];

    const int total_arcs = xadj[num_nodes]; // 2 × nedges_directed
    std::vector<int> adjncy(total_arcs), adjcwgt(total_arcs);

    // cursor[u] tracks the next free slot in adjncy for node u
    std::vector<int> cursor(xadj.begin(), xadj.begin() + num_nodes);

    for (int i = 0; i < nedges_directed; ++i) {
        int s = src[i], t = dst[i], w = wgt[i];

        adjncy [cursor[s]] = t;  adjcwgt[cursor[s]] = w;  ++cursor[s];
        adjncy [cursor[t]] = s;  adjcwgt[cursor[t]] = w;  ++cursor[t];
    }

    // -----------------------------------------------------------------------
    // Node weights and optional overrides
    // -----------------------------------------------------------------------
    auto counts = load_counts(counts_csv);

    std::vector<int> vwgt(num_nodes, 1);
    if (use_node_weights)
        for (int u = 0; u < num_nodes; ++u)
            vwgt[u] = counts[u];

    if (!use_edge_weights)
        std::fill(adjcwgt.begin(), adjcwgt.end(), 1);

    // -----------------------------------------------------------------------
    // Run KaFFPa
    // -----------------------------------------------------------------------
    int edgecut = 0;
    std::vector<int> part(num_nodes, 0);

    kaffpa(&num_nodes, vwgt.data(), xadj.data(), adjcwgt.data(), adjncy.data(),
           &nblocks, &imbalance, /*suppress_output=*/true, seed, mode,
           &edgecut, part.data());

    return part;
}

// ---------------------------------------------------------------------------
// Relabel new_blocks to maximise overlap with prev_blocks
// ---------------------------------------------------------------------------

static std::vector<int> match_partitions(
    const std::vector<int>& new_blocks,
    const std::vector<int>& prev_blocks,
    int nblocks)
{
    std::vector<std::vector<int>> cost(nblocks, std::vector<int>(nblocks, 0));
    for (int i = 0; i < static_cast<int>(new_blocks.size()); ++i)
        cost[new_blocks[i]][prev_blocks[i]]++;

    std::vector<int> relabel_map;
    maximum_matching(cost, relabel_map);

    std::vector<int> relabeled(new_blocks.size());
    for (int i = 0; i < static_cast<int>(new_blocks.size()); ++i)
        relabeled[i] = relabel_map[new_blocks[i]];

    int moved = 0;
    for (int i = 0; i < static_cast<int>(new_blocks.size()); ++i)
        if (relabeled[i] != prev_blocks[i]) ++moved;
    std::cout << "Nodes that changed partition: " << moved
              << " / " << new_blocks.size() << "\n";

    return relabeled;
}

// ---------------------------------------------------------------------------
// Save partition vector
// ---------------------------------------------------------------------------

static void save_blocks(const fs::path& path, const std::vector<int>& blocks) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path.string());
    for (int b : blocks) f << b << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

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

    std::vector<fs::path> files;
    std::regex base_re("step_\\d+_base_layer\\.csv");
    for (auto& entry : fs::directory_iterator(in_dir)) {
        if (entry.is_regular_file() &&
            std::regex_match(entry.path().filename().string(), base_re))
            files.push_back(entry.path());
    }

    if (files.empty()) {
        std::cerr << "No step*_base_layer.csv files found in " << in_dir << "\n";
        return 1;
    }
    std::sort(files.begin(), files.end(),
              [](const fs::path& a, const fs::path& b) {
                  return step_num_base(a) < step_num_base(b);
              });

    const int    mode      = 2;
    const double imbalance = 0.03;
    const int    seed      = 0;

    std::vector<int> prev_blocks;

    for (const auto& f : files) {
        int step = step_num_base(f);
        std::cout << "Processing " << f << "...\n";

        std::ostringstream oss;
        oss << "step_" << std::setw(6) << std::setfill('0') << step << "_center_counts.csv";

        auto blocks = process_csv(f, in_dir / oss.str(),
                                  use_edge_weights, use_node_weights,
                                  nblocks, imbalance, mode, seed);

        if (!prev_blocks.empty())
            blocks = match_partitions(blocks, prev_blocks, nblocks);

        std::string out_name = f.filename().string();
        auto pos = out_name.find("base_layer");
        if (pos != std::string::npos) out_name.replace(pos, 10, "partitions");

        save_blocks(out_dir / out_name, blocks);
        std::cout << "Saved partitions (" << blocks.size()
                  << " nodes) to " << (out_dir / out_name) << "\n";

        prev_blocks = std::move(blocks);
    }

    return 0;
}