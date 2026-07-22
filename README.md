# Vector SURGE: A Distributed and Updatable Graph Vector Index

SURGE is a distributed, graph-based approximate nearest neighbor (ANN) index that
supports **online updates**.

This repository contains the SURGE system, the extended streaming benchmark
tooling, and the scripts used to reproduce the experiments.

---

## Setup

Clone the repo and initialize submodules:

```bash
git clone https://github.com/dkhimey/surge.git
cd surge
git submodule update --init --recursive
```

`hnswlib` and `nlohmann/json` are cloned directly under `external/` — no
submodule needed. hnswlib contains a small local patch for deleted label handling.

---

## Prerequisites

- Linux, a C++17 compiler (`g++`), and OpenMP.
- **Open MPI** (`mpic++` / `mpirun`) — the distributed experiments run one
  coordinator plus one process per worker.
- **CMake** — used to build the KaHIP and yaml-cpp libraries.
- If running oracle python scripts, Python 3 with `numpy`, `pyyaml`, `matplotlib`, and `hnswlib`.

```bash
sudo apt -y install libtbb-dev
sudo apt -y install cmake
sudo apt -y install libopenmpi-dev
```

## Building

```bash
make experiments
```

The resulting binaries are:

| Binary | Source | Purpose |
|--------|--------|---------|
| `bin/static_partitioning` | `experiments/static_datasets/static_partitioning.cpp` | Builds and saves the static distributed index (meta-HNSW + per-worker shards) to `<dataset>_<num_partitions>/`, consumed by `static_qps` (index build for Fig. 3). |
| `bin/theoretical_partitioning_quality` | `experiments/static_datasets/theoretical_partitioning_quality.cpp` | Partitioning quality (theoretical recall vs. nprobe) — Fig. 2. |
| `bin/static_qps` | `experiments/static_datasets/static_qps.cpp` | Distributed static recall-vs-QPS sweep over all routing modes — Fig. 3. |
| `bin/dynamic_runbook_experiment` | `experiments/dynamic_runbook_experiment.cpp` | Distributed dynamic run that sweeps all routing modes/params per search step — Figs. 4, 6, 7. |
| `bin/runbook_centers` | `experiments/theoretical_dynamic_simulation/runbook_centers.cpp` | Per-step routing-state generator (centroids, counts, routing HNSW, base layer); feeds the theoretical-recall / threshold-sweep pipeline — Fig. 5. |
| `bin/runbook_partitions_parallel` | `experiments/theoretical_dynamic_simulation/runbook_partitions_parallel.cpp` | KaHIP partitioner over the generated base layer; feeds the theoretical-recall / threshold-sweep pipeline — Fig. 5. |

---

## Datasets and configuration

Experiments are selected by a **dataset key** passed on the command line. Keys and
their file paths are defined in the `DATASETS` registry at the top of
[`src/utils.cpp`](src/utils.cpp); either place your data at the same locations or 
edit this table so each entry points at your
local copy of the data. Each entry provides a `base_file`, `runbook` (streaming
runbook YAML), `query_file`, and `ground_truth_dir`.

The paper evaluates two base datasets, using the standard
[big-ann-benchmarks](https://big-ann-benchmarks.com/) file format:

- **SIFT / BIGANN** — 128-dim, `L2` (registry keys `bigann-*`).
- **MSTuring** — 100-dim, `L2` (registry keys `msturing-*`).

Both are used at 100M and 500M scale, each with three streaming workloads
(`clustered`, `uniform`/`random`, `shift`), e.g. `msturing-100M-clustered`,
`bigann-500M-shift`.

**Runbooks.** The streaming runbooks extend the NeurIPS'23 Big-ANN streaming
generator with `uniform` and `shift` update patterns. Given a runbook YAML and a
base vector file, `bin/runbook_centers` replays the runbook and emits, per step,
the coordinator routing state (`step_NNNNNN_centers.csv`, `_center_counts.csv`,
`_hnsw.bin`, `_base_layer.csv`); `bin/runbook_partitions_parallel` then produces
the KaHIP partition assignment consumed by the dynamic experiments and oracles.

Default evaluation parameters follow the paper: `k = 10`, target recall `0.9`,
10 workers, and a routing layer of 10,000 centroids (`NCENTERS`).

---

## Reproducing the experiments

Run all scripts from the repository root. The scripts contain the exact dataset
keys, thresholds, and paths used in the paper.

### Static quality and routing

Partitioning/routing quality, no local search (**Fig. 2**):

```bash
# Runs bin/theoretical_partitioning_quality, then the routing oracles.
./scripts/run_theoretical_partition_quality.sh <dataset> <num_partitions> <sample_size>
```

Static recall-vs-QPS, sweeping all routing modes on the cluster (**Fig. 3**). 

```bash
# 1. Build the static index (meta-HNSW + per-worker shards).
mpirun -np <num_workers+1> ./bin/static_partitioning \
    <dataset> <num_partitions>

# 2. Sweep routing modes against the pre-built index.
mpirun -np <num_workers+1> ./bin/static_qps \
    <dataset> <num_partitions> <k> <output_file>
```

### Dynamic / streaming experiments

The script sweeps the three routing modes at each search step of a streaming runbook:

**Fig 6 & 7**

```bash
mpirun -np <num_workers+1> ./bin/dynamic_runbook_experiment \
    <dataset> <num_partitions> <full_threshold> <k> <gt_prefix> <output_file> \
    [--init-state-dir <dir> --init-partitions <file> [--init-state-step <N>]] \
    [--search-fraction <p1,p2,...>]
```

- `full_threshold` — number of centroids that must migrate to trigger a routing
  rebuild. With 10,000 centroids, the paper's maintenance threshold τ = 0.6
  corresponds to `6000`; set it `>= 10000` (`NCENTERS`) to disable maintenance
  (the "never maintain" baseline).
- `--init-state-dir` / `--init-partitions` — start from a pre-generated
  `runbook_centers` state instead of building from scratch.
- `--search-fraction` — run each search step at one or more target search:update
  ratios in a single pass; used for the **Fig 7** time breakdown.

For convenience:

- `scripts/run_overnight.sh [rankfile]` — **Fig. 4** (index quality under updates):
  builds from scratch and sweeps the 100M runbooks at different maintenance thresholds.
- `scripts/run_overnight_500M.sh [rankfile]` — **Figs. 6–7** (end-to-end
  performance and time breakdown): the 500M end-to-end runs.
- `scripts/run_threshold_sweep.sh` — **Fig. 5**: sweeps the maintenance threshold.

---

## Citation

```bibtex
#TODO
```