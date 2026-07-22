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
- **CMake** — used to build the KaHIP library.
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
| `bin/dynamic_runbook_experiment` | `experiments/dynamic_runbook_experiment.cpp` | End-to-end distributed dynamic run that sweeps all routing modes/params per search step — Figs. 6, 7. |
| `bin/runbook_centers` | `experiments/theoretical_dynamic_simulation/runbook_centers.cpp` | Per-step routing-state generator (centroids, counts, routing HNSW, base layer) — Figs. 4–5. |
| `bin/runbook_partitions_parallel` | `experiments/theoretical_dynamic_simulation/runbook_partitions_parallel.cpp` | Per-step partitioner over an existing base layer — Figs. 4–5. |

---

## Datasets and configuration

Experiments are selected by a **dataset key** passed on the command line. Keys and
their file paths are defined in the `DATASETS` registry at the top of
[`src/utils.cpp`](src/utils.cpp); either place your data at the same locations or 
edit this table so each entry points at your
local copy of the data.
The paper evaluates two base datasets, using the standard
[big-ann-benchmarks](https://big-ann-benchmarks.com/) file format:

- **SIFT / BIGANN** — 128-dim, `L2`.
- **MSTuring** — 100-dim, `L2`.

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

Partitioning & theoretical recall, no local graphs **(Figure 2)**:

```bash
# build routing layer & compute theoretical recall
./bin/theoretical_partitioning_quality <dataset> <num_partitions> <sample_size>
# run oracles
python oracles/static_datasets/nprobe_oracle.py         --router <run_dir>/metaHNSW.bin --partition-file <run_dir>/partitions.bin ...
python oracles/static_datasets/recall_target_oracle.py  --router <run_dir>/metaHNSW.bin --partition-file <run_dir>/partitions.bin ...
python oracles/static_datasets/branching_factor_oracle.py --router <run_dir>/metaHNSW.bin --partition-file <run_dir>/partitions.bin ...
```

Recall-vs-QPS **(Figure 3)**:

```bash
# 1. Build the static index (meta-HNSW + per-worker shards).
mpirun -np <num_workers+1> ./bin/static_partitioning \
    <dataset> <num_partitions>

# 2. Run searches, sweep routing modes and params against the index.
mpirun -np <num_workers+1> ./bin/static_qps \
    <dataset> <num_partitions> <k> <output_file>
```

### Dynamic theoretical recall (Figs. 4–5)

Figs. 4 and 5 come from *simulated* runs that isolate routing and partitioning
from local search. Three stages:

**1. Per-step routing state.** Replay the runbook and, at every step, update the
routing layer for the current active set:

```bash
./bin/runbook_centers \
    --dataset <dataset> --centers 10000 \
    --out-dir cluster_history_<dataset>_10000
```

The base vectors and runbook are read from the `DATASETS` registry for the key. Per
step this writes `step_NNNNNN_centers.csv`, `_center_counts.csv`, `_hnsw.bin` (the
routing HNSW), and `_base_layer.csv` (the proximity graph). Outputs are saved to the `cluster_history_<dataset>_10000` - the "base dir" used in step 3.

**2. Per-step partitioning.** Partition each step's base layer into
`num_partitions` blocks with KaHIP:

```bash
./bin/runbook_partitions_parallel \
    cluster_history_<dataset>_10000 \
    cluster_history_<dataset>_10000_<num_partitions> \
    <num_partitions>
```

This writes `step_NNNNNN_partitions.csv` (centroid→worker) per step to `cluster_history_<dataset>_10000_<num_partitions>` — the "partitions dir" used in step 3.

**3. Theoretical recall across steps.** For a routing mode, route the query set at
each step using that step's HNSW + partitions and measure the fraction of true
neighbors in the selected partitions:

```bash
python oracles/theoretical_dynamic_simulation/compute_theoretical_recall_updated.py \
    --base-dir       cluster_history_<dataset>_10000 \
    --partitions-dir cluster_history_<dataset>_10000_<num_partitions> \
    --runbook-path <runbook.yaml> --query-file <query> --base-file <base> --gt-dir <gt> \
    --mode {NProbe|RecallTarget|BranchingFactor} \
    --threshold <τ>
```

`--threshold τ` simulates maintenance: the router/partitions switch to a fresh state only when drift exceeds τ.

<!-- (The `run_branching_factor_sweep.sh` / `run_nprobe_sweep.sh` / `run_threshold_sweep.sh`
scripts sweep steps and params with a fixed `--mode`.) -->

### End-to-end dynamic performance (Figs. 6–7)

Replay a streaming runbook end-to-end (insert/delete/search with local search and
online maintenance), sweeping the three routing modes at each search step:

```bash
mpirun -np <num_workers+1> ./bin/dynamic_runbook_experiment \
    <dataset> <num_partitions> <full_threshold> <k> <gt_prefix> <output_file> \
    [--init-state-dir <dir> --init-partitions <file> [--init-state-step <N>]] \
    [--search-fraction <p1,p2,...>]
```

- `full_threshold` — centroids that must migrate to trigger a routing rebuild.
  With 10,000 centroids, τ = 0.6 corresponds to `6000`; set it `>= 10000`
  (`NCENTERS`) to disable maintenance (the "never maintain" baseline).
- `--init-state-dir` / `--init-partitions` — start from a pre-generated
  `runbook_centers` state (see below) instead of building from scratch.
- `--search-fraction` — run each search step at one or more search:update ratios
  in one pass; used for the **Fig. 7** time breakdown.

<!-- Fig. 6 uses the 500M datasets; `scripts/run_overnight_500M.sh` batches these runs
across datasets and thresholds (`run_overnight.sh` is the 100M variant). -->

---

## Citation

```bibtex
#TODO
```