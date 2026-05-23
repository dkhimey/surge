"""
compute_theoretical_recall_sweep.py

Sweeps all three routing modes and their full parameter grids to compute
theoretical recall (assuming perfect recall within each partition), matching
the sweep grids used in experiments/shared_batch_experiment_sweep.cpp.

The three routing modes correspond to the implementations in src/index.cpp:

  BranchingFactor  – getPartitionsForSearch_Branching_
      Query the HNSW router for k=bf nearest centers; take all unique
      partitions among those k centers.

  NProbe           – getPartitionsForSearch_Nprobe_
      Expand the HNSW k query until nprobe unique partitions are found,
      then walk centers in nearest-first order collecting partitions until
      nprobe unique ones are seen.

  RecallTarget     – getPartitionsForSearch_RecallTgt_
      Query HNSW for the nearest 50 centers, compute a score for each
      partition as the sum of (part_size * exp(-d/d0)) over its centers,
      normalise to a probability distribution, then greedily visit
      partitions in descending probability order until the cumulative
      probability meets the recall target.

Usage
-----
python compute_theoretical_recall_sweep.py \
    --base-dir      /path/to/hnsw_output \
    --partitions-dir /path/to/partitions \
    --runbook-path  /path/to/runbook.yaml \
    --query-file    /path/to/queries.fbin \
    --base-file     /path/to/base.fbin \
    --gt-dir        /path/to/gt \
    [--no-rebuilds]

Outputs
-------
  {partitions-dir}/full_results_sweep.csv          (default)
  {partitions-dir}/full_results_sweep_no_rebuilds.csv  (with --no-rebuilds)

Column naming convention:
  {metric}_{tag}_{mode}_{param}

  metric : recall | activation | query_time_s
  tag    : rebuild | no_rebuilds
  mode   : BranchingFactor | NProbe | RecallTarget
  param  : integer for BranchingFactor/NProbe; two-decimal float for RecallTarget

Plus cof / cof_no_rebuilds columns (coefficient of variation of per-partition
activation counts, computed using BranchingFactor=1 as the reference combo).
"""

import numpy as np
import struct
from pathlib import Path
import re
import yaml
import argparse
import time
import pandas as pd
import hnswlib
from typing import Dict, List, Tuple, Any, Optional

# ─── Sweep parameter grids (match shared_batch_experiment_sweep.cpp) ──────────

BRANCHING_FACTOR_PARAMS: List[int]   = [1, 2, 5, 10, 20, 40, 80]
NPROBE_PARAMS:           List[int]   = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TARGET_PARAMS:           List[float] = [0.60, 0.70, 0.75, 0.80, 0.85,
                                        0.90, 0.95, 0.97, 0.98, 0.99]


def build_sweep_combos() -> List[Tuple[str, float]]:
    """Ordered list of (mode, param) pairs matching build_sweep_combos() in C++."""
    combos: List[Tuple[str, float]] = []
    for v in BRANCHING_FACTOR_PARAMS:
        combos.append(("BranchingFactor", float(v)))
    for v in NPROBE_PARAMS:
        combos.append(("NProbe", float(v)))
    for v in TARGET_PARAMS:
        combos.append(("RecallTarget", float(v)))
    return combos


SWEEP_COMBOS = build_sweep_combos()


# ─── File I/O helpers ─────────────────────────────────────────────────────────

def read_fbin_slice(filename: str, start: int, end: int) -> np.ndarray:
    """Read vectors [start, end) from an fbin file, returned as float32."""
    with open(filename, "rb") as f:
        header = f.read(8)
        num_points, dim = struct.unpack("II", header)
        print(num_points)
        if start < 0 or end > num_points:
            raise ValueError("Slice out of bounds")
        count = end - start
        vec_size = dim * 4
        f.seek(8 + start * vec_size, 0)
        data = np.fromfile(f, dtype=np.float32, count=count * dim)
        return data.reshape(count, dim)


def read_u8bin_slice(filename: str, start: int, end: int) -> np.ndarray:
    """Read vectors [start, end) from a u8bin file, returned as float32."""
    with open(filename, "rb") as f:
        header = f.read(8)
        num_points, dim = struct.unpack("II", header)
        print(num_points)
        if start < 0 or end > num_points:
            raise ValueError("Slice out of bounds")
        count = end - start
        vec_size = dim * 1  # 1 byte per element
        f.seek(8 + start * vec_size, 0)
        data = np.fromfile(f, dtype=np.uint8, count=count * dim)
        return data.reshape(count, dim).astype(np.float32)


def mmap_fbin(filename: str) -> np.memmap:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_points, dim = header
    return np.memmap(filename, dtype=np.float32, mode="r",
                     offset=8, shape=(num_points, dim))


def mmap_u8bin(filename: str) -> np.memmap:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_points, dim = header
    return np.memmap(filename, dtype=np.uint8, mode="r",
                     offset=8, shape=(num_points, dim))


def read_fbin_ground_truth(filename: str) -> List[List[int]]:
    with open(filename, "rb") as f:
        num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        K = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        total_elements = num_queries * K
        all_ids = np.frombuffer(f.read(total_elements * 4), dtype=np.uint32)
        if all_ids.size != total_elements:
            raise ValueError("Failed to read neighbor IDs")
        f.seek(total_elements * 4, 1)  # skip distances
        return all_ids.reshape((num_queries, K)).astype(int).tolist()


def parse_runbook(path: str) -> Dict[int, int]:
    """Parse runbook YAML, returning {step: delta_count} for insert/delete steps."""
    with open(path) as f:
        data = yaml.safe_load(f)
    root = next(iter(data.values()))
    deltas: Dict[int, int] = {}
    for step, payload in root.items():
        if not isinstance(step, int):
            continue
        op    = payload.get("operation")
        start = payload.get("start")
        end   = payload.get("end")
        if op in {"delete", "insert"} and start is not None and end is not None:
            count = end - start + 1
            deltas[step] = -count if op == "delete" else count
        else:
            deltas[step] = 0
    return deltas


def find_first_available_step(base_dir: str, partitions_dir: str) -> int:
    """Auto-detect the smallest step N for which both
    step_NNNNNN_hnsw.bin and step_NNNNNN_partitions.csv exist.

    Works with output directories produced by msturing-cluster-analysis.cpp
    when run with --start-step N (so the first file may not be step_000001_*).
    """
    router_re    = re.compile(r"^step_(\d+)_hnsw\.bin$")
    partition_re = re.compile(r"^step_(\d+)_partitions\.csv$")

    router_steps: set = set()
    for f in Path(base_dir).iterdir():
        m = router_re.match(f.name)
        if m:
            router_steps.add(int(m.group(1)))

    partition_steps: set = set()
    for f in Path(partitions_dir).iterdir():
        m = partition_re.match(f.name)
        if m:
            partition_steps.add(int(m.group(1)))

    common = router_steps & partition_steps
    if not common:
        raise FileNotFoundError(
            f"Could not auto-detect a starting step.\n"
            f"  base-dir       = {base_dir}\n"
            f"    HNSW files found at steps: "
            f"{sorted(router_steps) if router_steps else 'none'}\n"
            f"  partitions-dir = {partitions_dir}\n"
            f"    partition files found at steps: "
            f"{sorted(partition_steps) if partition_steps else 'none'}\n"
            f"  Need at least one step with both an HNSW file and a partitions file."
        )
    return min(common)


# ─── Shared recall metrics ────────────────────────────────────────────────────

def _compute_recall(
    visited: Dict[int, set],
    ground_truth_partitions: Dict[int, List[int]],
    num_queries: int,
) -> float:
    total = 0.0
    for i in range(num_queries):
        gt   = ground_truth_partitions[i]
        hits = sum(1 for p in gt if p in visited[i])
        total += hits / len(gt)
    return total / num_queries


def _compute_activation(
    visited: Dict[int, set],
    num_queries: int,
    num_partitions: int,
) -> float:
    return sum(len(v) / num_partitions for v in visited.values()) / num_queries


def _compute_per_partition_counts(
    visited: Dict[int, set],
    num_partitions: int,
) -> np.ndarray:
    counts = np.zeros(num_partitions, dtype=np.int64)
    for vps in visited.values():
        for pid in vps:
            counts[pid] += 1
    return counts


# ─── Per-step sweep computation ───────────────────────────────────────────────

def compute_step_sweep(
    queries:        np.ndarray,
    base_mmap,
    gt_file:        str,
    base_dir:       str,
    partition_dir:  str,
    hnsw_step:      int,
    partition_step: int,
    k_neighbors:    int,
    num_partitions: int,
) -> Tuple[
    Dict[Tuple[str, float], Tuple[float, float, np.ndarray, float]],
    Dict[float, Tuple[float, float]],
]:
    """
    Run all (mode, param) sweep combinations for a single runbook step,
    and compute oracle lower bounds for each recall target.

    Parameters
    ----------
    queries         : query vectors, shape (Q, dim), float32
    base_mmap       : mmap of the base vector file
    gt_file         : path to the ground-truth file for this step
    base_dir        : directory with step_NNNNNN_hnsw.bin files
    partition_dir   : directory with step_NNNNNN_partitions.csv files
    hnsw_step       : step number to load HNSW from
    partition_step  : step number to load partitions from
    k_neighbors     : number of ground-truth neighbors (k for recall@k)
    num_partitions  : total number of partitions

    Returns
    -------
    routing_results : dict keyed by (mode_str, param) ->
        (recall, activation_rate, per_partition_counts, query_time_s)
    """
    ground_truth = read_fbin_ground_truth(gt_file)
    num_queries  = len(ground_truth)

    # Load HNSW router
    router = hnswlib.Index(space='l2', dim=100)
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    current_count = router.get_current_count()

    # Load partition assignments: center_id -> partition_id
    partition_file = f"{partition_dir}/step_{partition_step:06d}_partitions.csv"
    partitions = np.loadtxt(partition_file, delimiter=",", dtype=int)[:-1]

    # ── Route GT vectors to determine their partition assignments ─────────────
    # Each GT neighbor is assigned to the partition of its nearest center.
    gt_indices = np.empty(num_queries * k_neighbors, dtype=np.int64)
    pos = 0
    for gt_list in ground_truth:
        gt_indices[pos:pos + k_neighbors] = gt_list[:k_neighbors]
        pos += k_neighbors

    vecs_gt = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)
    router.set_ef(100)
    labels_gt, _ = router.knn_query(vecs_gt, k=1)
    gt_part_ids  = partitions[labels_gt[:, 0]]

    ground_truth_partitions: Dict[int, List[int]] = {}
    pos = 0
    for i in range(num_queries):
        ground_truth_partitions[i] = gt_part_ids[pos:pos + k_neighbors].tolist()
        pos += k_neighbors

    results: Dict[Tuple[str, float], Tuple[float, float, np.ndarray, float]] = {}

    # ── BranchingFactor sweep ──────────────────────────────────────────────────
    # Mirrors getPartitionsForSearch_Branching_: query HNSW for k nearest
    # centers, collect all unique partitions among those k centers.
    #
    # Optimization: query once for max_bf and reuse prefix for smaller values.
    max_bf = max(BRANCHING_FACTOR_PARAMS)
    k_bf   = min(max_bf, current_count)
    router.set_ef(max(100, k_bf))
    t0 = time.perf_counter()
    labels_bf_all, _ = router.knn_query(queries, k=k_bf)
    t_bf = time.perf_counter() - t0

    for bf in BRANCHING_FACTOR_PARAMS:
        k          = min(bf, current_count)
        labels_bf  = labels_bf_all[:, :k]

        visited: Dict[int, set] = {}
        for i, label_list in enumerate(labels_bf):
            visited[i] = set(int(partitions[l]) for l in label_list)

        recall     = _compute_recall(visited, ground_truth_partitions, num_queries)
        activation = _compute_activation(visited, num_queries, num_partitions)
        ppc        = _compute_per_partition_counts(visited, num_partitions)
        results[("BranchingFactor", float(bf))] = (recall, activation, ppc, t_bf)

    # ── NProbe sweep ───────────────────────────────────────────────────────────
    # Mirrors getPartitionsForSearch_Nprobe_: expand HNSW k until every query
    # has at least max(NPROBE_PARAMS) unique partitions in its result list,
    # then walk centers in nearest-first order collecting nprobe unique
    # partitions per query.
    #
    # The expansion loop matches the C++ while-loop (starting at nprobe and
    # multiplying by 10 each iteration).  All nprobe values share one final
    # knn_query result.
    max_nprobe = max(NPROBE_PARAMS)
    router.set_ef(100)
    k_np = max_nprobe
    labels_np_all: Optional[np.ndarray] = None
    t_np = 0.0
    while True:
        k_np = min(k_np, current_count)
        t0 = time.perf_counter()
        labels_np_all, _ = router.knn_query(queries, k=k_np)
        t_np = time.perf_counter() - t0  # time of the final (used) call
        min_unique = min(
            len(set(int(partitions[l]) for l in ll))
            for ll in labels_np_all
        )
        if min_unique >= max_nprobe or k_np >= current_count:
            break
        k_np *= 10

    def first_p_unique(label_list, p: int) -> set:
        """Collect the first p unique partition IDs from label_list."""
        seen: set = set()
        for label in label_list:
            pid = int(partitions[label])
            if pid not in seen:
                seen.add(pid)
                if len(seen) == p:
                    break
        return seen

    for nprobe in NPROBE_PARAMS:
        visited: Dict[int, set] = {}
        for i, label_list in enumerate(labels_np_all):
            visited[i] = first_p_unique(label_list, nprobe)

        recall     = _compute_recall(visited, ground_truth_partitions, num_queries)
        activation = _compute_activation(visited, num_queries, num_partitions)
        ppc        = _compute_per_partition_counts(visited, num_partitions)
        results[("NProbe", float(nprobe))] = (recall, activation, ppc, t_np)

    # ── RecallTarget sweep ─────────────────────────────────────────────────────
    # Mirrors getPartitionsForSearch_RecallTgt_:
    #
    # 1. Query HNSW for the nearest min(50, ncenters) centers.
    # 2. For each center c at squared-L2 distance d to the query:
    #      score contribution = part_size[partition(c)] * exp(−d / d0)
    #    where d0 = distance to the nearest center + ε (scale normalisation)
    #    and part_size[p] = number of centers assigned to partition p (used as
    #    the size prior; matches the C++ fallback when center_counts_ are all
    #    zero, which is always the case in the offline theoretical setting).
    # 3. Normalise scores to a probability distribution over partitions.
    # 4. Walk partitions in descending probability order, accumulating
    #    probability until the cumulative sum meets recall_target.
    k_rt = min(50, current_count)
    router.set_ef(max(100, k_rt))
    t0 = time.perf_counter()
    labels_rt_all, dists_rt_all = router.knn_query(queries, k=k_rt)
    t_rt = time.perf_counter() - t0

    # Size prior: number of centers per partition (computed once, shared
    # across all TARGET_PARAMS iterations).
    part_size = np.zeros(num_partitions, dtype=np.float64)
    for pid in partitions:
        part_size[int(pid)] += 1.0

    # Pre-compute normalised probability vectors and descending partition order
    # for every query — these are the same for all recall targets.
    query_part_probs: List[Tuple[np.ndarray, np.ndarray]] = []
    fallback_pids:    List[int] = []

    for i in range(num_queries):
        label_list = labels_rt_all[i]
        dist_list  = dists_rt_all[i]
        d0 = float(dist_list[0]) + 1e-10  # avoid division by zero

        partition_probs = np.zeros(num_partitions, dtype=np.float64)
        for label, d in zip(label_list, dist_list):
            pid   = int(partitions[label])
            rel_d = float(d) / d0
            partition_probs[pid] += part_size[pid] * np.exp(-rel_d)

        prob_sum = partition_probs.sum()
        fallback = int(partitions[label_list[0]])
        if prob_sum <= 0.0:
            partition_probs[fallback] = 1.0
            prob_sum = 1.0
        partition_probs /= prob_sum

        ordered_pids = np.argsort(-partition_probs)  # descending
        query_part_probs.append((partition_probs, ordered_pids))
        fallback_pids.append(fallback)

    for target in TARGET_PARAMS:
        visited: Dict[int, set] = {}
        for i, (partition_probs, ordered_pids) in enumerate(query_part_probs):
            vps: set = set()
            recall_estimate = 0.0
            for pid in ordered_pids:
                if partition_probs[pid] <= 0.0:
                    break
                vps.add(int(pid))
                recall_estimate += partition_probs[pid]
                if recall_estimate >= target:
                    break
            if not vps:
                vps.add(fallback_pids[i])
            visited[i] = vps

        recall     = _compute_recall(visited, ground_truth_partitions, num_queries)
        activation = _compute_activation(visited, num_queries, num_partitions)
        ppc        = _compute_per_partition_counts(visited, num_partitions)
        results[("RecallTarget", float(target))] = (recall, activation, ppc, t_rt)

    # ── Oracle lower bounds ────────────────────────────────────────────────────
    # For each recall target τ, compute the minimum number of partitions that
    # would need to be visited per query if partitions could be perfectly ranked
    # by GT-neighbour density.  This is derived entirely from ground_truth_partitions
    # (already computed above) so no extra file loading or routing is needed.
    #
    # query_dist[i, p] = fraction of query i's k GT neighbours in partition p.
    # Sorting descending and cumsumming gives the achievable recall curve; the
    # oracle picks the fewest partitions whose cumulative mass meets τ.
    query_dist = np.zeros((num_queries, num_partitions), dtype=np.float64)
    for i in range(num_queries):
        counts = np.bincount(
            np.array(ground_truth_partitions[i], dtype=np.intp),
            minlength=num_partitions,
        )
        query_dist[i] = counts / k_neighbors

    sorted_dist = np.sort(query_dist, axis=1)[:, ::-1]  # (Q, P) descending
    cumsum      = np.cumsum(sorted_dist, axis=1)         # (Q, P)

    oracle_results: Dict[float, Tuple[float, float]] = {}
    for target in TARGET_PARAMS:
        meets = cumsum >= target                          # (Q, P) bool
        idx   = np.where(
            meets.any(axis=1),
            np.argmax(meets, axis=1),
            num_partitions - 1,                          # fallback: all partitions needed
        )                                                # (Q,) 0-indexed
        n_parts         = idx + 1                        # (Q,) number of partitions visited
        achieved_recall = cumsum[np.arange(num_queries), idx]

        oracle_results[float(target)] = (
            float(n_parts.mean()) / num_partitions,      # mean activation
            float(achieved_recall.mean()),               # mean achieved recall
        )

    return results, oracle_results


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep BranchingFactor, NProbe, and RecallTarget routing modes "
            "over their full parameter grids and compute theoretical recall "
            "(assuming perfect recall within each partition)."
        )
    )
    parser.add_argument(
        "--base-dir", required=True,
        help="Directory containing step_NNNNNN_hnsw.bin files "
             "(output of msturing-cluster-analysis.cpp).",
    )
    parser.add_argument(
        "--partitions-dir", required=True,
        help="Directory containing step_NNNNNN_partitions.csv files "
             "(output of runbook_partitions_parallel.cpp).",
    )
    parser.add_argument(
        "--no-rebuilds", action="store_true",
        help="Compute only no-rebuild statistics "
             "(always use the step-start router and partitions).",
    )
    parser.add_argument(
        "--runbook-path", required=True,
        help="Path to the runbook YAML file.",
    )
    parser.add_argument(
        "--query-file", required=True,
        help="Path to the query file (.fbin or .u8bin format).",
    )
    parser.add_argument(
        "--base-file", required=True,
        help="Path to the base vectors file (.fbin or .u8bin format).",
    )
    parser.add_argument(
        "--gt-dir", required=True,
        help="Directory containing ground-truth files (step<N+1>.gt100).",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Parse runbook to get the list of steps
    deltas        = parse_runbook(args.runbook_path)
    runbook_steps = sorted(deltas)

    # Load queries and base vectors
    if args.query_file.endswith(".u8bin"):
        queries   = read_u8bin_slice(args.query_file, 0, 10000)
        base_mmap = mmap_u8bin(args.base_file)
    else:
        queries   = read_fbin_slice(args.query_file, 0, 10000)
        base_mmap = mmap_fbin(args.base_file)

    base_directory       = args.base_dir
    partitions_directory = args.partitions_dir

    # Auto-detect the starting step
    step_start = find_first_available_step(base_directory, partitions_directory)
    print(f"Starting step (auto-detected from directory contents): {step_start}")

    # Determine number of partitions from the first partition file
    first_partition_file = (
        f"{partitions_directory}/step_{step_start:06d}_partitions.csv"
    )
    first_partitions = np.loadtxt(first_partition_file, delimiter=",", dtype=int)[:-1]
    num_partitions   = int(first_partitions.max()) + 1
    print(f"Found {num_partitions} partitions")

    # ── Determine which steps to evaluate ────────────────────────────────────

    # Rebuild steps: walk even-spaced steps from step_start while HNSW files exist
    rebuild_steps: List[int] = []
    if not args.no_rebuilds:
        step = step_start
        while True:
            if not Path(f"{base_directory}/step_{step:06d}_hnsw.bin").exists():
                break
            rebuild_steps.append(step)
            step += 2

    # No-rebuild steps: same cadence as rebuilds but always use step_start router
    no_rebuild_steps = [
        s for s in runbook_steps
        if s >= step_start and (s - step_start) % 2 == 0
    ]

    # processed_steps: the steps for which we will also run the no-rebuild pass
    processed_steps: List[int] = []
    if not args.no_rebuilds:
        for step in rebuild_steps:
            gt_test        = f"{args.gt_dir}/step{step + 1}.gt100"
            partition_file = f"{partitions_directory}/step_{step:06d}_partitions.csv"
            hnsw_file      = f"{base_directory}/step_{step:06d}_hnsw.bin"
            required = [gt_test, partition_file, hnsw_file]
            if not all(Path(f).exists() for f in required):
                missing = [f for f in required if not Path(f).exists()]
                print(
                    f"Missing file(s) at step {step}: "
                    f"{', '.join(missing)}. Stopping rebuild scan."
                )
                break
            processed_steps.append(step)
    else:
        # In --no-rebuilds mode check that at least the initial files exist
        initial_hnsw      = f"{base_directory}/step_{step_start:06d}_hnsw.bin"
        initial_partition = f"{partitions_directory}/step_{step_start:06d}_partitions.csv"
        for f in [initial_hnsw, initial_partition]:
            if not Path(f).exists():
                raise FileNotFoundError(
                    f"Missing required step-start file for --no-rebuilds mode: {f}"
                )
        for step in no_rebuild_steps:
            gt_test = f"{args.gt_dir}/step{step + 1}.gt100"
            if not Path(gt_test).exists():
                print(f"Missing GT file at step {step}: {gt_test}. Skipping.")
                continue
            processed_steps.append(step)

    # ── Run evaluations ───────────────────────────────────────────────────────
    # rebuild_results[step]    = {(mode, param): (recall, activation, ppc, t)}
    # no_rebuild_results[step] = {(mode, param): (recall, activation, ppc, t)}
    # *_oracle_results[step]   = {target: (oracle_activation, oracle_recall)}
    rebuild_results:          Dict[int, Dict] = {}
    no_rebuild_results:       Dict[int, Dict] = {}
    rebuild_oracle_results:   Dict[int, Dict] = {}
    no_rebuild_oracle_results: Dict[int, Dict] = {}

    # Rebuild pass — per-step router and partitions
    if not args.no_rebuilds:
        for step in processed_steps:
            gt_test = f"{args.gt_dir}/step{step + 1}.gt100"
            print(f"\nREBUILDING: evaluating step {step} ...")
            res, oracle = compute_step_sweep(
                queries, base_mmap, gt_test,
                base_directory, partitions_directory,
                hnsw_step=step, partition_step=step,
                k_neighbors=10, num_partitions=num_partitions,
            )
            rebuild_results[step]        = res
            rebuild_oracle_results[step] = oracle
            for (mode, param), (recall, activation, _, _) in res.items():
                print(
                    f"  [{mode}={param:.4g}] "
                    f"recall={recall:.4f}  activation={activation:.4f}"
                )

    # No-rebuild pass — always use step_start router and partitions
    for step in processed_steps:
        gt_test = f"{args.gt_dir}/step{step + 1}.gt100"
        if not Path(gt_test).exists():
            continue
        print(f"\nNO REBUILDING: evaluating step {step} ...")
        res, oracle = compute_step_sweep(
            queries, base_mmap, gt_test,
            base_directory, partitions_directory,
            hnsw_step=step_start, partition_step=step_start,
            k_neighbors=10, num_partitions=num_partitions,
        )
        no_rebuild_results[step]        = res
        no_rebuild_oracle_results[step] = oracle
        for (mode, param), (recall, activation, _, _) in res.items():
            print(
                f"  [{mode}={param:.4g}] "
                f"recall={recall:.4f}  activation={activation:.4f}"
            )

    # ── Build output DataFrame ────────────────────────────────────────────────
    # Tidy / long format: one row per (step, rebuild_tag, mode, param).
    # Columns: step, rebuild, mode, param, recall, activation, cof, query_time_s
    # Oracle rows use mode="Oracle" and carry no cof or query_time_s.
    all_steps = sorted(set(rebuild_results) | set(no_rebuild_results))
    if not all_steps:
        print("No steps were evaluated. Exiting.")
        raise SystemExit(0)

    rows: List[Dict[str, Any]] = []
    for step in all_steps:
        tags = []
        if not args.no_rebuilds and step in rebuild_results:
            tags.append(("rebuild",     rebuild_results[step],     rebuild_oracle_results[step]))
        if step in no_rebuild_results:
            tags.append(("no_rebuilds", no_rebuild_results[step],  no_rebuild_oracle_results[step]))

        for tag, res, oracle in tags:
            # Routing rows
            for (mode, param) in SWEEP_COMBOS:
                r, act, ppc, t = res[(mode, param)]
                mean_val = np.mean(ppc)
                cof = (np.std(ppc) / mean_val) if mean_val > 0 else float("nan")
                rows.append({
                    "step":         step,
                    "rebuild":      tag,
                    "mode":         mode,
                    "param":        param,
                    "recall":       r,
                    "activation":   act,
                    "cof":          cof,
                    "query_time_s": t,
                })
            # Oracle rows — one per recall target
            for target in TARGET_PARAMS:
                oracle_act, oracle_recall = oracle[float(target)]
                rows.append({
                    "step":         step,
                    "rebuild":      tag,
                    "mode":         "Oracle",
                    "param":        target,
                    "recall":       oracle_recall,
                    "activation":   oracle_act,
                    "cof":          float("nan"),
                    "query_time_s": float("nan"),
                })

    table = pd.DataFrame(rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_name = (
        "full_results_sweep_no_rebuilds.csv"
        if args.no_rebuilds
        else "full_results_sweep.csv"
    )
    output_path = f"{partitions_directory}/{output_name}"
    table.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
