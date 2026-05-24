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


def read_fbin_ground_truth(filename: str) -> np.ndarray:
    """Read ground-truth neighbor IDs, returned as a (num_queries, K) int array.

    Returns np.intp dtype so the array can be used directly as an index into
    other numpy arrays without an extra cast.
    """
    with open(filename, "rb") as f:
        num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        K = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        total_elements = num_queries * K
        all_ids = np.frombuffer(f.read(total_elements * 4), dtype=np.uint32)
        if all_ids.size != total_elements:
            raise ValueError("Failed to read neighbor IDs")
        # distances follow in the file; we don't read further so no seek needed
    return all_ids.reshape((num_queries, K)).astype(np.intp)


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
    visited_mask: np.ndarray,
    gt_parts: np.ndarray,
    k: int,
) -> float:
    """Fraction of GT partition neighbours found, averaged over queries.

    Parameters
    ----------
    visited_mask : (Q, P) bool array — True where a partition was visited.
    gt_parts     : (Q, k) int array  — partition ID of each GT neighbour.
    k            : number of GT neighbours per query.
    """
    Q    = gt_parts.shape[0]
    hits = visited_mask[np.arange(Q)[:, None], gt_parts].sum(axis=1)  # (Q,)
    return float((hits / k).mean())


def _compute_activation(
    visited_mask: np.ndarray,
    num_partitions: int,
) -> float:
    """Mean fraction of partitions visited per query."""
    return float(visited_mask.sum(axis=1).mean()) / num_partitions


def _compute_per_partition_counts(
    visited_mask: np.ndarray,
) -> np.ndarray:
    """Number of queries that visited each partition."""
    return visited_mask.sum(axis=0).astype(np.int64)


# ─── Routing cache ────────────────────────────────────────────────────────────

def _build_routing_cache(
    queries:        np.ndarray,
    base_dir:       str,
    partition_dir:  str,
    hnsw_step:      int,
    partition_step: int,
    num_partitions: int,
) -> dict:
    """Load HNSW + partitions, run the unified knn_query, precompute per-mode structures.

    Separating routing setup from GT evaluation allows the no-rebuild pass to
    call this once and reuse the result across all steps — avoiding redundant
    HNSW loads and knn_query calls when hnsw_step/partition_step are constant.

    Returns a dict with keys:
      router, partitions, current_count, num_partitions, Q,
      labels_bf_all, unique_orders,
      ordered_pids_rt, cumprobs_rt, fallback_pids_rt,
      t_unified
    """
    router = hnswlib.Index(space='l2', dim=100)
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    current_count = router.get_current_count()

    partition_file = f"{partition_dir}/step_{partition_step:06d}_partitions.csv"
    partitions = np.loadtxt(partition_file, delimiter=",", dtype=int)[:-1]

    Q = len(queries)

    # ── Single unified HNSW query shared by all three routing modes ───────────
    max_bf     = max(BRANCHING_FACTOR_PARAMS)
    max_nprobe = max(NPROBE_PARAMS)
    k_rt_max   = min(50, current_count)
    k_unified  = min(max(max_bf, k_rt_max), current_count)
    router.set_ef(max(100, k_unified))
    t0 = time.perf_counter()
    labels_all, dists_all = router.knn_query(queries, k=k_unified)
    t_unified = time.perf_counter() - t0

    # Expand only if any query lacks max_nprobe unique partitions.
    k_np = k_unified
    while True:
        min_unique = min(
            len(set(int(partitions[l]) for l in ll))
            for ll in labels_all
        )
        if min_unique >= max_nprobe or k_np >= current_count:
            break
        k_np = min(k_np * 10, current_count)
        router.set_ef(max(100, k_np))
        t0 = time.perf_counter()
        labels_all, dists_all = router.knn_query(queries, k=k_np)
        t_unified = time.perf_counter() - t0

    # Derive per-mode slices (share memory, no copy).
    k_bf_actual   = min(max_bf, current_count)
    labels_bf_all = labels_all[:, :k_bf_actual]
    labels_rt_all = labels_all[:, :k_rt_max]
    dists_rt_all  = dists_all[:, :k_rt_max]

    # ── NProbe: ordered-unique partition list per query ───────────────────────
    part_seq = partitions[labels_all]   # (Q, k_np)
    unique_orders: List[List[int]] = []
    for row in part_seq:
        seen_set: set = set()
        order: List[int] = []
        for pid in map(int, row):
            if pid not in seen_set:
                seen_set.add(pid)
                order.append(pid)
        unique_orders.append(order)

    # ── RecallTarget: vectorised probability precomputation ───────────────────
    # part_size via np.bincount (no Python loop).
    part_size = np.bincount(partitions.astype(np.intp),
                             minlength=num_partitions).astype(np.float64)

    d0_rt      = dists_rt_all[:, :1] + 1e-10                      # (Q, 1)
    rel_d_rt   = dists_rt_all / d0_rt                              # (Q, k_rt)
    pids_rt    = partitions[labels_rt_all]                         # (Q, k_rt)
    weights_rt = part_size[pids_rt] * np.exp(-rel_d_rt)           # (Q, k_rt)

    # np.bincount with weights: faster scatter-add than np.add.at.
    linear_idx = (np.arange(Q)[:, None] * num_partitions + pids_rt).ravel()
    partition_probs_all = np.bincount(
        linear_idx,
        weights=weights_rt.ravel(),
        minlength=Q * num_partitions,
    ).reshape(Q, num_partitions)

    prob_sums        = partition_probs_all.sum(axis=1, keepdims=True)
    fallback_pids_rt = partitions[labels_rt_all[:, 0]]
    zero_mask        = prob_sums[:, 0] <= 0.0
    partition_probs_all[zero_mask, fallback_pids_rt[zero_mask]] = 1.0
    prob_sums[zero_mask] = 1.0
    partition_probs_all /= prob_sums

    ordered_pids_rt = np.argsort(-partition_probs_all, axis=1)         # (Q, P)
    sorted_probs_rt = np.take_along_axis(partition_probs_all,
                                          ordered_pids_rt, axis=1)
    cumprobs_rt     = np.cumsum(sorted_probs_rt, axis=1)               # (Q, P)

    return dict(
        router=router,
        partitions=partitions,
        current_count=current_count,
        num_partitions=num_partitions,
        Q=Q,
        labels_bf_all=labels_bf_all,
        unique_orders=unique_orders,
        ordered_pids_rt=ordered_pids_rt,
        cumprobs_rt=cumprobs_rt,
        fallback_pids_rt=fallback_pids_rt,
        t_unified=t_unified,
    )


def _eval_step(
    cache:       dict,
    gt_file:     str,
    base_mmap,
    k_neighbors: int,
) -> Tuple[
    Dict[Tuple[str, float], Tuple[float, float, np.ndarray, float]],
    Dict[float, Tuple[float, float]],
]:
    """Evaluate all sweep combinations for one GT file using a prebuilt routing cache.

    Parameters
    ----------
    cache       : dict returned by _build_routing_cache.
    gt_file     : path to the ground-truth .gt100 file for this step.
    base_mmap   : mmap of the base vector file (for GT vector lookup).
    k_neighbors : k for recall@k.
    """
    router           = cache["router"]
    partitions       = cache["partitions"]
    current_count    = cache["current_count"]
    num_partitions   = cache["num_partitions"]
    Q                = cache["Q"]
    labels_bf_all    = cache["labels_bf_all"]
    unique_orders    = cache["unique_orders"]
    ordered_pids_rt  = cache["ordered_pids_rt"]
    cumprobs_rt      = cache["cumprobs_rt"]
    fallback_pids_rt = cache["fallback_pids_rt"]
    t_unified        = cache["t_unified"]

    # ── Load GT and map each GT neighbour to its partition ────────────────────
    # read_fbin_ground_truth returns (Q, k) np.intp — no loop needed.
    ground_truth = read_fbin_ground_truth(gt_file)         # (Q, k)
    gt_indices   = ground_truth.ravel()                    # (Q*k,)
    vecs_gt      = np.ascontiguousarray(
                       base_mmap[gt_indices]).astype(np.float32)
    router.set_ef(100)
    labels_gt, _ = router.knn_query(vecs_gt, k=1)
    gt_part_ids  = partitions[labels_gt[:, 0]]             # (Q*k,)
    gt_parts_2d  = gt_part_ids.reshape(Q, k_neighbors)    # (Q, k)

    results: Dict[Tuple[str, float], Tuple[float, float, np.ndarray, float]] = {}

    # ── BranchingFactor sweep ─────────────────────────────────────────────────
    # Mirrors getPartitionsForSearch_Branching_: collect all unique partitions
    # among the k nearest centers.  Boolean mask replaces dict-of-sets.
    for bf in BRANCHING_FACTOR_PARAMS:
        k       = min(bf, current_count)
        pids_bf = partitions[labels_bf_all[:, :k]]         # (Q, k)
        visited_mask = np.zeros((Q, num_partitions), dtype=bool)
        visited_mask[np.arange(Q)[:, None], pids_bf] = True

        recall     = _compute_recall(visited_mask, gt_parts_2d, k_neighbors)
        activation = _compute_activation(visited_mask, num_partitions)
        ppc        = _compute_per_partition_counts(visited_mask)
        results[("BranchingFactor", float(bf))] = (recall, activation, ppc, t_unified)

    # ── NProbe sweep ──────────────────────────────────────────────────────────
    # Mirrors getPartitionsForSearch_Nprobe_: walk centers nearest-first,
    # collecting nprobe unique partitions.  unique_orders is pre-sorted.
    for nprobe in NPROBE_PARAMS:
        visited_mask = np.zeros((Q, num_partitions), dtype=bool)
        for i in range(Q):
            visited_mask[i, unique_orders[i][:nprobe]] = True

        recall     = _compute_recall(visited_mask, gt_parts_2d, k_neighbors)
        activation = _compute_activation(visited_mask, num_partitions)
        ppc        = _compute_per_partition_counts(visited_mask)
        results[("NProbe", float(nprobe))] = (recall, activation, ppc, t_unified)

    # ── RecallTarget sweep ────────────────────────────────────────────────────
    # cumprobs_rt is precomputed in the cache; only the threshold varies here.
    # np.put_along_axis scatters the rank mask into the partition-indexed mask.
    for target in TARGET_PARAMS:
        meets     = cumprobs_rt >= target                              # (Q, P)
        first_idx = np.where(meets.any(axis=1),
                             np.argmax(meets, axis=1),
                             num_partitions - 1)                       # (Q,)
        rank_mask    = np.arange(num_partitions)[None, :] <= first_idx[:, None]
        visited_mask = np.zeros((Q, num_partitions), dtype=bool)
        np.put_along_axis(visited_mask, ordered_pids_rt, rank_mask, axis=1)

        recall     = _compute_recall(visited_mask, gt_parts_2d, k_neighbors)
        activation = _compute_activation(visited_mask, num_partitions)
        ppc        = _compute_per_partition_counts(visited_mask)
        results[("RecallTarget", float(target))] = (recall, activation, ppc, t_unified)

    # ── Oracle lower bounds ───────────────────────────────────────────────────
    # query_dist[i, p] = fraction of query i's k GT neighbours in partition p.
    # Vectorised via np.bincount (replaces Python loop over queries).
    flat    = gt_parts_2d.ravel().astype(np.intp)
    row_idx = np.repeat(np.arange(Q), k_neighbors)
    linear  = row_idx * num_partitions + flat
    query_dist = np.bincount(
        linear,
        minlength=Q * num_partitions,
    ).reshape(Q, num_partitions).astype(np.float64) / k_neighbors

    sorted_dist = np.sort(query_dist, axis=1)[:, ::-1]
    cumsum      = np.cumsum(sorted_dist, axis=1)

    oracle_results: Dict[float, Tuple[float, float]] = {}
    for target in TARGET_PARAMS:
        meets = cumsum >= target
        idx   = np.where(
            meets.any(axis=1),
            np.argmax(meets, axis=1),
            num_partitions - 1,
        )
        n_parts         = idx + 1
        achieved_recall = cumsum[np.arange(Q), idx]
        oracle_results[float(target)] = (
            float(n_parts.mean()) / num_partitions,
            float(achieved_recall.mean()),
        )

    return results, oracle_results


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
    """Run all (mode, param) sweep combinations for a single runbook step.

    Thin wrapper around _build_routing_cache + _eval_step.  For repeated calls
    with the same hnsw_step/partition_step (the no-rebuild pass), call
    _build_routing_cache once and _eval_step per GT file instead.
    """
    cache = _build_routing_cache(
        queries, base_dir, partition_dir,
        hnsw_step, partition_step, num_partitions,
    )
    return _eval_step(cache, gt_file, base_mmap, k_neighbors)


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

    # No-rebuild pass — always use step_start router and partitions.
    # Build the routing cache once; only the GT file changes per step,
    # so the HNSW load and knn_query are not repeated.
    print(f"\nBuilding no-rebuild routing cache from step {step_start} ...")
    no_rebuild_cache = _build_routing_cache(
        queries, base_directory, partitions_directory,
        step_start, step_start, num_partitions,
    )
    for step in processed_steps:
        gt_test = f"{args.gt_dir}/step{step + 1}.gt100"
        if not Path(gt_test).exists():
            continue
        print(f"\nNO REBUILDING: evaluating step {step} ...")
        res, oracle = _eval_step(no_rebuild_cache, gt_test, base_mmap, k_neighbors=10)
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
