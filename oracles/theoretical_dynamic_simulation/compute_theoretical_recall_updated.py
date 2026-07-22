import numpy as np
import struct
from pathlib import Path
import re
import yaml
import argparse
import hnswlib
import time
import pandas as pd
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment


# I/O helpers

def read_fbin_slice(filename, start, end):
    """Read vectors [start, end) from an fbin file."""
    with open(filename, "rb") as f:
        header = f.read(8)
        num_points, dim = struct.unpack("II", header)
        print(num_points)
        if start < 0 or end > num_points:
            raise ValueError("Slice out of bounds")
        count = end - start
        f.seek(8 + start * dim * 4, 0)
        data = np.fromfile(f, dtype=np.float32, count=count * dim)
        return data.reshape(count, dim)


def read_fbin_ground_truth(filename: str) -> np.ndarray:
    """Return ground-truth neighbor IDs as a 2-D array (num_queries, K)."""
    with open(filename, "rb") as f:
        num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        K = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        total = num_queries * K
        all_ids = np.frombuffer(f.read(total * 4), dtype=np.uint32)
        if all_ids.size != total:
            raise ValueError("Failed to read neighbor IDs")
    return all_ids.reshape((num_queries, K)).astype(np.intp)


def mmap_fbin(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_points, dim = header
    return np.memmap(filename, dtype=np.float32, mode="r",
                     offset=8, shape=(num_points, dim))


def read_u8bin_slice(filename, start, end):
    """Read vectors [start, end) from a u8bin file, returned as float32."""
    with open(filename, "rb") as f:
        header = f.read(8)
        num_points, dim = struct.unpack("II", header)
        print(num_points)
        if start < 0 or end > num_points:
            raise ValueError("Slice out of bounds")
        count = end - start
        f.seek(8 + start * dim, 0)
        data = np.fromfile(f, dtype=np.uint8, count=count * dim)
        return data.reshape(count, dim).astype(np.float32)


def mmap_u8bin(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_points, dim = header
    return np.memmap(filename, dtype=np.uint8, mode="r",
                     offset=8, shape=(num_points, dim))


def load_runbook(path, dataset_key):
    with open(path, "r") as f:
        rb = yaml.safe_load(f)
    steps = rb[dataset_key]
    return [steps[k] for k in sorted(k for k in steps if isinstance(k, int))]


def parse_runbook(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    root = next(iter(data.values()))
    deltas = {}
    for step, payload in root.items():
        if not isinstance(step, int):
            continue
        op = payload.get("operation")
        start = payload.get("start")
        end = payload.get("end")
        if op in {"delete", "insert"} and start is not None and end is not None:
            count = end - start + 1
            deltas[step] = -count if op == "delete" else count
        else:
            deltas[step] = 0
    return deltas


def find_first_available_step(base_dir, partitions_dir):
    """Auto-detect the smallest step N for which both a router file
    (step_NNNNNN_hnsw.bin) and a partitions file exist."""
    router_re = re.compile(r"^step_(\d+)_hnsw\.bin$")
    router_steps = {int(m.group(1)) for f in Path(base_dir).iterdir()
                    if (m := router_re.match(f.name))}

    partition_re = re.compile(r"^step_(\d+)_partitions\.csv$")
    partition_steps = {int(m.group(1)) for f in Path(partitions_dir).iterdir()
                       if (m := partition_re.match(f.name))}

    common = router_steps & partition_steps
    if not common:
        raise FileNotFoundError(
            f"No step with both hnsw.bin and partitions.csv found.\n"
            f"  base_dir={base_dir}  router steps: {sorted(router_steps) or 'none'}\n"
            f"  partitions_dir={partitions_dir}  partition steps: "
            f"{sorted(partition_steps) or 'none'}"
        )
    return min(common)


# Shared metric helpers

def _cof(ppc: np.ndarray) -> float:
    """Coefficient of variation of per-partition counts."""
    m = ppc.mean()
    return float(ppc.std() / m) if m > 0 else 0.0


def _recall(visited_mask: np.ndarray, gt_parts: np.ndarray, k: int) -> float:
    """Mean per-query recall.  visited_mask: (Q, P) bool; gt_parts: (Q, k) int."""
    Q = gt_parts.shape[0]
    hits = visited_mask[np.arange(Q)[:, None], gt_parts].sum(axis=1)
    return float((hits / k).mean())


def _activation(visited_mask: np.ndarray, num_partitions: int) -> float:
    return float(visited_mask.sum(axis=1).mean()) / num_partitions


def _compute_phi(part_a: np.ndarray, part_b: np.ndarray, nblocks: int) -> float:
    """Fraction of centers whose partition changes under optimal bipartite relabeling.
    Pass [:-1] to strip trailing moved_nodes entry; result is relabeling-invariant."""
    cost = np.zeros((nblocks, nblocks), dtype=np.int64)
    np.add.at(cost, (part_a, part_b), 1)
    row_ind, col_ind = linear_sum_assignment(-cost)
    relabel = np.empty(nblocks, dtype=np.int64)
    relabel[row_ind] = col_ind
    moved = int((relabel[part_a] != part_b).sum())
    return moved / len(part_a)


# Routing functions — each returns List[(recall, activation, ppc, query_time_s)]
# one entry per param value

def compute_recall_branching_factor(
    queries: np.ndarray,
    base_mmap,
    gt_file: str,
    base_dir: str,
    partition_dir: str,
    hnsw_step: int,
    partition_step: int,
    branching_factors: List[int],
    k_neighbors: int,
    num_partitions: int,
    partitions: np.ndarray = None,
) -> List[Tuple]:
    """BranchingFactor routing: query k nearest centers, visit all unique partitions."""
    ground_truth = read_fbin_ground_truth(gt_file)      # (Q, K)
    Q = ground_truth.shape[0]

    router = hnswlib.Index(space='l2', dim=queries.shape[1])
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    max_bf = max(branching_factors)
    router.set_ef(max(100, max_bf))
    current_count = router.get_current_count()

    if partitions is None:
        partitions = np.loadtxt(
            f"{partition_dir}/step_{partition_step:06d}_partitions.csv",
            delimiter=",", dtype=int)[:-1]   # drop trailing moved_nodes entry

    # GT partition assignment
    gt_indices = ground_truth[:, :k_neighbors].ravel().astype(np.int64)
    gt_vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)
    gt_labels, _ = router.knn_query(gt_vecs, k=1)
    gt_parts = partitions[gt_labels[:, 0]].reshape(Q, k_neighbors)  # (Q, k)

    # One HNSW search at max_bf; slice per param
    k_query = min(max_bf, current_count)
    t0 = time.perf_counter()
    labels, _ = router.knn_query(queries, k=k_query)  # (Q, k_query)
    elapsed = time.perf_counter() - t0

    pids = partitions[labels]  # (Q, k_query) — partition ID per center

    results = []
    for bf in branching_factors:
        k_eff = min(bf, k_query)
        visited_mask = np.zeros((Q, num_partitions), dtype=bool)
        visited_mask[np.arange(Q)[:, None], pids[:, :k_eff]] = True

        r = _recall(visited_mask, gt_parts, k_neighbors)
        act = _activation(visited_mask, num_partitions)
        ppc = visited_mask.sum(axis=0).astype(np.int64)
        results.append((r, act, ppc, elapsed))

    return results


def compute_recall_nprobe(
    queries: np.ndarray,
    base_mmap,
    gt_file: str,
    base_dir: str,
    partition_dir: str,
    hnsw_step: int,
    partition_step: int,
    nprobe_values: List[int],
    k_neighbors: int,
    num_partitions: int,
    partitions: np.ndarray = None,
) -> List[Tuple]:
    """NProbe routing: walk centers in HNSW order until N unique partitions seen."""
    ground_truth = read_fbin_ground_truth(gt_file)      # (Q, K)
    Q = ground_truth.shape[0]

    router = hnswlib.Index(space='l2', dim=queries.shape[1])
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    max_p = max(nprobe_values)
    router.set_ef(100)
    current_count = router.get_current_count()

    if partitions is None:
        partitions = np.loadtxt(
            f"{partition_dir}/step_{partition_step:06d}_partitions.csv",
            delimiter=",", dtype=int)[:-1]   # drop trailing moved_nodes entry

    # GT partition assignment
    gt_indices = ground_truth[:, :k_neighbors].ravel().astype(np.int64)
    gt_vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)
    gt_labels, _ = router.knn_query(gt_vecs, k=1)
    gt_parts = partitions[gt_labels[:, 0]].reshape(Q, k_neighbors)  # (Q, k)

    # Find k ensuring every query sees >= max_p unique partitions
    k = max_p * 1000
    final_elapsed = 0.0
    niter = 0
    while True:
        niter += 1
        k_capped = min(k, current_count)
        print(f"Iteration {niter}: querying k={k_capped} centers...")
        t0 = time.perf_counter()
        labels, _ = router.knn_query(queries, k=k_capped)  # (Q, k_capped)
        final_elapsed = time.perf_counter() - t0
        pids_all = partitions[labels]  # (Q, k_capped)
        # Check: every query has >= max_p unique partitions?
        min_unique = min(len(np.unique(row)) for row in pids_all)
        if min_unique >= max_p or k_capped >= current_count:
            break
        k *= 10
    print(f"NProbe: k={k_capped}, min_unique={min_unique}, iters={niter}, "
          f"time={final_elapsed:.4f}s")

    # Per-query: unique partition order (first-occurrence)
    unique_orders = []
    for i in range(Q):
        seen = set()
        order = []
        for pid in pids_all[i]:
            if pid not in seen:
                seen.add(pid)
                order.append(pid)
        unique_orders.append(np.array(order, dtype=np.int32))

    results = []
    for nprobe in nprobe_values:
        visited_mask = np.zeros((Q, num_partitions), dtype=bool)
        for i in range(Q):
            selected = unique_orders[i][:nprobe]
            visited_mask[i, selected] = True

        r = _recall(visited_mask, gt_parts, k_neighbors)
        act = _activation(visited_mask, num_partitions)
        ppc = visited_mask.sum(axis=0).astype(np.int64)
        results.append((r, act, ppc, final_elapsed))

    return results


def compute_recall_recall_target(
    queries: np.ndarray,
    base_mmap,
    gt_file: str,
    base_dir: str,
    partition_dir: str,
    hnsw_step: int,
    partition_step: int,
    targets: List[float],
    k_neighbors: int,
    num_partitions: int,
    k_rt: int = 50,
    partitions: np.ndarray = None,
    per_query_out: dict = None,
) -> List[Tuple]:
    """RecallTarget routing: score partitions by size * exp(-d/d0), greedily
    accumulate until cumulative probability >= target."""
    ground_truth = read_fbin_ground_truth(gt_file)      # (Q, K)
    Q = ground_truth.shape[0]

    router = hnswlib.Index(space='l2', dim=queries.shape[1])
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    router.set_ef(max(100, k_rt))

    if partitions is None:
        partitions = np.loadtxt(
            f"{partition_dir}/step_{partition_step:06d}_partitions.csv",
            delimiter=",", dtype=int)[:-1]   # drop trailing moved_nodes entry
    part_size = np.bincount(partitions, minlength=num_partitions).astype(np.float32)

    # GT partition assignment
    gt_indices = ground_truth[:, :k_neighbors].ravel().astype(np.int64)
    gt_vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)
    gt_labels, _ = router.knn_query(gt_vecs, k=1)
    gt_parts = partitions[gt_labels[:, 0]].reshape(Q, k_neighbors)  # (Q, k)

    # Query k_rt nearest centers
    k_capped = min(k_rt, router.get_current_count())
    t0 = time.perf_counter()
    labels, distances = router.knn_query(queries, k=k_capped)   # (Q, k_rt)
    elapsed = time.perf_counter() - t0

    pids = partitions[labels]  # (Q, k_rt)

    # Score: part_size * exp(-d/d0) where d0 = nearest-center distance
    d0 = distances[:, :1] + 1e-10                                 # (Q, 1)
    weights = part_size[pids] * np.exp(-distances / d0)           # (Q, k_rt)

    # Scatter weights into per-partition scores
    linear_idx = (np.arange(Q)[:, None] * num_partitions + pids).ravel()
    partition_scores = np.bincount(
        linear_idx, weights=weights.ravel(), minlength=Q * num_partitions
    ).reshape(Q, num_partitions)

    # Normalize to probabilities
    row_sums = partition_scores.sum(axis=1, keepdims=True).clip(min=1e-10)
    partition_probs = partition_scores / row_sums                  # (Q, P)

    # Sort by descending probability; cumulative sum
    ordered_pids = np.argsort(-partition_probs, axis=1).astype(np.int32)  # (Q, P)
    sorted_probs = np.take_along_axis(partition_probs, ordered_pids, axis=1)
    cumprobs = np.cumsum(sorted_probs, axis=1)                    # (Q, P)

    # Heuristic weight per partition (sum of part_size*exp(-d/d_min)); format matches oracle
    mean_probs        = partition_probs.mean(axis=0)              # (P,)
    mean_ranking      = np.argsort(-mean_probs).astype(np.int32)  # (P,)
    mean_ranked_probs = mean_probs[mean_ranking]                  # (P,)
    ranking_list = mean_ranking.tolist()
    probs_list   = [float(p) for p in mean_ranked_probs]

    # Optional sidecar: per-query ordering + probs for 1:1 comparison with oracle
    if per_query_out is not None:
        per_query_out["ordering"]     = ordered_pids   # (Q, P) int32
        per_query_out["sorted_probs"] = sorted_probs   # (Q, P) float64

    results = []
    for target in targets:
        reached = cumprobs >= target
        # First index where cumprob >= target; argmax fallback to last partition
        first_idx = np.where(reached.any(axis=1),
                             reached.argmax(axis=1),
                             num_partitions - 1)                   # (Q,)

        # Visited mask: top (first_idx+1) partitions per query
        visited_mask = np.zeros((Q, num_partitions), dtype=bool)
        for i in range(Q):
            n = int(first_idx[i]) + 1
            visited_mask[i, ordered_pids[i, :n]] = True

        r = _recall(visited_mask, gt_parts, k_neighbors)
        act = _activation(visited_mask, num_partitions)
        ppc = visited_mask.sum(axis=0).astype(np.int64)
        results.append((r, act, ppc, elapsed, ranking_list, probs_list))

    return results


def compute_recall_recall_target_extended(
    queries: np.ndarray,
    base_mmap,
    gt_file: str,
    base_dir: str,
    partition_dir: str,
    hnsw_step: int,
    partition_step: int,
    targets: List[float],
    k_neighbors: int,
    num_partitions: int,
    k_rt: int = 50,
    k_kl: int = 3,
    partitions: np.ndarray = None,
    per_query_out: dict = None,
) -> List[Tuple]:
    """RecallTargetExtended routing: Kozachenko-Leonenko density estimator per partition.
    Computes density(q,p) ∝ k_p / d_{k_p,p}(q)**dim for top k_rt centers per query.
    Softmax-normalized; improvements over RecallTarget: no part_size bias, top-k_kl cap,
    volume-normalized density. Does not calibrate cumulative scores to true recall.
    """
    ground_truth = read_fbin_ground_truth(gt_file)
    Q = ground_truth.shape[0]

    router = hnswlib.Index(space='l2', dim=queries.shape[1])
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    router.set_ef(max(100, k_rt))

    if partitions is None:
        partitions = np.loadtxt(
            f"{partition_dir}/step_{partition_step:06d}_partitions.csv",
            delimiter=",", dtype=int)[:-1]   # drop trailing moved_nodes entry

    # GT partition assignment via routing HNSW
    gt_indices = ground_truth[:, :k_neighbors].ravel().astype(np.int64)
    gt_vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)
    gt_labels, _ = router.knn_query(gt_vecs, k=1)
    gt_parts = partitions[gt_labels[:, 0]].reshape(Q, k_neighbors)

    # Top-k_rt nearest centers per query
    k_capped = min(k_rt, router.get_current_count())
    t0 = time.perf_counter()
    labels, distances = router.knn_query(queries, k=k_capped)
    elapsed = time.perf_counter() - t0

    # hnswlib returns squared distances; sqrt so dim-exponent treats as proper L2
    distances = np.sqrt(np.maximum(distances, 0.0)).astype(np.float64)

    pids = partitions[labels]                                # (Q, k_rt)
    dim = int(queries.shape[1])

    # Find k_kl-th nearest p-center per (q,p); cap at m_p if fewer p-centers in top-k_rt
    d_kth_per_p = np.full((Q, num_partitions), np.inf, dtype=np.float64)
    count_per_p = np.zeros((Q, num_partitions), dtype=np.int32)

    for p in range(num_partitions):
        masked = np.where(pids == p, distances, np.inf)      # (Q, k_rt)
        sorted_masked = np.sort(masked, axis=1)
        count_p = np.sum(np.isfinite(masked), axis=1)
        count_per_p[:, p] = count_p
        # k_kl-th nearest entry (or m_p-th if fewer exist)
        k_idx = np.clip(np.minimum(count_p - 1, k_kl - 1), 0, k_capped - 1)
        has_centers = count_p > 0
        if has_centers.any():
            rows = np.where(has_centers)[0]
            d_kth_per_p[rows, p] = sorted_masked[rows, k_idx[rows]]

    # Effective k: capped at actual count per (q,p)
    k_used = np.minimum(count_per_p, k_kl).astype(np.float64)
    # Clip distances to avoid log(0)
    d_kth_clipped = np.clip(d_kth_per_p, 1e-10, None)

    # Kozachenko-Leonenko log-density; no centers in top-k_rt → log(-inf) → excluded from softmax
    with np.errstate(divide='ignore', invalid='ignore'):
        log_density = np.where(
            count_per_p > 0,
            np.log(k_used) - dim * np.log(d_kth_clipped),
            -np.inf,
        )

    # Softmax with stable per-query max subtraction
    finite_log = np.where(np.isfinite(log_density), log_density, -np.inf)
    log_density_max = np.max(finite_log, axis=1, keepdims=True)
    log_density_max = np.where(np.isfinite(log_density_max), log_density_max, 0.0)
    shifted = log_density - log_density_max
    density_unnorm = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
    norm = density_unnorm.sum(axis=1, keepdims=True)
    norm = np.where(norm > 0.0, norm, 1.0)
    partition_probs = density_unnorm / norm                  # (Q, P)

    # Sort by descending probability; cumsum drives stopping rule (same shape as RecallTarget)
    ordered_pids = np.argsort(-partition_probs, axis=1).astype(np.int32)
    sorted_probs = np.take_along_axis(partition_probs, ordered_pids, axis=1)
    cumprobs = np.cumsum(sorted_probs, axis=1)

    # Per-step ranking (same convention as RecallTarget/Oracle)
    mean_probs        = partition_probs.mean(axis=0)
    mean_ranking      = np.argsort(-mean_probs).astype(np.int32)
    mean_ranked_probs = mean_probs[mean_ranking]
    ranking_list = mean_ranking.tolist()
    probs_list   = [float(p) for p in mean_ranked_probs]

    # Optional sidecar matching RecallTarget format for row-by-row comparison
    if per_query_out is not None:
        per_query_out["ordering"] = ordered_pids
        per_query_out["sorted_probs"] = sorted_probs

    results = []
    for target in targets:
        reached = cumprobs >= target
        first_idx = np.where(reached.any(axis=1),
                             reached.argmax(axis=1),
                             num_partitions - 1)
        visited_mask = np.zeros((Q, num_partitions), dtype=bool)
        for i in range(Q):
            n = int(first_idx[i]) + 1
            visited_mask[i, ordered_pids[i, :n]] = True

        r = _recall(visited_mask, gt_parts, k_neighbors)
        act = _activation(visited_mask, num_partitions)
        ppc = visited_mask.sum(axis=0).astype(np.int64)
        results.append((r, act, ppc, elapsed, ranking_list, probs_list))

    return results


# Oracle

BRANCHING_FACTOR_PARAMS: List[int]   = [1, 2, 5, 10, 20, 40, 80]
NPROBE_PARAMS:           List[int]   = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TARGET_PARAMS:           List[float] = [0.60, 0.70, 0.75, 0.80, 0.85,
                                        0.90, 0.95, 0.97, 0.98, 0.99]

DEFAULT_PARAMS = {
    "BranchingFactor": BRANCHING_FACTOR_PARAMS,
    "NProbe":          NPROBE_PARAMS,
    "RecallTarget":    TARGET_PARAMS,
    "RecallTargetExtended": TARGET_PARAMS,
}

ORACLE_TARGETS = TARGET_PARAMS


def _compute_oracle(
    queries: np.ndarray,
    base_mmap,
    gt_file: str,
    base_dir: str,
    partition_dir: str,
    hnsw_step: int,
    partition_step: int,
    k_neighbors: int,
    num_partitions: int,
    partitions: np.ndarray = None,
    per_query_out: dict = None,
) -> List[Tuple]:
    """Oracle: minimum partitions per target recall via greedy accumulation of
    optimal per-query partition selections. Returns List[(target, activation, recall, ranking, probs)]."""
    ground_truth = read_fbin_ground_truth(gt_file)
    Q = ground_truth.shape[0]

    router = hnswlib.Index(space='l2', dim=queries.shape[1])
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    router.set_ef(100)

    if partitions is None:
        partitions = np.loadtxt(
            f"{partition_dir}/step_{partition_step:06d}_partitions.csv",
            delimiter=",", dtype=int)[:-1]   # drop trailing moved_nodes entry

    # Map each GT neighbour to its partition via the routing HNSW
    gt_indices = ground_truth[:, :k_neighbors].ravel().astype(np.int64)
    gt_vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)
    gt_labels, _ = router.knn_query(gt_vecs, k=1)
    gt_part_ids = partitions[gt_labels[:, 0]]
    gt_parts_2d = gt_part_ids.reshape(Q, k_neighbors)   # (Q, k)

    # query_dist[i, p] = fraction of query i's GT neighbours in partition p
    flat    = gt_parts_2d.ravel().astype(np.intp)
    row_idx = np.repeat(np.arange(Q), k_neighbors)
    linear  = row_idx * num_partitions + flat
    query_dist = np.bincount(
        linear, minlength=Q * num_partitions,
    ).reshape(Q, num_partitions).astype(np.float64) / k_neighbors

    # Per-query: partition labels in descending density + corresponding sorted densities
    per_query_ordering    = np.argsort(-query_dist, axis=1).astype(np.int32)
    per_query_sorted_dist = np.take_along_axis(query_dist, per_query_ordering, axis=1)
    # Cumsum drives activation/recall reporting
    sorted_dist = per_query_sorted_dist
    cumsum      = np.cumsum(sorted_dist, axis=1)

    # Mean partition weight across queries (average GT fraction per query); comparable to RT
    mean_density        = query_dist.mean(axis=0)                       # (P,)
    mean_ranking        = np.argsort(-mean_density).astype(np.int32)    # (P,)
    mean_ranked_density = mean_density[mean_ranking]                    # (P,)
    ranking_list = mean_ranking.tolist()
    probs_list   = [float(p) for p in mean_ranked_density]

    # Optional sidecar: per-query ordering + sorted densities (matching RecallTarget format)
    if per_query_out is not None:
        per_query_out["ordering"]     = per_query_ordering      # (Q, P) int32
        per_query_out["sorted_probs"] = per_query_sorted_dist   # (Q, P) float64

    results = []
    for target in ORACLE_TARGETS:
        meets = cumsum >= target
        idx   = np.where(meets.any(axis=1), np.argmax(meets, axis=1), num_partitions - 1)
        results.append((
            float(target),
            float((idx + 1).mean()) / num_partitions,   # activation
            float(cumsum[np.arange(Q), idx].mean()),     # achieved recall
            ranking_list,
            probs_list,
        ))
    return results


def _call_oracle(queries, base_mmap, gt_file,
                 base_dir, partition_dir, hnsw_step, partition_step,
                 k_neighbors, num_partitions,
                 partitions: np.ndarray = None,
                 per_query_out: dict = None):
    try:
        return _compute_oracle(queries, base_mmap, gt_file,
                               base_dir, partition_dir, hnsw_step, partition_step,
                               k_neighbors, num_partitions,
                               partitions=partitions,
                               per_query_out=per_query_out)
    except RuntimeError as e:
        print(f"WARNING: skipping oracle (hnsw_step={hnsw_step}): {e}")
        return None


def _compute_nprobe_oracle(
    queries: np.ndarray,
    base_mmap,
    gt_file: str,
    base_dir: str,
    partition_dir: str,
    hnsw_step: int,
    partition_step: int,
    k_neighbors: int,
    num_partitions: int,
    nprobe_values: List[int],
    partitions: np.ndarray = None,
) -> List[Tuple]:
    """NProbe oracle: for each nprobe, probe the nprobe partitions with max GT
    neighbours per query and report theoretical recall. Returns List[(nprobe, activation, recall)]."""
    ground_truth = read_fbin_ground_truth(gt_file)
    Q = ground_truth.shape[0]

    router = hnswlib.Index(space='l2', dim=queries.shape[1])
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    router.set_ef(100)

    if partitions is None:
        partitions = np.loadtxt(
            f"{partition_dir}/step_{partition_step:06d}_partitions.csv",
            delimiter=",", dtype=int)[:-1]   # drop trailing moved_nodes entry

    # Map GT neighbours to partitions via routing HNSW; identical convention to all modes
    gt_indices = ground_truth[:, :k_neighbors].ravel().astype(np.int64)
    gt_vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)
    gt_labels, _ = router.knn_query(gt_vecs, k=1)
    gt_part_ids = partitions[gt_labels[:, 0]]
    gt_parts_2d = gt_part_ids.reshape(Q, k_neighbors)   # (Q, k)

    # Compute per-query partition distributions
    flat    = gt_parts_2d.ravel().astype(np.intp)
    row_idx = np.repeat(np.arange(Q), k_neighbors)
    linear  = row_idx * num_partitions + flat
    query_dist = np.bincount(
        linear, minlength=Q * num_partitions,
    ).reshape(Q, num_partitions).astype(np.float64) / k_neighbors

    # Optimal recall curves: sort desc; cumsum = max recall for top-n partitions
    sorted_dist = np.sort(query_dist, axis=1)[:, ::-1]   # (Q, P) descending
    cumsum      = np.cumsum(sorted_dist, axis=1)          # (Q, P)

    results = []
    for nprobe in nprobe_values:
        n          = min(nprobe, num_partitions)
        recall     = float(cumsum[:, n - 1].mean())
        activation = n / num_partitions
        results.append((nprobe, activation, recall))
    return results


def _call_nprobe_oracle(queries, base_mmap, gt_file,
                        base_dir, partition_dir, hnsw_step, partition_step,
                        k_neighbors, num_partitions, nprobe_values,
                        partitions: np.ndarray = None):
    try:
        return _compute_nprobe_oracle(queries, base_mmap, gt_file,
                                      base_dir, partition_dir,
                                      hnsw_step, partition_step,
                                      k_neighbors, num_partitions, nprobe_values,
                                      partitions=partitions)
    except RuntimeError as e:
        print(f"WARNING: skipping NProbe oracle (hnsw_step={hnsw_step}): {e}")
        return None


# Routing dispatch

ROUTING_FUNCS = {
    "BranchingFactor": compute_recall_branching_factor,
    "NProbe": compute_recall_nprobe,
    "RecallTarget": compute_recall_recall_target,
    "RecallTargetExtended": compute_recall_recall_target_extended,
}

# Modes with per-step ranking/probs that trigger oracle comparison + per-query sidecar
RANKED_MODES = {"RecallTarget", "RecallTargetExtended"}


def _call_routing(mode, queries, base_mmap, gt_file,
                  base_dir, partition_dir, hnsw_step, partition_step,
                  params, k_neighbors, num_partitions,
                  partitions: np.ndarray = None,
                  per_query_out: dict = None):
    fn = ROUTING_FUNCS[mode]
    if mode in ("BranchingFactor", "NProbe"):
        params_cast = [int(p) for p in params]
    else:
        params_cast = [float(p) for p in params]
    # per_query_out: only used by ranked modes; others don't accept the kwarg
    extra_kwargs = {}
    if mode in RANKED_MODES and per_query_out is not None:
        extra_kwargs["per_query_out"] = per_query_out
    try:
        return fn(queries, base_mmap, gt_file,
                  base_dir, partition_dir, hnsw_step, partition_step,
                  params_cast, k_neighbors, num_partitions,
                  partitions=partitions,
                  **extra_kwargs)
    except RuntimeError as e:
        print(f"WARNING: skipping step (hnsw_step={hnsw_step}, "
              f"partition_step={partition_step}): {e}")
        return None


# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute theoretical recall for a given routing mode."
    )
    parser.add_argument("--base-dir", required=True,
                        help="Directory containing step_NNNNNN_hnsw.bin files")
    parser.add_argument("--partitions-dir", required=True,
                        help="Directory containing step_NNNNNN_partitions.csv files")
    parser.add_argument("--mode", required=True,
                        choices=["BranchingFactor", "NProbe",
                                 "RecallTarget", "RecallTargetExtended"],
                        help="Routing mode. RecallTargetExtended: kNN-density scorer per partition.")
    parser.add_argument("--params", nargs="+", type=float, default=None,
                        help="Parameter values (ints for BF/NProbe, floats 0-1 for RT); "
                             "defaults to full sweep grid.")
    parser.add_argument("--no-rebuilds", action="store_true",
                        help="Evaluate only the no-rebuild path "
                             "(always use first-step router/partitions)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold τ ∈ [0,1]; rebuild when φ>τ. Mutually exclusive with --no-rebuilds.")
    parser.add_argument("--runbook-path", required=True,
                        help="Path to the runbook YAML file")
    parser.add_argument("--query-file", required=True,
                        help="Path to the query file (fbin or u8bin format)")
    parser.add_argument("--base-file", required=True,
                        help="Path to the base vectors file")
    parser.add_argument("--gt-dir", required=True,
                        help="Directory containing stepN.gt100 ground-truth files")
    parser.add_argument("--per-query-rankings", action="store_true",
                        help="Dump per-query rankings from routing and Oracle every N steps.")
    parser.add_argument("--per-query-every", type=int, default=100,
                        help="Dump interval (in steps) for per-query rankings (default: 100).")
    return parser.parse_args()


# Entry point

if __name__ == "__main__":
    args = parse_args()

    if args.no_rebuilds and args.threshold is not None:
        raise ValueError("--no-rebuilds and --threshold are mutually exclusive")

    mode = args.mode
    params = args.params if args.params is not None else DEFAULT_PARAMS[mode]

    deltas = parse_runbook(args.runbook_path)
    runbook_steps = sorted(deltas)

    if args.base_file.endswith(".u8bin"):
        queries = read_u8bin_slice(args.query_file, 0, 10000)
        base_mmap = mmap_u8bin(args.base_file)
    else:
        queries = read_fbin_slice(args.query_file, 0, 10000)
        base_mmap = mmap_fbin(args.base_file)

    base_directory = args.base_dir
    partitions_directory = args.partitions_dir

    step_start = find_first_available_step(base_directory, partitions_directory)
    print(f"Starting step (auto-detected): {step_start}")

    # Detect step_end for rebuild mode (not needed for threshold/no-rebuild)
    step_end = step_start
    if not args.no_rebuilds and args.threshold is None:
        while Path(f"{base_directory}/step_{step_end:06d}_hnsw.bin").exists():
            step_end += 2

    first_partitions = np.loadtxt(
        f"{partitions_directory}/step_{step_start:06d}_partitions.csv",
        delimiter=",", dtype=int)[:-1]   # drop trailing moved_nodes entry
    num_partitions = int(first_partitions.max()) + 1
    print(f"Found {num_partitions} partitions")

    if args.threshold is not None:
        tau_str = f"{round(args.threshold * 100)}"
        suffix = f"_t{tau_str}"
    elif args.no_rebuilds:
        suffix = "_no_rebuilds"
    else:
        suffix = ""

    per_query_path = None
    dump_every     = max(1, int(args.per_query_every))
    if args.per_query_rankings and mode in RANKED_MODES:
        # Include mode in filename so RecallTarget and RecallTargetExtended don't clobber
        per_query_path = (
            f"{partitions_directory}/per_query_rankings_{mode}{suffix}.csv"
        )
        with open(per_query_path, "w") as _pq:
            _pq.write("step,query_id,rt_ranking,rt_scores,"
                      "oracle_ranking,oracle_scores\n")
        print(f"Per-query rankings will be dumped to {per_query_path} "
              f"every {dump_every} processed steps.")

    def _append_per_query(step_value, rt_data, oracle_data):
        """Append per-query rows for one step (rt_data and oracle_data from routing/oracle)."""
        if per_query_path is None:
            return
        if not rt_data or not oracle_data:
            return
        rt_ord, rt_sc = rt_data["ordering"], rt_data["sorted_probs"]
        or_ord, or_sc = oracle_data["ordering"], oracle_data["sorted_probs"]
        Q_local = rt_ord.shape[0]
        with open(per_query_path, "a") as _pq:
            for q in range(Q_local):
                rt_r = rt_ord[q].tolist()
                rt_s = [round(float(p), 6) for p in rt_sc[q]]
                or_r = or_ord[q].tolist()
                or_s = [round(float(p), 6) for p in or_sc[q]]
                # Wrap list reprs in double quotes to preserve embedded commas in CSV
                _pq.write(
                    f'{step_value},{q},'
                    f'"{rt_r}","{rt_s}","{or_r}","{or_s}"\n'
                )

    rows = []

    # Rebuild pass
    if not args.no_rebuilds and args.threshold is None:
        pass_iter = -1
        for step in range(step_start, step_end, 2):
            pass_iter += 1
            gt_test = f"{args.gt_dir}/step{step + 1}.gt100"
            hnsw_file = f"{base_directory}/step_{step:06d}_hnsw.bin"
            partition_file = f"{partitions_directory}/step_{step:06d}_partitions.csv"
            if not all(Path(f).exists() for f in [gt_test, hnsw_file, partition_file]):
                missing = [f for f in [gt_test, hnsw_file, partition_file]
                           if not Path(f).exists()]
                print(f"Step {step}: missing {missing} — stopping rebuild pass.")
                break

            dump_now = (per_query_path is not None
                        and mode in RANKED_MODES
                        and pass_iter % dump_every == 0)
            rt_pq     = {} if dump_now else None
            oracle_pq = {} if dump_now else None

            step_results = _call_routing(
                mode, queries, base_mmap, gt_test,
                base_directory, partitions_directory,
                step, step, params, 10, num_partitions,
                per_query_out=rt_pq)

            if step_results is None:
                continue
            for param, result in zip(params, step_results):
                # RecallTarget tuples carry two extra fields (ranking + probs)
                r, act, ppc, t = result[:4]
                row = {
                    "step": step,
                    "rebuild": "rebuild",
                    "mode": mode,
                    "param": param,
                    "recall": r,
                    "activation": act,
                    "cof": _cof(ppc),
                    "query_time_s": t,
                }
                if mode in RANKED_MODES:
                    _, _, _, _, ranking, probs = result
                    row["partition_ranking"] = str(ranking)
                    row["partition_probs"]   = str([round(p, 6) for p in probs])
                rows.append(row)
            if mode in RANKED_MODES:
                oracle_results = _call_oracle(
                    queries, base_mmap, gt_test,
                    base_directory, partitions_directory,
                    step, step, 10, num_partitions,
                    per_query_out=oracle_pq)
                if oracle_results is not None:
                    for target, act, rec, ranking, probs in oracle_results:
                        rows.append({
                            "step": step,
                            "rebuild": "rebuild",
                            "mode": "Oracle",
                            "param": target,
                            "recall": rec,
                            "activation": act,
                            "cof": float("nan"),
                            "query_time_s": float("nan"),
                            "partition_ranking": str(ranking),
                            "partition_probs":   str([round(p, 6) for p in probs]),
                        })
            if mode == "NProbe":
                nprobe_oracle_results = _call_nprobe_oracle(
                    queries, base_mmap, gt_test,
                    base_directory, partitions_directory,
                    step, step, 10, num_partitions,
                    [int(p) for p in params])
                if nprobe_oracle_results is not None:
                    for nprobe, act, rec in nprobe_oracle_results:
                        rows.append({
                            "step": step,
                            "rebuild": "rebuild",
                            "mode": "NProbeOracle",
                            "param": nprobe,
                            "recall": rec,
                            "activation": act,
                            "cof": float("nan"),
                            "query_time_s": float("nan"),
                        })
            if dump_now:
                _append_per_query(step, rt_pq, oracle_pq)
                print(f"  [per-query] dumped step {step} "
                      f"({rt_pq['ordering'].shape[0]} queries) to "
                      f"{per_query_path}")
            print(f"REBUILD step {step}: "
                  f"recall={[r[0] for r in step_results]}")

    # No-rebuild pass  (always use step_start router/partitions)
    no_rebuild_steps = [
        s for s in runbook_steps
        if s >= step_start and (s - step_start) % 2 == 0
    ]

    if args.threshold is None:
        pass_iter = -1
        for step in no_rebuild_steps:
            pass_iter += 1
            gt_test = f"{args.gt_dir}/step{step + 1}.gt100"
            if not Path(gt_test).exists():
                print(f"Missing GT at step {step}: {gt_test}. Skipping.")
                continue

            dump_now = (per_query_path is not None
                        and mode in RANKED_MODES
                        and pass_iter % dump_every == 0)
            rt_pq     = {} if dump_now else None
            oracle_pq = {} if dump_now else None

            step_results = _call_routing(
                mode, queries, base_mmap, gt_test,
                base_directory, partitions_directory,
                step_start, step_start, params, 10, num_partitions,
                per_query_out=rt_pq)

            if step_results is None:
                continue
            for param, result in zip(params, step_results):
                # RecallTarget tuples carry two extra fields (ranking + probs)
                r, act, ppc, t = result[:4]
                row = {
                    "step": step,
                    "rebuild": "no_rebuild",
                    "mode": mode,
                    "param": param,
                    "recall": r,
                    "activation": act,
                    "cof": _cof(ppc),
                    "query_time_s": t,
                }
                if mode in RANKED_MODES:
                    _, _, _, _, ranking, probs = result
                    row["partition_ranking"] = str(ranking)
                    row["partition_probs"]   = str([round(p, 6) for p in probs])
                rows.append(row)
            if mode in RANKED_MODES:
                oracle_results = _call_oracle(
                    queries, base_mmap, gt_test,
                    base_directory, partitions_directory,
                    step_start, step_start, 10, num_partitions,
                    per_query_out=oracle_pq)
                if oracle_results is not None:
                    for target, act, rec, ranking, probs in oracle_results:
                        rows.append({
                            "step": step,
                            "rebuild": "no_rebuild",
                            "mode": "Oracle",
                            "param": target,
                            "recall": rec,
                            "activation": act,
                            "cof": float("nan"),
                            "query_time_s": float("nan"),
                            "partition_ranking": str(ranking),
                            "partition_probs":   str([round(p, 6) for p in probs]),
                        })
            if mode == "NProbe":
                nprobe_oracle_results = _call_nprobe_oracle(
                    queries, base_mmap, gt_test,
                    base_directory, partitions_directory,
                    step_start, step_start, 10, num_partitions,
                    [int(p) for p in params])
                if nprobe_oracle_results is not None:
                    for nprobe, act, rec in nprobe_oracle_results:
                        rows.append({
                            "step": step,
                            "rebuild": "no_rebuild",
                            "mode": "NProbeOracle",
                            "param": nprobe,
                            "recall": rec,
                            "activation": act,
                            "cof": float("nan"),
                            "query_time_s": float("nan"),
                        })
            if dump_now:
                _append_per_query(step, rt_pq, oracle_pq)
                print(f"  [per-query] dumped step {step} "
                      f"({rt_pq['ordering'].shape[0]} queries) to "
                      f"{per_query_path}")
            print(f"NO-REBUILD step {step}: "
                  f"recall={[r[0] for r in step_results]}")

    # Threshold pass: at each step s, compare partition against last rebuild using optimal
    # bipartite matching. φ = fraction of centers whose group changes under that matching.
    # If φ > τ, rebuild triggered (s becomes active); else active step stays fixed.
    # Bipartite matching finds optimal alignment regardless of prior relabeling.
    if args.threshold is not None:
        tau = args.threshold
        last_rebuild = step_start
        active_partition = np.loadtxt(
            f"{partitions_directory}/step_{step_start:06d}_partitions.csv",
            delimiter=",", dtype=int)[:-1]   # drop trailing moved_nodes entry

        pass_iter = -1
        for step in no_rebuild_steps:
            pass_iter += 1
            gt_test = f"{args.gt_dir}/step{step + 1}.gt100"
            hnsw_file = f"{base_directory}/step_{step:06d}_hnsw.bin"
            partition_file = f"{partitions_directory}/step_{step:06d}_partitions.csv"

            if not all(Path(f).exists() for f in [gt_test, hnsw_file, partition_file]):
                missing = [f for f in [gt_test, hnsw_file, partition_file]
                           if not Path(f).exists()]
                print(f"Step {step}: missing {missing} — stopping threshold pass.")
                break

            part_s = np.loadtxt(partition_file, delimiter=",", dtype=int)[:-1]   # drop trailing moved_nodes entry
            phi = _compute_phi(part_s, active_partition, num_partitions)
            did_rebuild = phi > tau
            if did_rebuild:
                last_rebuild = step
                active_partition = part_s

            print(f"THRESHOLD(τ={tau}) step {step}: φ={phi:.4f} "
                  f"{'→ REBUILD' if did_rebuild else '  (skip)  '} "
                  f"active={last_rebuild}")

            dump_now = (per_query_path is not None
                        and mode in RANKED_MODES
                        and pass_iter % dump_every == 0)
            rt_pq     = {} if dump_now else None
            oracle_pq = {} if dump_now else None

            step_results = _call_routing(
                mode, queries, base_mmap, gt_test,
                base_directory, partitions_directory,
                last_rebuild, last_rebuild, params, 10, num_partitions,
                partitions=active_partition,
                per_query_out=rt_pq)

            if step_results is None:
                continue
            for param, result in zip(params, step_results):
                # RecallTarget tuples carry two extra fields (ranking + probs)
                r, act, ppc, t = result[:4]
                row = {
                    "step": step,
                    "rebuild": f"threshold_{tau:.4g}",
                    "mode": mode,
                    "param": param,
                    "recall": r,
                    "activation": act,
                    "cof": _cof(ppc),
                    "query_time_s": t,
                    "phi": phi,
                    "did_rebuild": did_rebuild,
                    "active_step": last_rebuild,
                }
                if mode in RANKED_MODES:
                    _, _, _, _, ranking, probs = result
                    row["partition_ranking"] = str(ranking)
                    row["partition_probs"]   = str([round(p, 6) for p in probs])
                rows.append(row)
            if mode in RANKED_MODES:
                oracle_results = _call_oracle(
                    queries, base_mmap, gt_test,
                    base_directory, partitions_directory,
                    last_rebuild, last_rebuild, 10, num_partitions,
                    partitions=active_partition,
                    per_query_out=oracle_pq)
                if oracle_results is not None:
                    for target, act, rec, ranking, probs in oracle_results:
                        rows.append({
                            "step": step,
                            "rebuild": f"threshold_{tau:.4g}",
                            "mode": "Oracle",
                            "param": target,
                            "recall": rec,
                            "activation": act,
                            "cof": float("nan"),
                            "query_time_s": float("nan"),
                            "phi": phi,
                            "did_rebuild": did_rebuild,
                            "active_step": last_rebuild,
                            "partition_ranking": str(ranking),
                            "partition_probs":   str([round(p, 6) for p in probs]),
                        })
            if mode == "NProbe":
                nprobe_oracle_results = _call_nprobe_oracle(
                    queries, base_mmap, gt_test,
                    base_directory, partitions_directory,
                    last_rebuild, last_rebuild, 10, num_partitions,
                    [int(p) for p in params],
                    partitions=active_partition)
                if nprobe_oracle_results is not None:
                    for nprobe, act, rec in nprobe_oracle_results:
                        rows.append({
                            "step": step,
                            "rebuild": f"threshold_{tau:.4g}",
                            "mode": "NProbeOracle",
                            "param": nprobe,
                            "recall": rec,
                            "activation": act,
                            "cof": float("nan"),
                            "query_time_s": float("nan"),
                            "phi": phi,
                            "did_rebuild": did_rebuild,
                            "active_step": last_rebuild,
                        })
            if dump_now:
                _append_per_query(step, rt_pq, oracle_pq)
                print(f"  [per-query] dumped step {step} "
                      f"({rt_pq['ordering'].shape[0]} queries) to "
                      f"{per_query_path}")
            print(f"THRESHOLD step {step}: "
                  f"recall={[r[0] for r in step_results]}")

    # Save results
    # partition_ranking/partition_probs: on RT/Oracle rows (directly comparable); blank for BF/NProbe
    if args.threshold is not None:
        columns = ["step", "rebuild", "mode", "param",
                   "recall", "activation", "cof", "query_time_s",
                   "phi", "did_rebuild", "active_step",
                   "partition_ranking", "partition_probs"]
    else:
        columns = ["step", "rebuild", "mode", "param",
                   "recall", "activation", "cof", "query_time_s",
                   "partition_ranking", "partition_probs"]

    df = pd.DataFrame(rows, columns=columns)
    output_name = f"full_results_{mode}{suffix}.csv"
    output_path = f"{partitions_directory}/{output_name}"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows → {output_path}")
