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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Routing functions — each returns List[(recall, activation, ppc, query_time_s)]
# one entry per param value
# ---------------------------------------------------------------------------

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
) -> List[Tuple]:
    """BranchingFactor routing: query k nearest centers, visit all unique partitions."""
    ground_truth = read_fbin_ground_truth(gt_file)      # (Q, K)
    Q = ground_truth.shape[0]

    router = hnswlib.Index(space='l2', dim=queries.shape[1])
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    max_bf = max(branching_factors)
    router.set_ef(max(100, max_bf))
    current_count = router.get_current_count()

    partitions = np.loadtxt(
        f"{partition_dir}/step_{partition_step:06d}_partitions.csv",
        delimiter=",", dtype=int)

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
) -> List[Tuple]:
    """NProbe routing: walk centers in HNSW order until N unique partitions seen."""
    ground_truth = read_fbin_ground_truth(gt_file)      # (Q, K)
    Q = ground_truth.shape[0]

    router = hnswlib.Index(space='l2', dim=queries.shape[1])
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    max_p = max(nprobe_values)
    router.set_ef(100)
    current_count = router.get_current_count()

    partitions = np.loadtxt(
        f"{partition_dir}/step_{partition_step:06d}_partitions.csv",
        delimiter=",", dtype=int)

    # GT partition assignment
    gt_indices = ground_truth[:, :k_neighbors].ravel().astype(np.int64)
    gt_vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)
    gt_labels, _ = router.knn_query(gt_vecs, k=1)
    gt_parts = partitions[gt_labels[:, 0]].reshape(Q, k_neighbors)  # (Q, k)

    # Find k large enough that every query sees >= max_p unique partitions
    k = max_p
    final_elapsed = 0.0
    niter = 0
    while True:
        niter += 1
        k_capped = min(k, current_count)
        t0 = time.perf_counter()
        labels, _ = router.knn_query(queries, k=k_capped)  # (Q, k_capped)
        final_elapsed = time.perf_counter() - t0
        pids_all = partitions[labels]  # (Q, k_capped)
        # Check: does every query have >= max_p unique partitions?
        min_unique = min(len(np.unique(row)) for row in pids_all)
        if min_unique >= max_p or k_capped >= current_count:
            break
        k *= 10
    print(f"NProbe: k={k_capped}, min_unique={min_unique}, iters={niter}, "
          f"time={final_elapsed:.4f}s")

    # Precompute unique-partition order per query
    # unique_orders[i] = array of unique pids in first-occurrence order
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
) -> List[Tuple]:
    """RecallTarget routing: score partitions by size * exp(-d/d0), greedily
    accumulate until cumulative probability >= target."""
    ground_truth = read_fbin_ground_truth(gt_file)      # (Q, K)
    Q = ground_truth.shape[0]

    router = hnswlib.Index(space='l2', dim=queries.shape[1])
    router.load_index(f"{base_dir}/step_{hnsw_step:06d}_hnsw.bin")
    router.set_ef(max(100, k_rt))

    partitions = np.loadtxt(
        f"{partition_dir}/step_{partition_step:06d}_partitions.csv",
        delimiter=",", dtype=int)
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

    # Score: part_size[pid] * exp(-d / d0) where d0 = mean distance per query
    d0 = distances.mean(axis=1, keepdims=True).clip(min=1e-10)   # (Q, 1)
    weights = part_size[pids] * np.exp(-distances / d0)           # (Q, k_rt)

    # Scatter-add weights into per-partition scores (Q, P)
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

    results = []
    for target in targets:
        # first column index where cumprob >= target
        reached = cumprobs >= target                               # (Q, P) bool
        # argmax returns 0 if no True; use max with last column as fallback
        first_idx = np.where(reached.any(axis=1),
                             reached.argmax(axis=1),
                             num_partitions - 1)                   # (Q,)

        # Build visited mask from first_idx + 1 partitions per query
        visited_mask = np.zeros((Q, num_partitions), dtype=bool)
        for i in range(Q):
            n = int(first_idx[i]) + 1
            visited_mask[i, ordered_pids[i, :n]] = True

        r = _recall(visited_mask, gt_parts, k_neighbors)
        act = _activation(visited_mask, num_partitions)
        ppc = visited_mask.sum(axis=0).astype(np.int64)
        results.append((r, act, ppc, elapsed))

    return results


# ---------------------------------------------------------------------------
# Routing dispatch
# ---------------------------------------------------------------------------

ROUTING_FUNCS = {
    "BranchingFactor": compute_recall_branching_factor,
    "NProbe": compute_recall_nprobe,
    "RecallTarget": compute_recall_recall_target,
}


def _call_routing(mode, queries, base_mmap, gt_file,
                  base_dir, partition_dir, hnsw_step, partition_step,
                  params, k_neighbors, num_partitions):
    fn = ROUTING_FUNCS[mode]
    if mode in ("BranchingFactor", "NProbe"):
        params_cast = [int(p) for p in params]
    else:
        params_cast = [float(p) for p in params]
    return fn(queries, base_mmap, gt_file,
              base_dir, partition_dir, hnsw_step, partition_step,
              params_cast, k_neighbors, num_partitions)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute theoretical recall for a given routing mode."
    )
    parser.add_argument("--base-dir", required=True,
                        help="Directory containing step_NNNNNN_hnsw.bin files")
    parser.add_argument("--partitions-dir", required=True,
                        help="Directory containing step_NNNNNN_partitions.csv files")
    parser.add_argument("--mode", required=True,
                        choices=["BranchingFactor", "NProbe", "RecallTarget"],
                        help="Routing mode to evaluate")
    parser.add_argument("--params", required=True, nargs="+", type=float,
                        help="Parameter values for the chosen mode "
                             "(integers for BranchingFactor/NProbe, "
                             "floats 0-1 for RecallTarget)")
    parser.add_argument("--no-rebuilds", action="store_true",
                        help="Evaluate only the no-rebuild path "
                             "(always use first-step router/partitions)")
    parser.add_argument("--runbook-path", required=True,
                        help="Path to the runbook YAML file")
    parser.add_argument("--query-file", required=True,
                        help="Path to the query file (fbin or u8bin format)")
    parser.add_argument("--base-file", required=True,
                        help="Path to the base vectors file")
    parser.add_argument("--gt-dir", required=True,
                        help="Directory containing stepN.gt100 ground-truth files")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    mode = args.mode
    params = args.params

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

    # Detect step_end for rebuild mode
    step_end = step_start
    if not args.no_rebuilds:
        while Path(f"{base_directory}/step_{step_end:06d}_hnsw.bin").exists():
            step_end += 2

    first_partitions = np.loadtxt(
        f"{partitions_directory}/step_{step_start:06d}_partitions.csv",
        delimiter=",", dtype=int)
    num_partitions = int(first_partitions.max()) + 1
    print(f"Found {num_partitions} partitions")

    rows = []

    # ------------------------------------------------------------------
    # Rebuild pass
    # ------------------------------------------------------------------
    if not args.no_rebuilds:
        for step in range(step_start, step_end, 2):
            gt_test = f"{args.gt_dir}/step{step + 1}.gt100"
            hnsw_file = f"{base_directory}/step_{step:06d}_hnsw.bin"
            partition_file = f"{partitions_directory}/step_{step:06d}_partitions.csv"
            if not all(Path(f).exists() for f in [gt_test, hnsw_file, partition_file]):
                missing = [f for f in [gt_test, hnsw_file, partition_file]
                           if not Path(f).exists()]
                print(f"Step {step}: missing {missing} — stopping rebuild pass.")
                break

            step_results = _call_routing(
                mode, queries, base_mmap, gt_test,
                base_directory, partitions_directory,
                step, step, params, 10, num_partitions)

            for param, (r, act, ppc, t) in zip(params, step_results):
                rows.append({
                    "step": step,
                    "rebuild": "rebuild",
                    "mode": mode,
                    "param": param,
                    "recall": r,
                    "activation": act,
                    "cof": _cof(ppc),
                    "query_time_s": t,
                })
            print(f"REBUILD step {step}: "
                  f"recall={[r for r, *_ in step_results]}")

    # ------------------------------------------------------------------
    # No-rebuild pass  (always use step_start router/partitions)
    # ------------------------------------------------------------------
    no_rebuild_steps = [
        s for s in runbook_steps
        if s >= step_start and (s - step_start) % 2 == 0
    ]

    for step in no_rebuild_steps:
        gt_test = f"{args.gt_dir}/step{step + 1}.gt100"
        if not Path(gt_test).exists():
            print(f"Missing GT at step {step}: {gt_test}. Skipping.")
            continue

        step_results = _call_routing(
            mode, queries, base_mmap, gt_test,
            base_directory, partitions_directory,
            step_start, step_start, params, 10, num_partitions)

        for param, (r, act, ppc, t) in zip(params, step_results):
            rows.append({
                "step": step,
                "rebuild": "no_rebuild",
                "mode": mode,
                "param": param,
                "recall": r,
                "activation": act,
                "cof": _cof(ppc),
                "query_time_s": t,
            })
        print(f"NO-REBUILD step {step}: "
              f"recall={[r for r, *_ in step_results]}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows, columns=[
        "step", "rebuild", "mode", "param",
        "recall", "activation", "cof", "query_time_s"])

    suffix = "_no_rebuilds" if args.no_rebuilds else ""
    output_name = f"full_results_{mode}{suffix}.csv"
    output_path = f"{partitions_directory}/{output_name}"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows → {output_path}")
