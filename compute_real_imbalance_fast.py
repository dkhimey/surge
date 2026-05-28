"""
compute_real_imbalance_fast.py  –  optimised rewrite of compute_real_imbalance_updated.py

Key changes vs. the original
─────────────────────────────
1. Precompute all active-vector sets in one incremental O(S) pass instead of
   replaying the runbook from scratch for every step  (was O(S²)).

2. to_rebuild=False  →  pool initialiser loads the HNSW index + partitions file
   once per *worker process* (8 loads) instead of once per *step* (345 loads).

3. All per-vector Python loops replaced with vectorised NumPy operations
   (np.add.at, np.unique) for counts, dispersion, and entropy.

4. Results are returned from workers and written by the main process;
   the multiprocessing.Manager lock is gone entirely.

5. Reduced HNSW query threads per worker to avoid over-subscription
   (8 processes × 4 threads vs. the original 8×8 on a typical machine).

6. Active-set construction uses set.update/difference_update instead of
   per-element add/remove inside Python loops.
"""

import yaml
import hnswlib
import numpy as np
import multiprocessing
import sys


# ── helpers ──────────────────────────────────────────────────────────────────

def mmap_fbin(filename):
    """Memory-map a .fbin file (uint32 header: num_points, dim)."""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_points, dim = header
    return np.memmap(
        filename,
        dtype=np.float32,
        mode="r",
        offset=8,
        shape=(num_points, dim),
    )


def precompute_active_sets(runbook_data, step_numbers):
    """
    Single incremental pass over the runbook → active index set for every
    requested step number.  O(total operations) instead of O(S × total ops).

    Handles the case where no runbook operation lands exactly on a requested
    step: the snapshot is taken from the most recent prior state.
    """
    sorted_steps = sorted(step_numbers)
    root = next(iter(runbook_data.values()))

    # Collect (step, payload) pairs in ascending order.
    ordered_ops = sorted(
        [(s, p) for s, p in root.items() if isinstance(s, int)],
        key=lambda x: x[0],
    )

    active: set = set()
    result: dict = {}
    req_idx = 0

    for rb_step, payload in ordered_ops:
        # Capture state for every requested step that falls *before* this op.
        while req_idx < len(sorted_steps) and sorted_steps[req_idx] < rb_step:
            result[sorted_steps[req_idx]] = np.array(list(active), dtype=np.int64)
            req_idx += 1

        op = payload.get("operation")
        start = payload.get("start")
        end = payload.get("end")
        if start is not None and end is not None:
            if op == "insert":
                active.update(range(start, end))
            elif op == "delete":
                active.difference_update(range(start, end))

        # Capture state for requested steps that coincide with this op.
        while req_idx < len(sorted_steps) and sorted_steps[req_idx] == rb_step:
            result[sorted_steps[req_idx]] = np.array(list(active), dtype=np.int64)
            req_idx += 1

    # Any remaining requested steps after all runbook ops.
    while req_idx < len(sorted_steps):
        result[sorted_steps[req_idx]] = np.array(list(active), dtype=np.int64)
        req_idx += 1

    return result


def compute_metrics(vectors, labels_flat, get_partition, num_partitions):
    """
    Vectorised computation of per-partition count, dispersion, and entropy.

    Parameters
    ----------
    vectors       : (N, dim) float32
    labels_flat   : (N,)    int64  – HNSW nearest-neighbour labels
    get_partition : (K,)    int64  – maps label → partition id
    num_partitions: int

    Returns
    -------
    counts       : (P,) int64
    dispersions  : (P,) float64  – mean squared distance from partition centroid / dim
    entropy      : (P,) float64  – normalised Shannon entropy over label distribution
    """
    dim = vectors.shape[1]
    vf = vectors.astype(np.float64)

    # Map each vector to its partition via its nearest-neighbour label.
    partitions = get_partition[labels_flat]   # (N,)

    # ── counts ────────────────────────────────────────────────────────────────
    counts = np.zeros(num_partitions, dtype=np.int64)
    np.add.at(counts, partitions, 1)

    # ── means ─────────────────────────────────────────────────────────────────
    sums = np.zeros((num_partitions, dim), dtype=np.float64)
    np.add.at(sums, partitions, vf)

    valid = counts > 0
    means = np.zeros((num_partitions, dim), dtype=np.float64)
    means[valid] = sums[valid] / counts[valid, None]

    # ── dispersion ────────────────────────────────────────────────────────────
    diffs = vf - means[partitions]                  # (N, dim)
    sq_norms = (diffs ** 2).sum(axis=1)             # (N,)
    dispersions = np.zeros(num_partitions, dtype=np.float64)
    np.add.at(dispersions, partitions, sq_norms)
    dispersions[valid] /= (counts[valid] * dim)

    # ── entropy over label distribution within each partition ─────────────────
    # Find unique (partition, label) pairs and their counts.
    pl_pairs = np.stack([partitions, labels_flat], axis=1)   # (N, 2)
    unique_pl, pl_counts = np.unique(pl_pairs, axis=0, return_counts=True)
    part_col = unique_pl[:, 0]                               # partition for each unique pair

    p = pl_counts.astype(np.float64) / counts[part_col]     # prob of label within partition
    entropy_contrib = -p * np.log(p)

    entropy = np.zeros(num_partitions, dtype=np.float64)
    np.add.at(entropy, part_col, entropy_contrib)

    # Normalise H by log(#unique_labels) where there is more than one label.
    unique_parts, label_counts_per_part = np.unique(part_col, return_counts=True)
    multi = label_counts_per_part > 1
    if multi.any():
        entropy[unique_parts[multi]] /= np.log(label_counts_per_part[multi])

    return counts, dispersions, entropy


# ── worker globals (to_rebuild=False path) ────────────────────────────────────

_router        = None
_get_partition = None
_base_mm       = None


def _pool_init_static(hnsw_file, partitions_file, base_file):
    """
    Pool initialiser: load shared resources once per worker process.
    Called automatically by multiprocessing.Pool before any tasks are sent.
    """
    global _router, _get_partition, _base_mm
    _router = hnswlib.Index(space="l2", dim=100)
    _router.load_index(hnsw_file)
    _router.set_ef(100)
    _get_partition = np.loadtxt(partitions_file, dtype=np.int64)[:10000]
    _base_mm       = mmap_fbin(base_file)


# ── worker functions ──────────────────────────────────────────────────────────

# 64 cores ÷ 16 worker processes = 4 threads each → all 64 cores fully used
# with no over-subscription.  Increase num_processes further (e.g. 32 w/ 2
# threads) if step-level parallelism matters more than per-step throughput.
_HNSW_THREADS = 4


def process_step_static(args):
    """Worker for to_rebuild=False.  Uses globals set by _pool_init_static."""
    stepNum, step_idx = args
    print(f"evaluating step {stepNum}")

    if step_idx.size == 0:
        return []

    vectors     = _base_mm[step_idx]
    labels, _   = _router.knn_query(vectors, k=1, num_threads=_HNSW_THREADS)
    labels_flat = labels[:, 0]

    num_partitions = int(_get_partition.max()) + 1
    counts, dispersions, entropy = compute_metrics(
        vectors, labels_flat, _get_partition, num_partitions
    )

    return [
        f"{stepNum},{p},{counts[p]},{dispersions[p]},{entropy[p]}"
        for p in range(num_partitions)
    ]


def process_step_rebuild(args):
    """Worker for to_rebuild=True.  Loads per-step HNSW + partitions."""
    stepNum, step_idx, base_file, base_dir, partitions_dir = args
    print(f"evaluating step {stepNum}")

    if step_idx.size == 0:
        return []

    hnsw_file       = f"{base_dir}/step_{stepNum:06d}_hnsw.bin"
    partitions_file = f"{partitions_dir}/step_{stepNum:06d}_partitions.csv"

    get_partition = np.loadtxt(partitions_file, dtype=np.int64)[:10000]
    router        = hnswlib.Index(space="l2", dim=100)
    router.load_index(hnsw_file)
    router.set_ef(100)

    base_mm     = mmap_fbin(base_file)
    vectors     = base_mm[step_idx]
    labels, _   = router.knn_query(vectors, k=1, num_threads=_HNSW_THREADS)
    labels_flat = labels[:, 0]

    num_partitions = int(get_partition.max()) + 1
    print(f"  step {stepNum}: {num_partitions} partitions")

    counts, dispersions, entropy = compute_metrics(
        vectors, labels_flat, get_partition, num_partitions
    )

    return [
        f"{stepNum},{p},{counts[p]},{dispersions[p]},{entropy[p]}"
        for p in range(num_partitions)
    ]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python compute_real_imbalance_fast.py "
            "<base_dir> <partitions_dir> <runbook_path> <base_file> <to_rebuild>"
        )
        sys.exit(1)

    base_dir       = sys.argv[1]
    partitions_dir = sys.argv[2]
    runbook_path   = sys.argv[3]
    base_file      = sys.argv[4]
    to_rebuild     = sys.argv[5].lower() == "true"

    with open(runbook_path) as f:
        runbook_data = yaml.safe_load(f)

    output_file = (
        f"{partitions_dir}/full_real_imbalance_rebuilding_results.csv"
        if to_rebuild
        else f"{partitions_dir}/full_real_imbalance_no_rebuilding_results.csv"
    )

    # Write CSV header if the file is new / empty.
    with open(output_file, "a") as f:
        if f.tell() == 0:
            f.write("stepNum,partition,count,dispersion,entropy\n")

    # Step numbers to process (identical to the original range).
    step_numbers = [sn - 255 for sn in range(256, 946, 2)]   # 1, 3, …, 691

    # ── ONE-TIME: precompute all active-vector sets incrementally ─────────────
    print("Pre-computing active vector sets (single incremental pass)…")
    active_sets = precompute_active_sets(runbook_data, step_numbers)
    print(f"Done: {len(active_sets)} steps captured.")

    num_processes = 16   # 16 workers × 4 HNSW threads = 64 cores

    if to_rebuild:
        # Each step needs a different HNSW index → load inside the worker.
        args_list = [
            (sn,
             active_sets.get(sn, np.array([], dtype=np.int64)),
             base_file, base_dir, partitions_dir)
            for sn in step_numbers
        ]
        with multiprocessing.Pool(processes=num_processes) as pool:
            all_results = pool.map(process_step_rebuild, args_list)

    else:
        # All steps share the same HNSW index → load once per worker process.
        hnsw_file       = f"{base_dir}/step_{1:06d}_hnsw.bin"
        partitions_file = f"{partitions_dir}/step_{1:06d}_partitions.csv"

        args_list = [
            (sn, active_sets.get(sn, np.array([], dtype=np.int64)))
            for sn in step_numbers
        ]
        with multiprocessing.Pool(
            processes=num_processes,
            initializer=_pool_init_static,
            initargs=(hnsw_file, partitions_file, base_file),
        ) as pool:
            all_results = pool.map(process_step_static, args_list)

    # ── Write all results in the main process (no lock needed) ────────────────
    with open(output_file, "a") as f:
        for step_rows in all_results:
            if step_rows:
                f.write("\n".join(step_rows) + "\n")
        f.flush()

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
