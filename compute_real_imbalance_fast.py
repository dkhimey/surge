"""
compute_real_imbalance_fast.py  –  optimised rewrite of compute_real_imbalance_updated.py

Key changes vs. the original
─────────────────────────────
1. Runbook is parsed into a compact sorted op-list and stored as a module
   global before the pool is forked.  Workers inherit it via Linux CoW fork —
   nothing is pickled through the IPC pipe.  Eliminates the ~138 GB
   precomputed-active-sets dict that caused OOM on the 100M dataset.

2. Active-set replay inside each worker uses a numpy boolean array + slice
   assignment instead of a Python set.  O(max_vectors) space (100 MB for
   100M vectors) and vectorised O(1) per insert/delete op.

3. to_rebuild=False  →  pool initialiser loads the HNSW index + partitions
   file once per worker process (N loads) instead of once per step (345 loads).

4. All per-vector Python loops replaced with vectorised NumPy operations
   (np.add.at, np.unique) for counts, dispersion, and entropy.

5. Results are returned from workers and written by the main process;
   the multiprocessing.Manager lock is gone entirely.

6. num_processes is sized conservatively for large datasets: each worker
   copies ~(active_vectors × dim × 4) bytes of vector data, so 4–6 workers
   are typically safe on a 128 GB machine with a 40 GB base file.
   Tune _NUM_PROCESSES below if needed.
"""

import glob
import os
import re
import yaml
import hnswlib
import numpy as np
import multiprocessing
import sys


# ── tuneable constants ────────────────────────────────────────────────────────

# Number of worker processes.  Each worker holds ~(active_frac × base_file_GB)
# of vector data in RAM at peak.  For a 40 GB fbin with ~50% active vectors
# that's ~20 GB per worker.  On a 128 GB machine, 4–6 workers is safe.
_NUM_PROCESSES = 4

# HNSW query threads per worker.  _NUM_PROCESSES × _HNSW_THREADS should not
# exceed the physical core count (64 here → 4 × 16 = 64, or 6 × 10 ≈ 60).
_HNSW_THREADS = 16


# ── module globals shared with workers via fork (never pickled) ───────────────

_runbook_ops   = None   # list[(step:int, op:str, start:int, end:int)], sorted
_max_vectors   = None   # int — size of the boolean active-set scratch array
_router        = None   # to_rebuild=False: loaded once per worker
_get_partition = None   # to_rebuild=False: loaded once per worker
_base_mm       = None   # to_rebuild=False: loaded once per worker


# ── helpers ───────────────────────────────────────────────────────────────────

def detect_last_step(base_dir):
    """
    Scan base_dir for step_XXXXXX_hnsw.bin files and return the highest
    step number found.
    """
    pattern = os.path.join(base_dir, "step_*_hnsw.bin")
    matches = glob.glob(pattern)
    if not matches:
        print(f"Error: no step_*_hnsw.bin files found in {base_dir}")
        sys.exit(1)
    step_re = re.compile(r"step_(\d+)_hnsw\.bin$")
    steps = [int(step_re.search(os.path.basename(p)).group(1)) for p in matches]
    return max(steps)


def _build_runbook_ops(runbook_data):
    """
    Parse the runbook YAML into a compact sorted list of
    (step, op, start, end) tuples.  This tiny structure is set as a module
    global so workers inherit it via CoW fork — no IPC overhead.
    """
    root = next(iter(runbook_data.values()))
    ops = []
    for step, payload in root.items():
        if not isinstance(step, int):
            continue
        op    = payload.get("operation")
        start = payload.get("start")
        end   = payload.get("end")
        if op in ("insert", "delete") and start is not None and end is not None:
            ops.append((step, op, start, end))
    ops.sort(key=lambda x: x[0])
    return ops


def get_active_set(step_num):
    """
    Replay runbook operations up to step_num and return active vector indices.

    Uses a numpy boolean array of length _max_vectors:
      • insert [s, e)  →  active[s:e] = True    (vectorised memset)
      • delete [s, e)  →  active[s:e] = False   (vectorised memset)

    Peak memory: _max_vectors bytes (100 MB for 100M vectors).
    Uses module globals _runbook_ops and _max_vectors, inherited via fork.
    """
    active = np.zeros(_max_vectors, dtype=bool)
    for step, op, start, end in _runbook_ops:
        if step > step_num:
            break
        if op == "insert":
            active[start:end] = True
        else:
            active[start:end] = False
    return np.nonzero(active)[0].astype(np.int64)


def _safe_memmap(filename, dtype, bytes_per_element):
    """
    Derive num_points from actual file size to avoid 'mmap length greater
    than file size' when the header is stale or the file is still being written.
    """
    file_size = os.path.getsize(filename)
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
    num_points_hdr, dim = int(header[0]), int(header[1])

    data_bytes = file_size - 8
    num_points = data_bytes // (dim * bytes_per_element)

    if num_points != num_points_hdr:
        print(
            f"Warning: {os.path.basename(filename)} header says {num_points_hdr} "
            f"points but file size implies {num_points}; using {num_points}."
        )

    return np.memmap(filename, dtype=dtype, mode="r",
                     offset=8, shape=(num_points, dim))


def mmap_fbin(filename):
    return _safe_memmap(filename, np.float32, 4)


def mmap_u8bin(filename):
    return _safe_memmap(filename, np.uint8, 1)


def open_vectors(filename):
    """Dispatch to the correct mmap helper based on file extension."""
    if filename.endswith(".u8bin"):
        return mmap_u8bin(filename)
    return mmap_fbin(filename)


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
    dispersions  : (P,) float64  – mean squared distance from centroid / dim
    entropy      : (P,) float64  – normalised Shannon entropy over labels
    """
    dim = vectors.shape[1]
    vf  = vectors.astype(np.float64)

    partitions = get_partition[labels_flat]   # (N,)

    # counts
    counts = np.zeros(num_partitions, dtype=np.int64)
    np.add.at(counts, partitions, 1)

    # means
    sums  = np.zeros((num_partitions, dim), dtype=np.float64)
    np.add.at(sums, partitions, vf)
    valid = counts > 0
    means = np.zeros((num_partitions, dim), dtype=np.float64)
    means[valid] = sums[valid] / counts[valid, None]

    # dispersion
    diffs     = vf - means[partitions]
    sq_norms  = (diffs ** 2).sum(axis=1)
    dispersions = np.zeros(num_partitions, dtype=np.float64)
    np.add.at(dispersions, partitions, sq_norms)
    dispersions[valid] /= (counts[valid] * dim)

    # entropy over label distribution within each partition
    pl_pairs          = np.stack([partitions, labels_flat], axis=1)
    unique_pl, pl_counts = np.unique(pl_pairs, axis=0, return_counts=True)
    part_col          = unique_pl[:, 0]
    p                 = pl_counts.astype(np.float64) / counts[part_col]
    entropy_contrib   = -p * np.log(p)
    entropy           = np.zeros(num_partitions, dtype=np.float64)
    np.add.at(entropy, part_col, entropy_contrib)
    unique_parts, label_counts_per_part = np.unique(part_col, return_counts=True)
    multi = label_counts_per_part > 1
    if multi.any():
        entropy[unique_parts[multi]] /= np.log(label_counts_per_part[multi])

    return counts, dispersions, entropy


# ── pool initialiser (to_rebuild=False path) ──────────────────────────────────

def _pool_init_static(hnsw_file, partitions_file, base_file):
    """
    Load shared resources once per worker process.
    _runbook_ops and _max_vectors are already in memory via fork.
    """
    global _router, _get_partition, _base_mm
    _router = hnswlib.Index(space="l2", dim=100)
    _router.load_index(hnsw_file)
    _router.set_ef(100)
    _get_partition = np.loadtxt(partitions_file, dtype=np.int64)[:10000]
    _base_mm       = open_vectors(base_file)


# ── worker functions ──────────────────────────────────────────────────────────

def process_step_static(stepNum):
    """Worker for to_rebuild=False.  Active set computed inline from fork'd globals."""
    print(f"evaluating step {stepNum}")

    step_idx = get_active_set(stepNum)
    if step_idx.size == 0:
        return []

    vectors     = np.asarray(_base_mm[step_idx], dtype=np.float32)
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
    stepNum, base_file, base_dir, partitions_dir = args
    print(f"evaluating step {stepNum}")

    step_idx = get_active_set(stepNum)
    if step_idx.size == 0:
        return []

    hnsw_file       = f"{base_dir}/step_{stepNum:06d}_hnsw.bin"
    partitions_file = f"{partitions_dir}/step_{stepNum:06d}_partitions.csv"

    get_partition = np.loadtxt(partitions_file, dtype=np.int64)[:10000]
    router        = hnswlib.Index(space="l2", dim=100)
    router.load_index(hnsw_file)
    router.set_ef(100)

    base_mm     = open_vectors(base_file)
    vectors     = np.asarray(base_mm[step_idx], dtype=np.float32)
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

    with open(output_file, "a") as f:
        if f.tell() == 0:
            f.write("stepNum,partition,count,dispersion,entropy\n")

    # ── Parse runbook into a compact global before forking ────────────────────
    # Workers inherit _runbook_ops and _max_vectors via CoW fork.
    # Nothing large is ever pickled into the IPC pipe.
    global _runbook_ops, _max_vectors
    _runbook_ops = _build_runbook_ops(runbook_data)
    _max_vectors = max(end for _, op, start, end in _runbook_ops if op == "insert")
    print(f"Runbook: {len(_runbook_ops)} ops, max vector id: {_max_vectors - 1}")

    last_step    = detect_last_step(base_dir)
    step_numbers = list(range(1, last_step + 1, 2))
    print(f"Steps to evaluate: {len(step_numbers)}  (1 … {last_step})")

    if to_rebuild:
        args_list = [
            (sn, base_file, base_dir, partitions_dir)
            for sn in step_numbers
        ]
        with multiprocessing.Pool(processes=_NUM_PROCESSES) as pool:
            all_results = pool.map(process_step_rebuild, args_list)

    else:
        hnsw_file       = f"{base_dir}/step_{1:06d}_hnsw.bin"
        partitions_file = f"{partitions_dir}/step_{1:06d}_partitions.csv"

        with multiprocessing.Pool(
            processes=_NUM_PROCESSES,
            initializer=_pool_init_static,
            initargs=(hnsw_file, partitions_file, base_file),
        ) as pool:
            all_results = pool.map(process_step_static, step_numbers)

    with open(output_file, "a") as f:
        for step_rows in all_results:
            if step_rows:
                f.write("\n".join(step_rows) + "\n")
        f.flush()

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
