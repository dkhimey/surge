#!/usr/bin/env python3
"""
Matched-activation oracle for BranchingFactor routing.

For each K, computes oracle recall using the optimal K partitions.
"""

import argparse
import numpy as np
import hnswlib
from pathlib import Path



def read_fbin_ground_truth(filename):
    """Read ground-truth file (fbin or ibin format); trailing seek handles both."""
    with open(filename, "rb") as f:
        num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        K           = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        total       = int(num_queries) * int(K)
        all_ids     = np.frombuffer(f.read(total * 4), dtype=np.uint32)
        if all_ids.size != total:
            raise ValueError(
                f"Ground-truth file truncated: expected {total} IDs, got {all_ids.size}"
            )
        f.seek(total * 4, 1)   # skip distances if present (fbin); no-op for ibin
    return all_ids.reshape((int(num_queries), int(K))).astype(np.int64)


def read_fbin_slice(filename, start, end):
    """Read vectors [start, end) from a float32 fbin file."""
    with open(filename, "rb") as f:
        num_points, dim = np.frombuffer(f.read(8), dtype=np.uint32)
        if start < 0 or end > int(num_points):
            raise ValueError(
                f"Slice [{start}, {end}) out of bounds (file has {num_points} vectors)"
            )
        count = end - start
        f.seek(8 + start * int(dim) * 4)
        data = np.frombuffer(f.read(count * int(dim) * 4), dtype=np.float32)
    return data.reshape(count, int(dim))


def read_u8bin_slice(filename, start, end):
    """Read vectors [start, end) from a uint8 u8bin file, returned as float32."""
    with open(filename, "rb") as f:
        num_points, dim = np.frombuffer(f.read(8), dtype=np.uint32)
        if start < 0 or end > int(num_points):
            raise ValueError(
                f"Slice [{start}, {end}) out of bounds (file has {num_points} vectors)"
            )
        count = end - start
        f.seek(8 + start * int(dim))
        data = np.frombuffer(f.read(count * int(dim)), dtype=np.uint8)
    return data.reshape(count, int(dim)).astype(np.float32)


def mmap_fbin(filename):
    """Memory-map a float32 fbin base file."""
    with open(filename, "rb") as f:
        num_points, dim = np.frombuffer(f.read(8), dtype=np.uint32)
    return np.memmap(filename, dtype=np.float32, mode="r",
                     offset=8, shape=(int(num_points), int(dim)))


def mmap_u8bin(filename):
    """Memory-map a uint8 u8bin base file."""
    with open(filename, "rb") as f:
        num_points, dim = np.frombuffer(f.read(8), dtype=np.uint32)
    return np.memmap(filename, dtype=np.uint8, mode="r",
                     offset=8, shape=(int(num_points), int(dim)))


def read_partitions_bin(filename):
    """Read a partitions.bin file written by Coordinator::save().
    Format: uint64 size | int32[size]"""
    with open(filename, "rb") as f:
        size = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        data = np.frombuffer(f.read(size * 4), dtype=np.int32)
    if data.size != size:
        raise ValueError(
            f"partitions.bin truncated: expected {size} entries, got {data.size}"
        )
    return data


def load_cached_gt_vectors(filename):
    """Read cached GT vector file written by saveCachedGTVectors().
    Format: uint32 num_vecs | uint32 dim | float32[num_vecs * dim]"""
    with open(filename, "rb") as f:
        header   = np.frombuffer(f.read(8), dtype=np.uint32)
        num_vecs = int(header[0])
        dim      = int(header[1])
        vecs     = np.frombuffer(f.read(num_vecs * dim * 4), dtype=np.float32)
    if vecs.size != num_vecs * dim:
        raise ValueError(
            f"Cached GT file truncated: expected {num_vecs * dim} floats, got {vecs.size}"
        )
    return vecs.reshape(num_vecs, dim)


def find_cached_gt_vectors(base_file: str, dataset_name: str = None) -> str | None:
    """Find cached_gt_vectors_*.bin in base_file's directory; glob for most recent if no dataset_name."""
    base_dir = Path(base_file).parent

    if dataset_name:
        specific_file = base_dir / f"cached_gt_vectors_{dataset_name}.bin"
        if specific_file.exists():
            print(f"Found cached GT vectors for dataset '{dataset_name}': {specific_file}")
            return str(specific_file)
        else:
            print(f"No cached GT vectors found for dataset '{dataset_name}' at {specific_file}")
            return None
    else:
        candidates = sorted(
            base_dir.glob("cached_gt_vectors_*.bin"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        if len(candidates) > 1:
            print(f"Found {len(candidates)} cached GT vector files; using the most recent:")
            for c in candidates:
                print(f"  {c}")
        return str(candidates[0])



def compute_branching_factor_oracle(
    router_path:       str,
    gt_file:           str,
    partition_file:    str,
    base_file:         str,
    query_file:        str,
    dim:               int,
    k_neighbors:       int,
    base_format:       str  = "fbin",
    query_format:      str  = "fbin",
    ef:                int  = 100,
    dataset_name:      str  = None,
    branching_factors: list = None,
    max_queries:       int  = 10000,
    out_csv:           str  = "branching_factor_oracle_results.csv",
):
    """Matched-activation oracle for BranchingFactor routing.

    Parameters
    ----------
    router_path       : Path to the HNSW meta-index (.bin) used for routing.
    gt_file           : Ground-truth file (fbin or ibin format).
    partition_file    : partitions.bin written by Coordinator::save().
    base_file         : Base vector file (fbin or u8bin).
    query_file        : Query vector file (fbin or u8bin).
    dim               : Vector dimensionality.
    k_neighbors       : Number of neighbours (k@k recall).
    base_format       : "fbin" or "u8bin" for the base file.
    query_format      : "fbin" or "u8bin" for the query file.
    ef                : HNSW ef parameter for routing searches.
    dataset_name      : Dataset name for locating cached GT vectors.
    branching_factors : List of K values to evaluate.
    max_queries       : Maximum number of queries to load (default: 10000).
    out_csv           : Output CSV path.

    Returns
    -------
    list of dicts with keys:
        branching_factor, mean_partitions_visited, activation,
        actual_recall, oracle_recall, recall_gap,
        p5_actual_recall, p95_actual_recall,
        p5_oracle_recall, p95_oracle_recall
    """
    if branching_factors is None:
        branching_factors = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50]

    gt          = read_fbin_ground_truth(gt_file)       # (Q_gt, K_gt)
    num_queries = gt.shape[0]

    print(f"Loaded ground truth: {num_queries} queries, k_gt={gt.shape[1]}, "
          f"using k={k_neighbors}")

    partitions     = read_partitions_bin(partition_file)
    num_partitions = int(partitions.max()) + 1

    print(f"Loaded partitions: {len(partitions)} centers, {num_partitions} partitions")

    router = hnswlib.Index(space="l2", dim=dim)
    router.load_index(router_path)
    max_bf    = max(branching_factors)
    n_centers = router.get_current_count()
    router.set_ef(max(ef, max_bf))

    print(f"Loaded router index with {n_centers} elements, ef={max(ef, max_bf)}")

    end = min(max_queries, num_queries)
    if query_format == "u8bin":
        queries = read_u8bin_slice(query_file, 0, end)
    else:
        queries = read_fbin_slice(query_file, 0, end)

    # Trim GT to match queries actually loaded
    gt          = gt[:end]
    num_queries = end

    print(f"Loaded {num_queries} query vectors (dim={queries.shape[1]})")

    # Use unique GT indices to avoid redundant HNSW lookups and align with cached files
    gt_indices_flat         = gt[:, :k_neighbors].ravel()        # (Q*k,)
    unique_indices, inverse = np.unique(gt_indices_flat, return_inverse=True)

    cache_path = find_cached_gt_vectors(base_file, dataset_name=dataset_name)
    if cache_path:
        gt_vecs = load_cached_gt_vectors(cache_path)
        if gt_vecs.shape[0] != len(unique_indices):
            print(
                f"Warning: cache has {gt_vecs.shape[0]} vectors but GT requires "
                f"{len(unique_indices)} unique indices — cache may be stale. "
                f"Falling back to base file."
            )
            cache_path = None
        else:
            print(f"Loaded {gt_vecs.shape[0]:,} cached GT vectors from {cache_path}")

    if not cache_path:
        base_mmap = mmap_u8bin(base_file) if base_format == "u8bin" else mmap_fbin(base_file)
        gt_vecs   = np.ascontiguousarray(base_mmap[unique_indices], dtype=np.float32)
        print(f"Gathered {len(unique_indices):,} unique GT vectors from base file")

    # Route unique GT vectors; expand back via inverse index mapping
    gt_labels, _    = router.knn_query(gt_vecs, k=1)   # (num_unique, 1)
    unique_part_ids = partitions[gt_labels[:, 0]]       # (num_unique,)
    gt_part_ids     = unique_part_ids[inverse]           # (Q*k,)
    gt_part_matrix  = gt_part_ids.reshape(num_queries, k_neighbors)   # (Q, k)

    # Compute per-query GT partition distributions and optimal partitions by density
    query_dist = np.zeros((num_queries, num_partitions), dtype=np.float64)
    for i in range(num_queries):
        counts        = np.bincount(gt_part_matrix[i], minlength=num_partitions)
        query_dist[i] = counts / k_neighbors

    opt_order   = np.argsort(-query_dist, axis=1)   # (Q, P) descending by GT density
    sorted_dist = np.take_along_axis(query_dist, opt_order, axis=1)   # (Q, P)
    cumsum_gt   = np.cumsum(sorted_dist, axis=1)     # (Q, P)

    k_query = min(max_bf, n_centers)
    print(f"\nQuerying router HNSW: {num_queries} queries × top-{k_query} centers …")
    query_labels, _ = router.knn_query(queries, k=k_query)   # (Q, k_query)
    query_pids      = partitions[query_labels]                # (Q, k_query)

    rows = []

    print(f"\n{'K':>6} | {'Mean parts':>11} | {'Activation':>11} | "
          f"{'Actual recall':>14} | {'Oracle recall':>14} | {'Gap':>8}")
    print("-" * 75)

    for bf in branching_factors:
        k_eff    = min(bf, k_query)
        pids_k   = query_pids[:, :k_eff]   # (Q, k_eff)

    # Mark visited partitions for BF K
        visited_bf = np.zeros((num_queries, num_partitions), dtype=bool)
        visited_bf[np.arange(num_queries)[:, None], pids_k] = True

        # GT visited: which of each query's GT neighbours' partitions were visited?
        gt_visited              = visited_bf[np.arange(num_queries)[:, None], gt_part_matrix]
        actual_recall_per_query = gt_visited.sum(axis=1) / k_neighbors   # (Q,)

        # Partitions visited per query

        # Oracle recall: optimal partitions matching per-query visit count
        idx_oracle              = np.clip(n_q - 1, 0, num_partitions - 1)
        oracle_recall_per_query = cumsum_gt[np.arange(num_queries), idx_oracle]   # (Q,)

        mean_parts  = float(n_q.mean())
        activation  = mean_parts / num_partitions
        actual_mean = float(actual_recall_per_query.mean())
        oracle_mean = float(oracle_recall_per_query.mean())
        gap         = oracle_mean - actual_mean
        p5_actual   = float(np.percentile(actual_recall_per_query,  5))
        p95_actual  = float(np.percentile(actual_recall_per_query, 95))
        p5_oracle   = float(np.percentile(oracle_recall_per_query,  5))
        p95_oracle  = float(np.percentile(oracle_recall_per_query, 95))

        print(f"{bf:>6d} | {mean_parts:>11.2f} | {activation:>11.4f} | "
              f"{actual_mean:>14.4f} | {oracle_mean:>14.4f} | {gap:>8.4f}")

        rows.append({
            "branching_factor":        bf,
            "mean_partitions_visited": mean_parts,
            "activation":              activation,
            "actual_recall":           actual_mean,
            "oracle_recall":           oracle_mean,
            "recall_gap":              gap,
            "p5_actual_recall":        p5_actual,
            "p95_actual_recall":       p95_actual,
            "p5_oracle_recall":        p5_oracle,
            "p95_oracle_recall":       p95_oracle,
        })

    header = (
        "branching_factor,mean_partitions_visited,activation,"
        "actual_recall,oracle_recall,recall_gap,"
        "p5_actual_recall,p95_actual_recall,"
        "p5_oracle_recall,p95_oracle_recall"
    )
    lines = [header] + [
        f"{r['branching_factor']},"
        f"{r['mean_partitions_visited']:.4f},"
        f"{r['activation']:.6f},"
        f"{r['actual_recall']:.6f},"
        f"{r['oracle_recall']:.6f},"
        f"{r['recall_gap']:.6f},"
        f"{r['p5_actual_recall']:.6f},"
        f"{r['p95_actual_recall']:.6f},"
        f"{r['p5_oracle_recall']:.6f},"
        f"{r['p95_oracle_recall']:.6f}"
        for r in rows
    ]
    with open(out_csv, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {out_csv}")

    return rows



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("BranchingFactor oracle: actual vs. optimal recall for each K.")
    )
    parser.add_argument("--router",            required=True,
                        help="Path to HNSW meta-index (.bin)")
    parser.add_argument("--gt-file",           required=True,
                        help="Ground-truth file (fbin or ibin format)")
    parser.add_argument("--partition-file",    required=True,
                        help="partitions.bin written by Coordinator::save()")
    parser.add_argument("--base-file",         required=True,
                        help="Base vector file (fbin or u8bin)")
    parser.add_argument("--query-file",        required=True,
                        help="Query vector file (fbin or u8bin)")
    parser.add_argument("--dim",               required=True, type=int,
                        help="Vector dimensionality")
    parser.add_argument("--k",                 required=True, type=int, dest="k_neighbors",
                        help="Number of neighbours (k@k recall)")
    parser.add_argument("--format",            default="fbin", choices=["fbin", "u8bin"],
                        dest="base_format",
                        help="Base file format (default: fbin)")
    parser.add_argument("--query-format",      default="fbin", choices=["fbin", "u8bin"],
                        dest="query_format",
                        help="Query file format (default: fbin)")
    parser.add_argument("--ef",                default=100, type=int,
                        help="HNSW ef parameter (default: 100)")
    parser.add_argument("--branching-factors", nargs="+", type=int, default=None,
                        dest="branching_factors",
                        help="Branching factors to evaluate (default: 1 2 5 10 15 20 25 30 35 40 50)")
    parser.add_argument("--max-queries",       default=10000, type=int,
                        dest="max_queries",
                        help="Maximum number of queries to load (default: 10000)")
    parser.add_argument("--dataset-name",      default=None, dest="dataset_name",
                        help="Dataset name for locating cached GT vectors")
    parser.add_argument("--out-file",          default="branching_factor_oracle_results.csv",
                        dest="out_csv",
                        help="Output CSV file (default: branching_factor_oracle_results.csv)")
    args = parser.parse_args()

    compute_branching_factor_oracle(
        router_path       = args.router,
        gt_file           = args.gt_file,
        partition_file    = args.partition_file,
        base_file         = args.base_file,
        query_file        = args.query_file,
        dim               = args.dim,
        k_neighbors       = args.k_neighbors,
        base_format       = args.base_format,
        query_format      = args.query_format,
        ef                = args.ef,
        dataset_name      = args.dataset_name,
        branching_factors = args.branching_factors,
        max_queries       = args.max_queries,
        out_csv           = args.out_csv,
    )


# python branching_factor_oracle.py \
#   --router           /dataset/surge/results/.../metaHNSW.bin \
#   --gt-file          /dataset/surge/gt/step1.gt100 \
#   --partition-file   /dataset/surge/results/.../partitions.bin \
#   --base-file        /dataset/msturing-1B/base.u8bin \
#   --query-file       /dataset/msturing-1B/query.fbin \
#   --dim              100 \
#   --k                10 \
#   --format           u8bin \
#   --query-format     fbin \
#   --ef               100 \
#   --branching-factors 1 2 5 10 15 20 25 30 35 40 50 \
#   --dataset-name     msturing-500M \
#   --out-file         /dataset/surge/results/.../branching_factor_oracle_results.csv
