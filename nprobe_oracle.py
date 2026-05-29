import argparse
import numpy as np
import struct
import hnswlib
from pathlib import Path


# ── I/O helpers ─────────────────────────────────────────────────────────────

def read_fbin_ground_truth(filename):
    """Read a ground-truth file in either fbin (IDs + distances) or ibin (IDs only) format.

    Both formats share the same header and ID block; ibin simply has no trailing
    distances section.  The trailing seek is harmless if the file ends early.
    """
    with open(filename, "rb") as f:
        num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        K           = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        total       = int(num_queries) * int(K)
        all_ids     = np.frombuffer(f.read(total * 4), dtype=np.uint32)
        if all_ids.size != total:
            raise ValueError(
                f"Ground-truth file truncated: expected {total} IDs, got {all_ids.size}"
            )
        f.seek(total * 4, 1)          # skip distances if present (fbin); no-op for ibin
    return all_ids.reshape((int(num_queries), int(K))).astype(np.int64)


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

    Format: uint64 size | int32[size]
    (size_t header on 64-bit, then one int32 partition ID per center)
    """
    with open(filename, "rb") as f:
        size = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        data = np.frombuffer(f.read(size * 4), dtype=np.int32)
    if data.size != size:
        raise ValueError(
            f"partitions.bin truncated: expected {size} entries, got {data.size}"
        )
    return data


def load_cached_gt_vectors(filename):
    """Read a cached GT vector file written by saveCachedGTVectors() in the C++ experiment.

    Format: uint32 num_vecs | uint32 dim | float32[num_vecs * dim]
    Vectors are stored in sorted-GT-index order (matching the C++ needed_indices ordering).
    """
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


# ── Oracle ───────────────────────────────────────────────────────────────────

def find_cached_gt_vectors(base_file: str, dataset_name: str = None) -> str | None:
    """Look for a cached_gt_vectors_*.bin file in the same directory as the base file.

    theoretical_partitioning_quality.cpp writes:
        {base_file_parent}/cached_gt_vectors_{dataset_name}.bin
    If dataset_name is provided, searches for the specific file.
    Otherwise, globs for the pattern and picks the most recently modified match.
    """
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


def compute_nprobe_oracle(
    router_path:    str,
    gt_file:        str,
    partition_file: str,
    base_file:      str,
    dim:            int,
    k_neighbors:    int,
    base_format:    str  = "fbin",
    ef:             int  = 100,
    dataset_name:   str  = None,
    nprobe_values:  list = None,
    out_csv:        str  = "nprobe_oracle_results.csv",
):
    """Compute the nprobe oracle: for each nprobe value, probe the nprobe partitions
    that hold the maximum number of GT neighbours for each query.

    For every query q the oracle ranks partitions by the descending fraction of
    GT neighbours they contain and selects the top nprobe — this is the best any
    nprobe-limited router could achieve given perfect knowledge of where the GT
    vectors live.  The reported recall is the mean across queries of the fraction
    of GT neighbours covered by those nprobe optimal partitions.

    Parameters
    ----------
    router_path    : Path to the HNSW meta-index (.bin) used for routing.
    gt_file        : Ground-truth file (fbin or ibin format).
    partition_file : partitions.bin written by Coordinator::save().
    base_file      : Base vector file (fbin or u8bin).
    dim            : Vector dimensionality.
    k_neighbors    : Number of neighbours (k@k recall).
    base_format    : "fbin" or "u8bin".
    ef             : ef parameter for HNSW search (used when routing GT vectors).
    dataset_name   : Dataset name for locating cached GT vectors.
    nprobe_values  : List of nprobe values to evaluate.
    out_csv        : Path for the output CSV file.

    Returns
    -------
    list of dicts with keys: nprobe, activation, mean_recall, p5_recall, p95_recall
    """
    if nprobe_values is None:
        nprobe_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

    # ── Load ground truth ─────────────────────────────────────────────────
    gt          = read_fbin_ground_truth(gt_file)       # (num_queries, K_gt)
    num_queries = gt.shape[0]
    gt_indices_flat = gt[:, :k_neighbors].ravel()       # (Q*k,)

    print(f"Loaded ground truth: {num_queries} queries, k={k_neighbors}")

    # ── Load partition map ────────────────────────────────────────────────
    partitions     = read_partitions_bin(partition_file)
    num_partitions = int(partitions.max()) + 1

    print(f"Loaded partitions: {len(partitions)} centers, {num_partitions} partitions")

    # ── Load router ───────────────────────────────────────────────────────
    router = hnswlib.Index(space="l2", dim=dim)
    router.load_index(router_path)
    router.set_ef(ef)

    print(f"Loaded router index with {router.get_current_count()} elements")

    # ── Gather GT vectors (unique only) and route to partitions ───────────
    # np.unique returns sorted unique indices and an inverse map such that
    #   unique_indices[inverse] == gt_indices_flat
    # This aligns with the C++ needed_indices (also sorted) so cache rows
    # correspond 1-to-1 with unique_indices.
    unique_indices, inverse = np.unique(gt_indices_flat, return_inverse=True)

    cache_path = find_cached_gt_vectors(base_file, dataset_name=dataset_name)
    if cache_path:
        gt_vecs = load_cached_gt_vectors(cache_path)    # (num_unique, dim) float32
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

    # Route unique vectors; expand back to (Q*k,) via the inverse index.
    labels, _       = router.knn_query(gt_vecs, k=1)   # (num_unique, 1)
    unique_part_ids = partitions[labels[:, 0]]          # (num_unique,)
    gt_part_ids     = unique_part_ids[inverse]          # (Q*k,)
    gt_part_matrix  = gt_part_ids.reshape(num_queries, k_neighbors)

    # ── Build per-query partition distributions ───────────────────────────
    # query_dist[i, p] = fraction of query i's GT neighbours in partition p
    query_dist = np.zeros((num_queries, num_partitions), dtype=np.float64)
    for i in range(num_queries):
        counts        = np.bincount(gt_part_matrix[i], minlength=num_partitions)
        query_dist[i] = counts / k_neighbors

    # Sort each row descending; cumsum gives the achievable recall curve
    # when the top-n optimal partitions are probed.
    sorted_dist = np.sort(query_dist, axis=1)[:, ::-1]     # (Q, P)
    cumsum      = np.cumsum(sorted_dist, axis=1)            # (Q, P)

    # ── Oracle per nprobe value ───────────────────────────────────────────
    rows = []

    print(f"\n{'NProbe':>8} | {'Activation':>12} | {'Mean recall':>12} | "
          f"{'p5 recall':>10} | {'p95 recall':>10}")
    print("-" * 62)

    for nprobe in nprobe_values:
        n = min(nprobe, num_partitions)
        # cumsum[:, n-1] is the recall achieved by probing the n optimal
        # partitions for each query independently.
        recall_per_query = cumsum[:, n - 1]

        mean_recall = float(recall_per_query.mean())
        p5_recall   = float(np.percentile(recall_per_query, 5))
        p95_recall  = float(np.percentile(recall_per_query, 95))
        activation  = n / num_partitions

        print(f"{nprobe:>8d} | {activation:>12.4f} | {mean_recall:>12.4f} | "
              f"{p5_recall:>10.4f} | {p95_recall:>10.4f}")

        rows.append({
            "nprobe":      nprobe,
            "activation":  activation,
            "mean_recall": mean_recall,
            "p5_recall":   p5_recall,
            "p95_recall":  p95_recall,
        })

    # ── Write CSV ─────────────────────────────────────────────────────────
    lines = ["nprobe,activation,mean_recall,p5_recall,p95_recall"] + [
        f"{r['nprobe']},"
        f"{r['activation']:.6f},"
        f"{r['mean_recall']:.6f},"
        f"{r['p5_recall']:.6f},"
        f"{r['p95_recall']:.6f}"
        for r in rows
    ]
    with open(out_csv, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {out_csv}")

    return rows


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NProbe oracle: ideal recall for a given number of partitions searched"
    )
    parser.add_argument("--router",         required=True,
                        help="Path to HNSW meta-index (.bin)")
    parser.add_argument("--gt-file",        required=True,
                        help="Ground-truth file (fbin or ibin format)")
    parser.add_argument("--partition-file", required=True,
                        help="partitions.bin written by Coordinator::save()")
    parser.add_argument("--base-file",      required=True,
                        help="Base vector file (fbin or u8bin)")
    parser.add_argument("--dim",            required=True, type=int,
                        help="Vector dimensionality")
    parser.add_argument("--k",              required=True, type=int, dest="k_neighbors",
                        help="Number of neighbours (k@k recall)")
    parser.add_argument("--format",         default="fbin", choices=["fbin", "u8bin"],
                        dest="base_format",
                        help="Base file format (default: fbin)")
    parser.add_argument("--ef",             default=100, type=int,
                        help="HNSW ef search parameter used when routing GT vectors "
                             "(default: 100)")
    parser.add_argument("--nprobe-values",  nargs="+", type=int, default=None,
                        dest="nprobe_values",
                        help="nprobe values to evaluate "
                             "(default: 1 2 3 4 5 6 7 8 9 10 15 20)")
    parser.add_argument("--out-file",       default="nprobe_oracle_results.csv",
                        dest="out_csv",
                        help="Output CSV path (default: nprobe_oracle_results.csv)")
    parser.add_argument("--dataset-name",   default=None, dest="dataset_name",
                        help="Dataset name for locating cached GT vectors")
    args = parser.parse_args()

    compute_nprobe_oracle(
        router_path    = args.router,
        gt_file        = args.gt_file,
        partition_file = args.partition_file,
        base_file      = args.base_file,
        dim            = args.dim,
        k_neighbors    = args.k_neighbors,
        base_format    = args.base_format,
        ef             = args.ef,
        dataset_name   = args.dataset_name,
        nprobe_values  = args.nprobe_values,
        out_csv        = args.out_csv,
    )


# python nprobe_oracle.py --router /dataset/surge/results/.../metaHNSW.bin
# --gt-file ...
# --partition-file ...
# --base-file ...
# --dim 128
# --k 10
# --format u8bin
# --ef 100
# --nprobe-values 1 2 3 4 5 6 7 8 9 10 15 20
# --dataset-name ...
# --out-file /dataset/surge/results/.../nprobe_oracle_results.csv
