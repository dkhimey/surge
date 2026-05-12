import numpy as np
import struct
from pathlib import Path
import re
import yaml
import argparse

import glob
import matplotlib.pyplot as plt
import hnswlib
import time
import pandas as pd

def read_fbin_slice(filename, start, end):
    """Read vectors [start, end) from an fbin file."""
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

def read_fbin_ground_truth(filename):
    with open(filename, "rb") as f:
        # Read number of queries and K
        num_queries = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        K = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        total_elements = num_queries * K

        # Read all IDs
        all_ids = np.frombuffer(f.read(total_elements * 4), dtype=np.uint32)

        if all_ids.size != total_elements:
            raise ValueError("Failed to read neighbor IDs")

        # Skip distances (float32)
        f.seek(total_elements * 4, 1)

        # Reshape into list of lists
        gt = all_ids.reshape((num_queries, K)).astype(int).tolist()

        return gt

def load_runbook(path, dataset_key):
    with open(path, "r") as f:
        rb = yaml.safe_load(f)
    steps = rb[dataset_key]
    # filter numeric steps, sorted
    ordered = [
        steps[k] for k in sorted(k for k in steps if isinstance(k, int))
    ]
    return ordered

def mmap_fbin(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_points, dim = header
    mm = np.memmap(
        filename,
        dtype=np.float32,
        mode="r",
        offset=8,
        shape=(num_points, dim)
    )
    return mm

def read_u8bin_slice(filename, start, end):
    """Read vectors [start, end) from a u8bin file (uint8 data), returned as float32."""
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

def mmap_u8bin(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_points, dim = header
    mm = np.memmap(
        filename,
        dtype=np.uint8,
        mode="r",
        offset=8,
        shape=(num_points, dim)
    )
    return mm

from typing import Dict, Any
def parse_runbook(path):
    with open(path) as f:
        data = yaml.safe_load(f)

    # Top-level key (e.g., "msturing-30M-clustered")
    root = next(iter(data.values()))

    deltas: Dict[int, int] = {}

    for step, payload in root.items():
        # Skip non-step metadata (e.g., max_pts)
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
def find_first_available_step(base_dir, partitions_dir, use_centers):
    """Auto-detect the smallest step N for which both a router file
    (step_NNNNNN_hnsw.bin or step_NNNNNN_centers.csv, depending on mode) and
    a partitions file (step_NNNNNN_partitions.csv) exist on disk.

    Returns the step number as an int, or raises FileNotFoundError if no
    common step is present in both directories.

    This lets the script work with output directories that don't start at
    step 1 — e.g., when msturing-cluster-analysis was run with --start-step N
    and collapsed earlier inserts into one initialization saved as step_<N-1>_*.
    """
    router_suffix = "_centers.csv" if use_centers else "_hnsw.bin"

    router_re = re.compile(r"^step_(\d+)" + re.escape(router_suffix) + r"$")
    router_steps = set()
    for f in Path(base_dir).iterdir():
        m = router_re.match(f.name)
        if m:
            router_steps.add(int(m.group(1)))

    partition_re = re.compile(r"^step_(\d+)_partitions\.csv$")
    partition_steps = set()
    for f in Path(partitions_dir).iterdir():
        m = partition_re.match(f.name)
        if m:
            partition_steps.add(int(m.group(1)))

    common = router_steps & partition_steps
    if not common:
        raise FileNotFoundError(
            f"Could not auto-detect a starting step.\n"
            f"  base-dir       = {base_dir}\n"
            f"    router files ({router_suffix}) found at steps: "
            f"{sorted(router_steps) if router_steps else 'none'}\n"
            f"  partitions-dir = {partitions_dir}\n"
            f"    partition files found at steps: "
            f"{sorted(partition_steps) if partition_steps else 'none'}\n"
            f"  Need at least one step number that has BOTH a router file "
            f"and a partitions file."
        )
    return min(common)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute theoretical recall with configurable directories and branching factors."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Directory containing hnsw_step_*.bin files",
    )
    parser.add_argument(
        "--partitions-dir",
        required=True,
        help="Directory containing partitions_step_*.csv files",
    )
    parser.add_argument(
        "--branching-factors",
        required=True,
        nargs="+",
        type=int,
        help="List of branching factors to evaluate",
    )
    parser.add_argument(
        "--enforce-p",
        action="store_true",
        help="Use compute_recall_enforce_p with the provided partition counts",
    )
    parser.add_argument(
        "--brute-force",
        action="store_true",
        help="Use compute_recall_brute_force (direct nearest-center routing) instead of HNSW",
    )
    parser.add_argument(
        "--brute-force-enforce-p",
        action="store_true",
        help="Use compute_recall_brute_force_enforce_p (brute-force routing + enforce p unique partitions)",
    )
    parser.add_argument(
        "--hnswlib-bf",
        action="store_true",
        help="Use compute_recall_hnswlib_bf (hnswlib BFIndex exact brute-force routing)",
    )
    parser.add_argument(
        "--no-rebuilds",
        action="store_true",
        help="Compute only no-rebuild statistics (always use step 1 router/partitions)",
    )
    parser.add_argument(
        "--runbook-path",
        required=True,
        help="Path to the runbook YAML file",
    )
    parser.add_argument(
        "--query-file",
        required=True,
        help="Path to the query file (fbin format)",
    )
    parser.add_argument(
        "--base-file",
        required=True,
        help="Path to the base vectors file (fbin format)",
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Directory path for ground truth files (without step number, e.g., /path/to/step)",
    )
    return parser.parse_args()

def compute_recall(
    base_mmap, 
    gt_file, 
    base_dir, 
    partition_dir, 
    hnswStepNum, 
    partitionStepNum, 
    branchingFactors, 
    k_neighbors, 
    numPartitions
):
    ground_truth = read_fbin_ground_truth(gt_file)
    num_queries = len(ground_truth)

    # routing hnsw
    router = hnswlib.Index(space='l2', dim=100)
    router.load_index(f"{base_dir}/step_{hnswStepNum:06d}_hnsw.bin") 
    max_k = max(branchingFactors) if branchingFactors else 1
    router.set_ef(max(100, max_k))
    current_count = router.get_current_count()

    # partition assignments
    partition_file = f"{partition_dir}/step_{partitionStepNum:06d}_partitions.csv"
    partitions = np.loadtxt(partition_file, delimiter=",", dtype=int)

    # flatten gt
    num_queries = len(ground_truth)
    gt_indices = np.empty(num_queries * k_neighbors, dtype=np.int64)
    pos = 0
    for gt_list in ground_truth:
        gt_indices[pos:pos + k_neighbors] = gt_list[:k_neighbors]
        pos += k_neighbors

    # fetch vecs
    vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)  # (Q*k, dim)
    labels, _ = router.knn_query(vecs, k=1)              # (Q*k, 1)
    parts = partitions[labels[:, 0]]                     # (Q*k,)

    # get relevant gts
    ground_truth_partitions = {}
    pos = 0
    for i in range(num_queries):
        ground_truth_partitions[i] = parts[pos:pos + k_neighbors].tolist()
        pos += k_neighbors

    results_recall = []
    results_activations = []
    partition_counts = []
    query_search_times = []

    for branchingFactor in branchingFactors:
        k = min(branchingFactor, current_count)
        if k <= 0:
            raise ValueError("HNSW index has no elements; cannot query.")
        t0 = time.perf_counter()
        labels, distances = router.knn_query(queries, k=k)
        query_search_times.append(time.perf_counter() - t0)
        # get the partitions visited for each query
        visited = {}
        activation_rates = 0.0
        per_partition_counts = np.zeros((numPartitions,), dtype=np.int64)

        for i, label_list in enumerate(labels):
            visited_partitions = set()
            for label in label_list:
                partition_id = partitions[label]
                per_partition_counts[partition_id] += 1

                visited_partitions.add(partition_id)
            visited[i] = visited_partitions

            activation = len(visited_partitions) 
            # print(activation)
            activation_rates += activation/numPartitions
        
        total_recall = 0.0
        for i in range(num_queries):
            gt_parts = ground_truth_partitions[i]
            visited_parts = visited[i]
            hits = 0

            for part in gt_parts:
                if part in visited_parts:
                    hits += 1

            recall = hits / len(gt_parts)
            total_recall += recall

        average_recall = total_recall / num_queries
        results_recall.append(average_recall)
        results_activations.append(activation_rates/num_queries)    
        partition_counts.append(per_partition_counts)

    return results_recall, results_activations, partition_counts, query_search_times

def compute_recall_enforce_p(
    base_mmap,
    gt_file,
    base_dir,
    partition_dir,
    hnswStepNum,
    partitionStepNum,
    partitions_to_search,
    k_neighbors,
    numPartitions,
):
    ground_truth = read_fbin_ground_truth(gt_file)
    num_queries = len(ground_truth)

    # routing hnsw
    router = hnswlib.Index(space='l2', dim=100)
    router.load_index(f"{base_dir}/step_{hnswStepNum:06d}_hnsw.bin")
    max_p = max(partitions_to_search) if partitions_to_search else 1
    router.set_ef(100)

    # partition assignments
    partition_file = f"{partition_dir}/step_{partitionStepNum:06d}_partitions.csv"
    partitions = np.loadtxt(partition_file, delimiter=",", dtype=int)

    # flatten gt
    gt_indices = np.empty(num_queries * k_neighbors, dtype=np.int64)
    pos = 0
    for gt_list in ground_truth:
        gt_indices[pos:pos + k_neighbors] = gt_list[:k_neighbors]
        pos += k_neighbors

    # fetch vecs
    vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)  # (Q*k, dim)
    labels, _ = router.knn_query(vecs, k=1)             # (Q*k, 1)
    parts = partitions[labels[:, 0]]                    # (Q*k,)

    # get relevant gts
    ground_truth_partitions = {}
    pos = 0
    for i in range(num_queries):
        ground_truth_partitions[i] = parts[pos:pos + k_neighbors].tolist()
        pos += k_neighbors

    current_count = router.get_current_count()
    k = 3200
    final_query_search_time = 0.0
    t0 = time.perf_counter()
    niter = 0
    while True:
        niter+=1
        labels, distances = router.knn_query(queries, k=k)
        final_query_search_time = time.perf_counter() - t0  # overwrite; only keep time of the final used call
        min_unique = min(len(set(partitions[label_list])) for label_list in labels)
        if min_unique >= max_p or k >= current_count:
            break
        k *= 2

    print(f"final query search time: {final_query_search_time:.4f} seconds over {niter} iterations, with final k={k} and min_unique={min_unique}")

    def first_p_unique(label_list, p):
        visited = set()
        for label in label_list:
            partition_id = partitions[label]
            visited.add(partition_id)
            if len(visited) == p:
                break
        return visited

    results_recall = []
    results_activations = []
    partition_counts = []
    query_search_times = []

    for p in partitions_to_search:
        visited = {}
        activation_rates = 0.0
        per_partition_counts = np.zeros((numPartitions,), dtype=np.int64)

        for i, label_list in enumerate(labels):
            visited_partitions = first_p_unique(label_list, p)
            for partition_id in visited_partitions:
                per_partition_counts[partition_id] += 1
            visited[i] = visited_partitions
            activation_rates += len(visited_partitions) / numPartitions

        total_recall = 0.0
        for i in range(num_queries):
            gt_parts = ground_truth_partitions[i]
            visited_parts = visited[i]
            hits = 0

            for part in gt_parts:
                if part in visited_parts:
                    hits += 1

            recall = hits / len(gt_parts)
            total_recall += recall

        average_recall = total_recall / num_queries
        results_recall.append(average_recall)
        results_activations.append(activation_rates / num_queries)
        partition_counts.append(per_partition_counts)
        query_search_times.append(final_query_search_time)

    return results_recall, results_activations, partition_counts, query_search_times


def compute_recall_hnswlib_bf(
    base_mmap,
    gt_file,
    centers_dir,
    partition_dir,
    centersStepNum,
    partitionStepNum,
    partitions_to_search,
    k_neighbors,
    numPartitions,
):
    """Compute recall using hnswlib.BFIndex (exact brute-force) for nearest-center routing."""
    ground_truth = read_fbin_ground_truth(gt_file)
    num_queries = len(ground_truth)

    # Load cluster centers and build hnswlib BFIndex
    centers_file = f"{centers_dir}/step_{centersStepNum:06d}_centers.csv"
    centers = np.loadtxt(centers_file, delimiter=",", dtype=np.float32)
    centers = centers.reshape(-1, 100)
    num_centers = centers.shape[0]

    router = hnswlib.BFIndex(space='l2', dim=100)
    router.init_index(max_elements=num_centers)
    router.add_items(centers, np.arange(num_centers))

    # Load partition assignments: maps center index -> partition ID
    partition_file = f"{partition_dir}/step_{partitionStepNum:06d}_partitions.csv"
    partitions = np.loadtxt(partition_file, delimiter=",", dtype=int)

    # flatten gt indices
    gt_indices = np.empty(num_queries * k_neighbors, dtype=np.int64)
    pos = 0
    for gt_list in ground_truth:
        gt_indices[pos:pos + k_neighbors] = gt_list[:k_neighbors]
        pos += k_neighbors

    # assign each GT vector to its nearest center via BFIndex
    vecs = np.ascontiguousarray(base_mmap[gt_indices]).astype(np.float32)  # (Q*k, dim)
    labels_gt, _ = router.knn_query(vecs, k=1)          # (Q*k, 1)
    gt_part_ids = partitions[labels_gt[:, 0]]           # (Q*k,)

    ground_truth_partitions = {}
    pos = 0
    for i in range(num_queries):
        ground_truth_partitions[i] = gt_part_ids[pos:pos + k_neighbors].tolist()
        pos += k_neighbors

    results_recall = []
    results_activations = []
    partition_counts = []
    query_search_times = []

    for p in partitions_to_search:
        k = min(p, num_centers)
        t0 = time.perf_counter()
        labels, _ = router.knn_query(queries, k=k)      # (Q, k)
        query_search_times.append(time.perf_counter() - t0)

        visited = {}
        activation_rates = 0.0
        per_partition_counts = np.zeros((numPartitions,), dtype=np.int64)

        for i, label_list in enumerate(labels):
            visited_partitions = set()
            for label in label_list:
                partition_id = partitions[label]
                visited_partitions.add(partition_id)
                per_partition_counts[partition_id] += 1
            visited[i] = visited_partitions
            activation_rates += len(visited_partitions) / numPartitions

        total_recall = 0.0
        for i in range(num_queries):
            gt_parts = ground_truth_partitions[i]
            visited_parts = visited[i]
            hits = sum(1 for part in gt_parts if part in visited_parts)
            total_recall += hits / len(gt_parts)

        average_recall = total_recall / num_queries
        results_recall.append(average_recall)
        results_activations.append(activation_rates / num_queries)
        partition_counts.append(per_partition_counts)

    return results_recall, results_activations, partition_counts, query_search_times


if __name__ == "__main__":
    args = parse_args()

    deltas = parse_runbook(args.runbook_path)
    counts = [deltas[s] for s in sorted(deltas)]
    vector_count_cumulative = np.cumsum(counts)

    if args.base_file.endswith(".u8bin"):
        queries = read_u8bin_slice(args.query_file, 0, 10000)
        base_mmap = mmap_u8bin(args.base_file)
    else:
        queries = read_fbin_slice(args.query_file, 0, 10000)
        base_mmap = mmap_fbin(args.base_file)

    base_directory = args.base_dir
    partitions_directory = args.partitions_dir
    BRANCHING_FACTORS = args.branching_factors

    recalls = {}
    activations = {}
    imbalance = {}
    query_times = {}

    for bf in BRANCHING_FACTORS:
        recalls[bf] = []
        activations[bf] = []
        imbalance[bf] = []
        query_times[bf] = []

    step_start = find_first_available_step(
        base_directory,
        partitions_directory,
        use_centers=(args.brute_force or args.brute_force_enforce_p or args.hnswlib_bf),
    )
    print(f"Starting step (auto-detected from directory contents): {step_start}")
    runbook_steps = sorted(deltas)
    # find step end by looking for the first missing step file in the directory
    # (used only when rebuild statistics are requested)
    step_end = step_start
    if not args.no_rebuilds:
        while True:
            if args.brute_force or args.brute_force_enforce_p or args.hnswlib_bf:
                step_file = f"{base_directory}/step_{step_end:06d}_centers.csv"
            else:
                step_file = f"{base_directory}/step_{step_end:06d}_hnsw.bin"
            if not Path(step_file).exists():
                break
            step_end += 2

    first_partition_file = f"{partitions_directory}/step_{step_start:06d}_partitions.csv"
    first_partitions = np.loadtxt(first_partition_file, delimiter=",", dtype=int)
    num_partitions = int(first_partitions.max()) + 1
    print(f"Found {num_partitions} partitions")

    processed_steps = []
    if not args.no_rebuilds:
        for step in range(step_start, step_end, 2):
            gt_test = f"{args.gt_dir}/step{step+1}.gt100"
            hnsw_file = f"{base_directory}/step_{step:06d}_hnsw.bin"
            centers_file = f"{base_directory}/step_{step:06d}_centers.csv" if (args.brute_force or args.brute_force_enforce_p or args.hnswlib_bf) else None
            partition_file = f"{partitions_directory}/step_{step:06d}_partitions.csv"
            required_files: list[str] = [partition_file, gt_test,
                                        centers_file if centers_file is not None else hnsw_file]
            if not all(Path(f).exists() for f in required_files):
                print(f"Missing file(s) at step {step}: {', '.join(f for f in required_files if not Path(f).exists())}. Writing results up to previous step.")
                break
            if args.hnswlib_bf:
                recall, activation_rates, per_partition_counts, search_times = compute_recall_hnswlib_bf(
                    base_mmap,
                    gt_test,
                    base_directory,
                    partitions_directory,
                    step,
                    step,
                    partitions_to_search=BRANCHING_FACTORS,
                    k_neighbors=10,
                    numPartitions=num_partitions
                )
            elif args.enforce_p:
                recall, activation_rates, per_partition_counts, search_times = compute_recall_enforce_p(
                    base_mmap,
                    gt_test,
                    base_directory,
                    partitions_directory,
                    step,
                    step,
                    partitions_to_search=BRANCHING_FACTORS,
                    k_neighbors=10,
                    numPartitions=num_partitions
                )
            else:
                recall, activation_rates, per_partition_counts, search_times = compute_recall(
                    base_mmap,
                    gt_test,
                    base_directory,
                    partitions_directory,
                    step,
                    step,
                    branchingFactors=BRANCHING_FACTORS,
                    k_neighbors=10,
                    numPartitions=num_partitions
                )

            for i, bf in enumerate(BRANCHING_FACTORS):
                recalls[bf].append(recall[i])
                activations[bf].append(activation_rates[i])
                imbalance[bf].append(per_partition_counts[i])
                query_times[bf].append(search_times[i])
            processed_steps.append(step)
            print(f"REBUILDING: Step {step}: \n\tRecall@10 = {recall}, \n\tActivation = {activation_rates}")
    else:
        step1_router_or_centers = (
            f"{base_directory}/step_{step_start:06d}_centers.csv"
            if (args.brute_force or args.brute_force_enforce_p or args.hnswlib_bf)
            else f"{base_directory}/step_{step_start:06d}_hnsw.bin"
        )
        step1_partition = f"{partitions_directory}/step_{step_start:06d}_partitions.csv"
        required_no_rebuild_files = [step1_router_or_centers, step1_partition]
        if not all(Path(f).exists() for f in required_no_rebuild_files):
            missing = ", ".join(f for f in required_no_rebuild_files if not Path(f).exists())
            raise FileNotFoundError(
                f"Missing required step-1 file(s) for --no-rebuilds mode: {missing}"
            )

        # In no-rebuild mode, evaluate on the same step cadence as rebuild mode
        # (e.g., 1, 3, 5, ... if step_start==1; 3, 5, 7, ... if step_start==3),
        # since GT files are commonly generated every other step.  Also require
        # s >= step_start so we never try to evaluate a step that predates the
        # files we have on disk — Python's % is sign-preserving, so the parity
        # filter alone is not enough: (1 - 3) % 2 == 0.
        no_rebuild_steps = [
            s for s in runbook_steps
            if s >= step_start and (s - step_start) % 2 == 0
        ]

        for step in no_rebuild_steps:
            gt_test = f"{args.gt_dir}/step{step+1}.gt100"
            if not Path(gt_test).exists():
                print(f"Missing ground truth file at step {step}: {gt_test}. Skipping this step.")
                continue
            processed_steps.append(step)

    # never rebuilding
    never_rebuilding_recalls = {}
    never_rebuilding_activations = {}
    never_rebuilding_imbalance = {}
    never_rebuilding_query_times = {}

    for bf in BRANCHING_FACTORS:
        never_rebuilding_recalls[bf] = []
        never_rebuilding_activations[bf] = []
        never_rebuilding_imbalance[bf] = []
        never_rebuilding_query_times[bf] = []

    for step in processed_steps:
        # recall, activation__rate = compute_recall(directory, step, 1, 1, 5)
        gt_test = f"{args.gt_dir}/step{step+1}.gt100"
        if args.hnswlib_bf:
            recall, activation_rates, per_partition_counts, search_times = compute_recall_hnswlib_bf(
                base_mmap,
                gt_test,
                base_directory,
                partitions_directory,
                step_start,
                step_start,
                partitions_to_search=BRANCHING_FACTORS,
                k_neighbors=10,
                numPartitions=num_partitions
            )
        elif args.enforce_p:
            recall, activation_rates, per_partition_counts, search_times = compute_recall_enforce_p(
                base_mmap,
                gt_test,
                base_directory,
                partitions_directory,
                step_start,
                step_start,
                partitions_to_search=BRANCHING_FACTORS,
                k_neighbors=10,
                numPartitions=num_partitions
            )
        else:
            recall, activation_rates, per_partition_counts, search_times = compute_recall(
                base_mmap,
                gt_test,
                base_directory,
                partitions_directory,
                step_start,
                step_start,
                branchingFactors=BRANCHING_FACTORS,
                k_neighbors=10,
                numPartitions=num_partitions
            )
        for i, bf in enumerate(BRANCHING_FACTORS):
            never_rebuilding_recalls[bf].append(recall[i])
            never_rebuilding_activations[bf].append(activation_rates[i])
            never_rebuilding_imbalance[bf].append(per_partition_counts[i])
            never_rebuilding_query_times[bf].append(search_times[i])

        # recall, activation__rate = compute_recall(directory, step, step-125, step-125, 5)
        print(f"NO REBUILDING: Step {step}: \n\tRecall@10 = {recall}, \n\tActivation = {activation_rates}")

    cof = []
    cof_never_rebuilding = []

    # coefficient of variance
    first_bf = BRANCHING_FACTORS[0]
    if not args.no_rebuilds:
        for step in range(0, len(imbalance[first_bf])):
            mean = np.mean(imbalance[first_bf][step])
            std = np.std(imbalance[first_bf][step])
            var = std/mean
            cof.append(var)

    for step in range(0, len(never_rebuilding_imbalance[first_bf])):
        mean = np.mean(never_rebuilding_imbalance[first_bf][step])
        std = np.std(never_rebuilding_imbalance[first_bf][step])
        var = std/mean
        cof_never_rebuilding.append(var)

    # create table and save to partitions directory
    if args.no_rebuilds:
        table = pd.DataFrame({
            "step": processed_steps,
            "cof_no_rebuilds": cof_never_rebuilding
        })
        for bf in BRANCHING_FACTORS:
            table[f"recall_no_rebuilds_bf_{bf}"] = never_rebuilding_recalls[bf]
            table[f"activation_no_rebuilds_bf_{bf}"] = never_rebuilding_activations[bf]
            table[f"imbalance_no_rebuilds_bf_{bf}"] = never_rebuilding_imbalance[bf]
            table[f"query_time_s_no_rebuilds_bf_{bf}"] = never_rebuilding_query_times[bf]
    else:
        table = pd.DataFrame({
            "step": processed_steps,
            "cof": cof,
            "cof_no_rebuilds": cof_never_rebuilding
        })

        for bf in BRANCHING_FACTORS:
            table[f"recall_bf_{bf}"] = recalls[bf]
            table[f"activation_bf_{bf}"] = activations[bf]
            table[f"imbalance_bf_{bf}"] = imbalance[bf]
            table[f"query_time_s_bf_{bf}"] = query_times[bf]
            table[f"recall_no_rebuilds_bf_{bf}"] = never_rebuilding_recalls[bf]
            table[f"activation_no_rebuilds_bf_{bf}"] = never_rebuilding_activations[bf]
            table[f"imbalance_no_rebuilds_bf_{bf}"] = never_rebuilding_imbalance[bf]
            table[f"query_time_s_no_rebuilds_bf_{bf}"] = never_rebuilding_query_times[bf]

    if args.no_rebuilds:
        if args.brute_force_enforce_p:
            output_name = "full_results_brute_force_enforce_p_no_rebuilds.csv"
        elif args.hnswlib_bf:
            output_name = "full_results_hnswlib_bf_no_rebuilds.csv"
        elif args.brute_force:
            output_name = "full_results_brute_force_no_rebuilds.csv"
        elif args.enforce_p:
            output_name = "full_results_enforced_p_no_rebuilds.csv"
        else:
            output_name = "full_results_no_rebuilds.csv"
    else:
        if args.brute_force_enforce_p:
            output_name = "full_results_brute_force_enforce_p.csv"
        elif args.hnswlib_bf:
            output_name = "full_results_hnswlib_bf.csv"
        elif args.brute_force:
            output_name = "full_results_brute_force.csv"
        elif args.enforce_p:
            output_name = "full_results_enforced_p.csv"
        else:
            output_name = "full_results.csv"
    table.to_csv(f"{partitions_directory}/{output_name}", index=False)


    # python compute_theoretical_recall.py --base-dir {} --partitions-dir {} --branching-factors {} --enforce-p
    # p5 - 1 2 3 4 5 10 15 
    #       1 2 5 7 8 10 15
    # p10 - 1 2 5 7 8 10 15 20 25
    # p20 - 1 2 5 10 15 20 25 40 50 60
