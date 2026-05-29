import glob
import os
import re
import yaml
import hnswlib
import numpy as np
import multiprocessing
import sys
from collections import defaultdict

def detect_last_step(base_dir):
    pattern = os.path.join(base_dir, "step_*_hnsw.bin")
    matches = glob.glob(pattern)
    if not matches:
        print(f"Error: no step_*_hnsw.bin files found in {base_dir}")
        sys.exit(1)
    step_re = re.compile(r"step_(\d+)_hnsw\.bin$")
    steps = [int(step_re.search(os.path.basename(p)).group(1)) for p in matches]
    return max(steps)


def _safe_memmap(filename, dtype, bytes_per_element):
    file_size = os.path.getsize(filename)
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
    num_points_hdr, dim = int(header[0]), int(header[1])
    num_points = (file_size - 8) // (dim * bytes_per_element)
    if num_points != num_points_hdr:
        print(f"Warning: {os.path.basename(filename)} header says {num_points_hdr} "
              f"points but file size implies {num_points}; using {num_points}.")
    return np.memmap(filename, dtype=dtype, mode="r",
                     offset=8, shape=(num_points, dim))

def mmap_fbin(filename):
    return _safe_memmap(filename, np.float32, 4)

def mmap_u8bin(filename):
    return _safe_memmap(filename, np.uint8, 1)

def open_vectors(filename):
    if filename.endswith(".u8bin"):
        return mmap_u8bin(filename)
    return mmap_fbin(filename)

def get_vectors_at_step(runbook_data, stepNum):
    # Top-level key (e.g., "msturing-30M-clustered")
    active_vectors = set()
    root = next(iter(runbook_data.values()))

    for step, payload in root.items():
        # Skip non-step metadata (e.g., max_pts)
        if not isinstance(step, int):
            continue

        if int(step) > stepNum:
            return list(active_vectors)

        op = payload.get("operation")
        start = payload.get("start")
        end = payload.get("end")

        if op in {"delete", "insert"} and start is not None and end is not None:

            if op == "insert":
                for i in range(start,end):
                    active_vectors.add(i)
            if op == 'delete':
                for i in range(start,end):
                    active_vectors.remove(i)   
    return list(active_vectors)
# read runbook
def process_step(args):
    stepNum, lock, output_file, runbook_data, base_file, base_dir, partitions_dir, to_rebuild = args
    print(f"evaluating step {stepNum}")

    if to_rebuild:
        rebuild_stepNum = stepNum
    else:
        rebuild_stepNum = 1 # TODO: change if rebalancing
    hnsw_file = f"{base_dir}/step_{rebuild_stepNum:06d}_hnsw.bin"
    partitions_file = f"{partitions_dir}/step_{rebuild_stepNum:06d}_partitions.csv"
    get_partition = np.loadtxt(partitions_file, dtype=np.int64)[:10000]

    # print("shape:", get_partition.shape)
    # print("dtype:", get_partition.dtype)
    # print("max:", get_partition.max())
    # print("len:", len(get_partition))
    # bad = np.where(get_partition > 9)[0]
    # print("bad indices:", bad[:10])

    router = hnswlib.Index(space="l2", dim=100)
    router.load_index(hnsw_file)
    router.set_ef(100)

    base_mm = open_vectors(base_file)

    step_idx = get_vectors_at_step(runbook_data, stepNum)
    vectors = np.asarray(base_mm[step_idx], dtype=np.float32)

    # route vectors through hnsw and compute imbalance
    labels, distances = router.knn_query(vectors, k=1, num_threads=8)

    # partition_counts = np.zeros(get_partition.max() + 1, dtype=int)
    # partition_sums = np.zeros(get_partition.max() + 1, dtype=float)

    num_partitions = get_partition.max() + 1
    print(f"step {stepNum}, num partitions: {num_partitions}")
    dim = vectors.shape[1]

    partition_counts = np.zeros(num_partitions, dtype=np.int64)

    partition_sums = np.zeros(
        (num_partitions, dim),
        dtype=np.float64
    )

    # for entropy
    partition_label_counts = defaultdict(
        lambda: defaultdict(int)
    )

    for i, label in enumerate(labels):
        partition = get_partition[label[0]]
        vec = vectors[i]

        partition_counts[partition] += 1
        partition_sums[partition] += vec

        partition_label_counts[partition][label[0]] += 1

    partition_means = (
        partition_sums /
        partition_counts[:, None]
    )

    partition_dispersions = np.zeros(num_partitions, dtype=np.float64)

    for i, label in enumerate(labels):
        partition = get_partition[label[0]]

        vec = vectors[i]

        diff = vec - partition_means[partition]
        partition_dispersions[partition] += diff @ diff

    partition_dispersions /= (partition_counts * dim)

    partition_entropy = np.zeros(
        num_partitions,
        dtype=np.float64
    )

    for partition, label_counts in partition_label_counts.items():
        total = sum(label_counts.values())
        entropy = 0.0

        for count in label_counts.values():
            p = count / total
            entropy -= p * np.log(p)

        if len(label_counts) > 1:
            entropy /= np.log(len(label_counts))

        partition_entropy[partition] = entropy

    with lock:
        with open(output_file, "a") as f:

            for partition in range(num_partitions):

                f.write(
                    f"{stepNum},"
                    f"{partition},"
                    f"{partition_counts[partition]},"
                    f"{partition_dispersions[partition]},"
                    f"{partition_entropy[partition]}\n"
                )

            f.flush()

def main():
    if len(sys.argv) != 6:
        print("Usage: python compute_real_imbalance.py <base_dir> <partitions_dir> <runbook_path> <base_file> <to_rebuild>")
        sys.exit(1)

    base_dir = sys.argv[1]
    partitions_dir = sys.argv[2]
    runbook_path = sys.argv[3]
    base_file = sys.argv[4]
    to_rebuild = sys.argv[5].lower() == "true"

    with open(runbook_path) as f:
        runbook_data = yaml.safe_load(f)

    if (to_rebuild):
        output_file = f"{partitions_dir}/full_real_imbalance_rebuilding_results.csv"
    else:
        output_file = f"{partitions_dir}/full_real_imbalance_no_rebuilding_results.csv"
    
    # Write header if file is empty
    with open(output_file, "a") as f:
        if f.tell() == 0:
            f.write("stepNum,partition,count,dispersion,entropy\n")

    # Set up multiprocessing with a shared lock
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    # Detect step range from directory
    last_step = detect_last_step(base_dir)
    print(f"Last step detected: {last_step}")

    # Parallelize the steps
    with multiprocessing.Pool(processes=8) as pool:
        args_list = [(stepNum, lock, output_file, runbook_data, base_file, base_dir, partitions_dir, to_rebuild) for stepNum in range(1, last_step + 1, 2)]
        pool.map(process_step, args_list)

if __name__ == "__main__":
    main()