import argparse
import numpy as np
import struct
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from pathlib import Path
import hnswlib

def read_fbin_slice(filename, start, end):
    with open(filename, "rb") as f:
        header = f.read(8)
        num_points, dim = struct.unpack("II", header)

        if start < 0 or end > num_points:
            raise ValueError("Slice out of bounds")

        count = end - start
        vec_size = dim * 4
        f.seek(8 + start * vec_size, 0)
        data = np.fromfile(f, dtype=np.float32, count=count * dim)
        return data.reshape(count, dim)


def persist_centers_csv(
    out_dir,
    step,
    centers,
    counts
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save centers
    centers_path = out_dir / f"step_{step:06d}_centers.csv"
    np.savetxt(centers_path, centers.astype(np.float32), delimiter=",")

    counts_path = out_dir / f"step_{step:06d}_center_counts.csv"
    np.savetxt(counts_path, counts.astype(np.int32), delimiter=",")

    # # Optionally save spreads if you need them
    # spreads_path = out_dir / f"clusters_step_{step:06d}_spreads.csv"
    # np.savetxt(spreads_path, spreads.astype(np.float32), delimiter=",")

    # # Save metadata (step, operation, inertia, num_points) as a small text file
    # meta_path = out_dir / f"clusters_step_{step:06d}_meta.txt"
    # with open(meta_path, "w") as f:
    #     f.write(f"step,{step}\n")
    #     f.write(f"operation,{operation}\n")
    #     f.write(f"inertia,{inertia}\n")
    #     f.write(f"num_points,{num_points}\n")

    # print(f"Saved CSVs for step {step} to {out_dir}")


def load_runbook(path, dataset_key):
    with open(path, "r") as f:
        rb = yaml.safe_load(f)
    steps = rb[dataset_key]
    # filter numeric steps, sorted
    ordered = [
        steps[k] for k in sorted(k for k in steps if isinstance(k, int))
    ]
    return ordered

def replay_runbook_delete_from_initial_hnsw(
    runbook_steps,
    vector_path,
    out_dir,
    k=1000,
    ef_construction=200,
    M=16,
    ef_search=200,
):
    kmeans = None
    dim = 100

    cluster_sums = np.zeros((k, dim), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)

    point_to_centroid = {}  # global_id -> centroid_idx
    hnsw_index = None

    def _build_hnsw(centers):
        """Build a fresh HNSW index on the current centers (float32)."""
        idx = hnswlib.Index(space='l2', dim=centers.shape[1])
        idx.init_index(max_elements=centers.shape[0], ef_construction=ef_construction, M=M)
        idx.set_ef(ef_search)
        idx.add_items(centers.astype(np.float32), np.arange(centers.shape[0]))
        return idx

    for step_id, step in enumerate(runbook_steps, 1):
        print(
            f"Step {step_id:03d} | "
            f"step={step}"
        )
        op = step["operation"]

        new_vecs = None
        del_vecs = None
        indices = None

        if op == "insert":
            start, end = step["start"], step["end"]
            if step_id == 1:
                new_vecs = read_fbin_slice(vector_path, start, end)
                indices = np.arange(start, end, dtype=np.uint32)
            else:
                new_vecs = read_fbin_slice(vector_path, start, end)

        elif op == "delete":
            start, end = step["start"], step["end"]
            del_vecs = read_fbin_slice(vector_path, start, end)

        else:
            continue

        if kmeans is None:
            # Initial clustering
            print("sampling")
            sample_size = min(100000, new_vecs.shape[0])
            idx = np.random.choice(new_vecs.shape[0], size=sample_size, replace=False)
            X_sample = new_vecs[idx]
            print("kmeans start")
            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
            )
            kmeans.fit(X_sample)
            print("kmeans done")

            # Build HNSW on initial kmeans centers
            hnsw_index = _build_hnsw(kmeans.cluster_centers_)

            counts = np.zeros(k, dtype=int)

            # Assign all vectors via HNSW
            labels_2d, _ = hnsw_index.knn_query(new_vecs.astype(np.float32), k=1)
            labels = labels_2d.ravel()

            for it, global_id in enumerate(indices):
                lbl = labels[it]
                vec = new_vecs[it]

                point_to_centroid[global_id] = lbl

                cluster_sums[lbl] += vec
                counts[lbl] += 1

            kmeans.cluster_centers_ = cluster_sums / np.maximum(counts, 1)[:, None]

            # Rebuild HNSW on updated centers
            hnsw_index = _build_hnsw(kmeans.cluster_centers_)

        else:
            if op == "insert":
                # Assign via HNSW in batches
                batch_size = 100000
                labels = np.empty(new_vecs.shape[0], dtype=np.int32)

                for start_idx in range(0, new_vecs.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, new_vecs.shape[0])
                    batch = new_vecs[start_idx:end_idx].astype(np.float32)
                    batch_labels, _ = hnsw_index.knn_query(batch, k=1)
                    labels[start_idx:end_idx] = batch_labels.ravel()

                for i, lbl in enumerate(labels):
                    pid = start + i
                    point_to_centroid[pid] = lbl

                sums_new = np.zeros_like(kmeans.cluster_centers_)
                counts_new = np.zeros(k, dtype=int)

                for c in range(k):
                    members = new_vecs[labels == c]
                    if len(members) > 0:
                        sums_new[c] = members.sum(axis=0)
                        counts_new[c] = len(members)

                cluster_sums += sums_new
                counts += counts_new

                kmeans.cluster_centers_ = cluster_sums / np.maximum(counts, 1)[:, None]

                # Rebuild HNSW on updated centers
                hnsw_index = _build_hnsw(kmeans.cluster_centers_)

            elif op == "delete":
                for i in range(del_vecs.shape[0]):
                    pid = start + i
                    vec = del_vecs[i]

                    lbl = point_to_centroid.get(pid)
                    if lbl is None:
                        raise RuntimeError(f"Point {pid} not found in active set during deletion")

                    if counts[lbl] < 0:
                        raise RuntimeError(f"Cluster {lbl} has negative count during deletion")

                    cluster_sums[lbl] -= vec
                    counts[lbl] -= 1
                    point_to_centroid.pop(pid)

                kmeans.cluster_centers_ = cluster_sums / np.maximum(counts, 1)[:, None]

                # Rebuild HNSW on updated centers
                hnsw_index = _build_hnsw(kmeans.cluster_centers_)

        persist_centers_csv(
            out_dir=out_dir,
            step=step_id,
            centers=kmeans.cluster_centers_,
            counts=counts,
        )
        np.save(Path(out_dir) / "point_to_centroid.npy", point_to_centroid)

        # Save a step-specific snapshot after step 1 so that the C++ pipeline
        # can load it via --load-state-assignments to bypass the initial
        # assignment and start from an identical state.
        if step_id == 1:
            np.save(Path(out_dir) / "point_to_centroid_step1.npy", point_to_centroid)
            step1_csv = Path(out_dir) / "point_to_centroid_step1.csv"
            with open(step1_csv, "w") as _f:
                for _gid, _cid in point_to_centroid.items():
                    _f.write(f"{_gid},{_cid}\n")
            print(f"  Saved step-1 assignments ({len(point_to_centroid)} entries) to {step1_csv}")

    return kmeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MSTuring cluster analysis")
    parser.add_argument(
        "--runbook",
        type=str,
        required=True,
        help="Path to the runbook YAML file",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        required=True,
        help="Top-level key in the runbook YAML for the dataset (e.g. msturing-30M-clustered)",
    )
    parser.add_argument(
        "--vector-path",
        type=str,
        required=True,
        help="Path to the main vector .fbin file",
    )
    parser.add_argument(
        "--centers",
        type=int,
    )
    args = parser.parse_args()

    steps = load_runbook(args.runbook, args.dataset_key)

    out_dir = f"cluster_history_{args.dataset_key}_{args.centers}"

    replay_runbook_delete_from_initial_hnsw(
        steps,
        args.vector_path,
        out_dir,
        args.centers,
    )


# python msturing-cluster-analysis.py \
#   --runbook /dataset/vectorDB/final_runbook_256.yaml \
#   --dataset-key msturing-30M-clustered \
#   --vector-path /dataset/big-ann-benchmarks/data/MSTuring-30M-clustered/30M-clustered64.fbin \
#   --centers 10000