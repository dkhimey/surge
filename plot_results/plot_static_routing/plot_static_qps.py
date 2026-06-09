#!/usr/bin/env python3
"""
plot_qps_combined.py

Combined end-to-end throughput figure for the SURGE paper (former Figure 6).
Plots all four datasets side by side, each as a recall@10-vs-QPS panel (log-y)
above a "partitions activated" strip, with ONE shared legend across the top:

    columns = datasets   (SIFT-100M, MSTuring-100M, SIFT-500M, MSTuring-500M)
    per col = QPS (log, 3 units)  over  Partitions Activated (1 unit)

Series (same in every panel):
    SURGE (NProbe) | SURGE (Routing Factor) | SURGE (Recall Target) | GP-ANN

Data (per dataset):
    SURGE  shared_static_experiment_<key>_results_newer_unoptimized.csv
           columns: mode, recall@10, qps, avg_parts_searched
           mode in {NProbe, BranchingFactor, RecallTarget}
    GP-ANN distributed_bench_results_<key>_withrouting_newer.csv   (100M only)
           columns: recall@10, qps, nprobe

Usage:
    python plot_qps_combined.py -o qps_combined.pdf
    python plot_qps_combined.py --selftest -o demo.png      # synthetic data
"""

import argparse
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def set_style():
    # Single row of four dataset panels spanning the full text width (figure*).
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.markersize": 4,
        "grid.color": "lightgray",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


# Single source of truth for every series' style + label (colors match the
# static-routing figure so the two share a palette across the paper).
STYLE = {
    "surge_nprobe": dict(color="C0", marker="o", ls="-", label="SURGE (NProbe)"),
    "surge_bf":     dict(color="C1", marker="s", ls="-", label="SURGE (Routing Factor)"),
    "surge_rt":     dict(color="C2", marker="^", ls="-", label="SURGE (Recall Target)"),
    "gpann":        dict(color="C8", marker="d", ls="-", label="GP-ANN"),
}
LEGEND_ORDER = ["surge_nprobe", "surge_bf", "surge_rt", "gpann"]

# routing_metrics mode string -> STYLE key
SURGE_MODES = [("NProbe", "surge_nprobe"),
               ("BranchingFactor", "surge_bf"),
               ("RecallTarget", "surge_rt")]


# --------------------------------------------------------------------------- #
# Dataset configuration
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    title: str
    surge_csv: str
    gpann_csv: Optional[str] = None      # None to skip GP-ANN (e.g. 500M)
    gpann_parts_col: str = "nprobe"      # GP-ANN partitions-probed column
    n_partitions: int = 10               # for converting counts -> activation fraction


_SROOT = "/dataset/surge/results/static_data_throughput"
_GP_SIFT = "/dataset/big-ann-benchmarks/data/bigann/gpann_partitions"
_GP_MST = "/dataset/big-ann-benchmarks/data/MSTuringANNS/gpann_partitions"

# >>> EDIT THESE PATHS <<<  (defaults copied from plot_figure6)
DATASETS = [
    DatasetSpec(
        title="SIFT-100M",
        surge_csv=f"{_SROOT}/shared_static_experiment_sift100M_results_newer_unoptimized.csv",
        gpann_csv=f"{_GP_SIFT}/distributed_bench_results_bigann100M_withrouting_newer.csv",
    ),
    DatasetSpec(
        title="MSTuring-100M",
        surge_csv=f"{_SROOT}/shared_static_experiment_msturing100M_results_newer_unoptimized.csv",
        gpann_csv=f"{_GP_MST}/distributed_bench_results_msturing100M_withrouting_newer.csv",
    ),
    DatasetSpec(
        title="SIFT-500M",
        surge_csv=f"{_SROOT}/shared_static_experiment_sift500M_results_newer_unoptimized.csv",
        gpann_csv=None,
    ),
    DatasetSpec(
        title="MSTuring-500M",
        surge_csv=f"{_SROOT}/shared_static_experiment_msturing500M_results_newer_unoptimized.csv",
        gpann_csv=None,
    ),
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dataset(ds: DatasetSpec) -> dict:
    return {
        "surge": pd.read_csv(ds.surge_csv),
        "gpann": pd.read_csv(ds.gpann_csv) if (ds.gpann_csv and os.path.exists(ds.gpann_csv)) else None,
    }


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _line(ax, x, y, key):
    s = {k: v for k, v in STYLE[key].items() if k != "label"}
    ax.plot(x, y, **s)


def plot_dataset(ax_top, ax_bot, ds, d):
    surge = d["surge"]
    p = float(ds.n_partitions)  # divide partition counts -> activation fraction
    for mode_name, key in SURGE_MODES:
        sub = surge[surge["mode"] == mode_name].sort_values("recall@10")
        _line(ax_top, sub["recall@10"], sub["qps"], key)
        _line(ax_bot, sub["recall@10"], sub["avg_parts_searched"] / p, key)
    if d["gpann"] is not None:
        g = d["gpann"].sort_values("recall@10")
        _line(ax_top, g["recall@10"], g["qps"], "gpann")
        _line(ax_bot, g["recall@10"], g[ds.gpann_parts_col] / p, "gpann")
    ax_top.set_yscale("log")
    top_ylim = ax_top.get_ylim()
    if top_ylim[0] < 2e4:
        ax_top.set_ylim(1.7e4, top_ylim[1])  # zoom in
    ax_bot.set_ylim(0, 1.05)
    ax_top.grid(True, which="both")
    ax_bot.grid(True, which="both")
    ax_bot.set_xlabel("Recall@10")


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def build_figure(datasets, data, grid_ncols=4):
    n = len(datasets)
    grid_nrows = (n + grid_ncols - 1) // grid_ncols
    # Full \textwidth, one row of dataset panels -- matches the static-routing
    # quality figure layout (the four datasets across columns).
    fig = plt.figure(figsize=(3.5 * grid_ncols, 3 * grid_nrows))
    outer = fig.add_gridspec(grid_nrows, grid_ncols, hspace=0.25, wspace=0.27)

    for i, (ds, d) in enumerate(zip(datasets, data)):
        gr, gc = divmod(i, grid_ncols)
        sub = outer[gr, gc].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax_top = fig.add_subplot(sub[0])
        ax_bot = fig.add_subplot(sub[1], sharex=ax_top)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plot_dataset(ax_top, ax_bot, ds, d)
        ax_top.set_title(ds.title, fontweight="bold")
        if gc == 0:                       # y-labels on the left column only
            ax_top.set_ylabel("Throughput (QPS)")
            ax_bot.set_ylabel("Partition\nActivation")
        if gr < grid_nrows - 1:           # x-label only on the bottom grid row
            ax_bot.set_xlabel("")

    # --- single shared legend across the TOP (all entries in one row) ---
    handles = [Line2D([0], [0], **{k: v for k, v in STYLE[key].items() if k != "label"})
               for key in LEGEND_ORDER]
    labels = [STYLE[key]["label"] for key in LEGEND_ORDER]
    fig.legend(handles, labels, loc="lower center", ncol=len(LEGEND_ORDER), frameon=False,
               bbox_to_anchor=(0.5, 0.95), columnspacing=1.6, handletextpad=0.5)
    return fig


# --------------------------------------------------------------------------- #
# Self-test: synthesize schema-matching CSVs so the layout can be checked
# without the cluster's /dataset files.
# --------------------------------------------------------------------------- #
def _write_synthetic(surge_path, gpann_path, seed=0, qps_scale=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for mode in ("NProbe", "BranchingFactor", "RecallTarget"):
        rec = np.linspace(0.80, 0.99, 8)
        parts = np.linspace(1, 9, 8)                       # partitions activated
        qps = qps_scale * 1e5 / (parts ** 1.8) * (1 + rng.normal(0, 0.02, 8))
        rows.append(pd.DataFrame({"mode": mode, "recall@10": rec,
                                  "qps": qps, "avg_parts_searched": parts}))
    pd.concat(rows, ignore_index=True).to_csv(surge_path, index=False)

    if gpann_path is not None:
        rec = np.linspace(0.80, 0.98, 7)
        npb = np.linspace(1, 9, 7)
        qps = qps_scale * 0.8e5 / (npb ** 1.8)
        pd.DataFrame({"recall@10": rec, "qps": qps, "nprobe": npb}).to_csv(gpann_path, index=False)


def make_selftest_datasets(tmp):
    demo = [("SIFT-100M (demo)", True, 1.0),
            ("MSTuring-100M (demo)", True, 0.9),
            ("SIFT-500M (demo)", False, 0.25),
            ("MSTuring-500M (demo)", False, 0.22)]
    specs = []
    for i, (title, has_gp, scale) in enumerate(demo):
        spath = os.path.join(tmp, f"surge{i}.csv")
        gpath = os.path.join(tmp, f"gpann{i}.csv") if has_gp else None
        _write_synthetic(spath, gpath, seed=i, qps_scale=scale)
        specs.append(DatasetSpec(title=title, surge_csv=spath, gpann_csv=gpath))
    return specs


# --------------------------------------------------------------------------- #
# TODO extraction
# --------------------------------------------------------------------------- #
def compute_qps_ratios(ds: DatasetSpec, d: dict) -> None:
    """
    Print the GP-ANN / SURGE QPS ratio at equal recall for the paper TODOs:
      'GP-ANN delivers X× higher QPS than SURGE on SIFT-100M'
      'GP-ANN delivers X× higher QPS than SURGE on MSTuring-100M'

    For each GP-ANN recall point, the best SURGE QPS across all routing modes
    is found by linear interpolation, then the ratio is computed.
    Only runs for datasets that have GP-ANN data (i.e. the 100M datasets).
    """
    if d["gpann"] is None:
        return

    surge = d["surge"]
    gpann = d["gpann"].sort_values("recall@10")

    ratios = []
    for _, row in gpann.iterrows():
        r = row["recall@10"]
        best_surge_qps = 0.0
        for mode_name, _ in SURGE_MODES:
            sub = surge[surge["mode"] == mode_name].sort_values("recall@10")
            if len(sub) < 2:
                continue
            if r < sub["recall@10"].min() or r > sub["recall@10"].max():
                continue
            qps_at_r = float(np.interp(r, sub["recall@10"].values, sub["qps"].values))
            best_surge_qps = max(best_surge_qps, qps_at_r)

        if best_surge_qps > 0:
            ratios.append((r, row["qps"] / best_surge_qps))

    if not ratios:
        print(f"[TODO] {ds.title}: no overlapping recall range between GP-ANN and SURGE.")
        return

    recall_vals = [r for r, _ in ratios]
    ratio_vals  = [v for _, v in ratios]

    print(f"\n[TODO] {ds.title}: GP-ANN / best-SURGE QPS ratio at equal recall")
    print(f"  Recall range : [{min(recall_vals):.3f}, {max(recall_vals):.3f}]")
    print(f"  Ratio range  : {min(ratio_vals):.2f}x – {max(ratio_vals):.2f}x")
    print(f"  Mean ratio   : {np.mean(ratio_vals):.2f}x")

    # Report at recall=0.9 if in range, otherwise at the midpoint
    target = 0.9
    if min(recall_vals) <= target <= max(recall_vals):
        ratio_at_target = float(np.interp(target, recall_vals, ratio_vals))
        print(f"  At recall=0.90: {ratio_at_target:.2f}x  ← suggested TODO value")
    else:
        mid = len(ratios) // 2
        print(f"  At recall={recall_vals[mid]:.3f}: {ratio_vals[mid]:.2f}x  ← suggested TODO value")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="qps_combined.pdf")
    ap.add_argument("--selftest", action="store_true",
                    help="render from synthetic data (no /dataset files needed)")
    args = ap.parse_args()

    set_style()
    if args.selftest:
        with tempfile.TemporaryDirectory() as tmp:
            datasets = make_selftest_datasets(tmp)
            data = [load_dataset(ds) for ds in datasets]
            fig = build_figure(datasets, data)
            fig.savefig(args.output, bbox_inches="tight", dpi=150)
    else:
        data = [load_dataset(ds) for ds in DATASETS]
        # ---- TODO measurements ----
        print("=== TODO: GP-ANN vs SURGE QPS ratio at equal recall ===")
        for ds, d in zip(DATASETS, data):
            compute_qps_ratios(ds, d)
        print("=======================================================\n")
        # ---------------------------
        fig = build_figure(DATASETS, data)
        fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()