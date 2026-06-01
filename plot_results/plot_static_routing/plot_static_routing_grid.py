#!/usr/bin/env python3
"""
plot_static_routing_quality.py

Merged "static routing quality" figure for the SURGE paper. Combines the former
Figure 1 (theoretical recall@10 vs nprobe -- partitioning quality) and Figure 3
(recall + activation vs routing factor K and vs recall target -- routing quality)
into a single grid with ONE shared legend instead of a legend per panel:

    rows    = routing modes  (NProbe | Routing Factor | Recall Target)
    columns = datasets       (SIFT-100M, MSTuring-100M, SIFT-500M, MSTuring-500M)

The NProbe row shows recall only; the Routing Factor and Recall Target rows add a
partition-activation strip beneath each recall panel (3:1 height ratio).

Per-dataset results directory (exactly as produced by
./bin/theoretical_partitioning_quality plus the oracle scripts):
    routing_metrics.csv               mode, param, recall, activation, imbalance
    oracle_nprobe_results.csv         nprobe, mean_recall
    oracle_target_recall_results.csv  recall_target, mean_recall, mean_partitions
    oracle_branching_results.csv      branching_factor, oracle_recall, activation
GP-ANN (NProbe column, optional):
    recall_results.csv                num_shards_searched, <gpann_col>

Usage:
    python plot_static_routing_quality.py -o static_routing_quality.pdf
    python plot_static_routing_quality.py --selftest -o demo.png   # synthetic data
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
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "grid.color": "lightgray",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


# Single source of truth for every series' style + label. The shared legend and
# every panel draw from this dict, so they can never disagree. (This also fixes
# the original notebooks, which mislabeled the Routing-Factor and Recall-Target
# oracles as "Oracle (Recall Target)" in several panels.)
STYLE = {
    "surge_nprobe": dict(color="C0", marker="o", ls="-",  label="SURGE (NProbe)"),
    "oracle_nprobe": dict(color="C0", ls="--", alpha=0.5, label="Oracle (NProbe)"),
    "gpann":         dict(color="C8", marker="s", ls="-", label="GP-ANN"),
    "surge_bf":      dict(color="C1", marker="s", ls="-", label="SURGE (Routing Factor)"),
    "oracle_bf":     dict(color="C1", ls="--", alpha=0.5, label="Oracle (Routing Factor)"),
    "surge_rt":      dict(color="C2", marker="^", ls="-", label="SURGE (Recall Target)"),
    "oracle_rt":     dict(color="C2", ls="--", alpha=0.5, label="Oracle (Recall Target)"),
    "target":        dict(color="red", ls=":", alpha=0.45, label="Target"),
}

# Shared legend laid out as three mode-grouped columns (one per routing mode).
LEGEND_COLUMNS = [
    ["surge_nprobe", "oracle_nprobe", "gpann"],   # NProbe column
    ["surge_bf", "oracle_bf"],                     # Routing Factor column
    ["surge_rt", "oracle_rt", "target"],           # Recall Target column
]

MODE_TITLES = ["NProbe", "Routing Factor", "Recall Target"]


# --------------------------------------------------------------------------- #
# Dataset configuration
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    title: str
    results_dir: str                  # directory holding the 4 SURGE csvs
    gpann_csv: Optional[str] = None   # recall_results.csv, or None to skip GP-ANN
    gpann_col: str = "route_456_HNSW_HierKMeans"
    recall_ylim: tuple = (0.55, 1.03)

    @property
    def routing_metrics(self): return os.path.join(self.results_dir, "routing_metrics.csv")
    @property
    def nprobe_oracle(self):   return os.path.join(self.results_dir, "oracle_nprobe_results.csv")
    @property
    def rt_oracle(self):       return os.path.join(self.results_dir, "oracle_target_recall_results.csv")
    @property
    def bf_oracle(self):       return os.path.join(self.results_dir, "oracle_branching_results.csv")


# >>> EDIT THESE PATHS <<<  (defaults copied from plot_figure1 / plot_figure3)
_ROOT = "/dataset/surge/results/theoretical_partition_quality"
DATASETS = [
    DatasetSpec(
        title="SIFT-100M",
        results_dir=f"{_ROOT}/theoretical_partition_quality_sift-100M_10_20260519_105832",
        gpann_csv="/dataset/big-ann-benchmarks/data/bigann/gpann_partitions/recall_results.csv",
        recall_ylim=(0.67, 1.03),
    ),
    DatasetSpec(
        title="MSTuring-100M",
        results_dir=f"{_ROOT}/theoretical_partition_quality_msturing-100M_10_20260519_105024",
        gpann_csv="/dataset/big-ann-benchmarks/data/MSTuringANNS/gpann_partitions/recall_results.csv",
        recall_ylim=(0.55, 1.03),
    ),
    # 500M scale-invariance rows (uncomment to include; GP-ANN unavailable at 500M):
    DatasetSpec("SIFT-500M",     f"{_ROOT}/theoretical_partition_quality_sift-500M_10_20260519_130459",     recall_ylim=(0.67, 1.03)),
    DatasetSpec("MSTuring-500M", f"{_ROOT}/theoretical_partition_quality_msturing-500M_10_20260520_133756", recall_ylim=(0.55, 1.03)),
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _append_nprobe_full(df):
    """SURGE's nprobe sweep stops at 9; nprobe=10 trivially gives recall 1.0."""
    return pd.concat(
        [df, pd.DataFrame({"mode": ["nprobe"], "param": [10], "recall": [1.0],
                           "activation": [1.0], "imbalance": [0]})],
        ignore_index=True,
    )


def load_dataset(ds: DatasetSpec) -> dict:
    rm = _append_nprobe_full(pd.read_csv(ds.routing_metrics))
    d = {
        "rm": rm,
        "np_oracle": pd.read_csv(ds.nprobe_oracle),
        "rt_oracle": pd.read_csv(ds.rt_oracle),
        "bf_oracle": pd.read_csv(ds.bf_oracle),
        "gpann": pd.read_csv(ds.gpann_csv) if (ds.gpann_csv and os.path.exists(ds.gpann_csv)) else None,
    }
    return d


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _line(ax, x, y, key):
    """Draw a styled line with NO label (labels live in the shared legend)."""
    s = {k: v for k, v in STYLE[key].items() if k != "label"}
    ax.plot(x, y, **s)


def plot_nprobe(ax, ds, d):
    sub = d["rm"][d["rm"]["mode"] == "nprobe"].sort_values("param")
    _line(ax, sub["param"], sub["recall"], "surge_nprobe")
    npo = d["np_oracle"]
    _line(ax, npo["nprobe"][:10], npo["mean_recall"][:10], "oracle_nprobe")
    if d["gpann"] is not None:
        g = d["gpann"]
        _line(ax, g["num_shards_searched"], g[ds.gpann_col], "gpann")
    ax.set_xlabel("NProbe")
    ax.set_ylim(*ds.recall_ylim)


def plot_branching(ax_top, ax_bot, ds, d):
    bf = d["rm"][d["rm"]["mode"] == "branching_factor"].sort_values("param")
    _line(ax_top, bf["param"], bf["recall"], "surge_bf")
    _line(ax_bot, bf["param"], bf["activation"], "surge_bf")
    o = d["bf_oracle"]
    _line(ax_top, o["branching_factor"], o["oracle_recall"], "oracle_bf")
    _line(ax_bot, o["branching_factor"], o["activation"], "oracle_bf")
    ax_bot.set_xlabel("Routing Factor (K)")


def plot_recall_target(ax_top, ax_bot, ds, d):
    rt = d["rm"][d["rm"]["mode"] == "recall_target"].sort_values("param")
    _line(ax_top, rt["param"], rt["recall"], "surge_rt")
    _line(ax_bot, rt["param"], rt["activation"], "surge_rt")
    o = d["rt_oracle"]
    _line(ax_top, o["recall_target"], o["mean_recall"], "oracle_rt")
    _line(ax_bot, o["recall_target"], o["mean_partitions"] / 10.0, "oracle_rt")
    _line(ax_top, rt["param"], rt["param"], "target")  # y = x reference
    ax_bot.set_xlabel("Target Recall")


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def _group_handles(group):
    """Proxy (handles, labels) for one mode group (lines are drawn unlabeled)."""
    handles = [Line2D([0], [0], **{k: v for k, v in STYLE[key].items() if k != "label"})
               for key in group]
    labels = [STYLE[key]["label"] for key in group]
    return handles, labels


def _row_label(ax, text):
    """Bold rotated mode label to the left of a row's first recall panel."""
    ax.text(-0.42, 0.5, text, transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontweight="bold", fontsize=13)


def build_figure(datasets, data, share_recall_ylim=False):
    ncols = len(datasets)
    fig = plt.figure(figsize=(3.4 * ncols, 9))
    # 3 mode rows; NProbe row is shorter (recall only) so that every recall
    # panel ends up the same height as the recall panels in the other two rows.
    outer = fig.add_gridspec(3, ncols, height_ratios=[3, 4, 4],
                             hspace=0.3, wspace=0.23)

    for c, (ds, d) in enumerate(zip(datasets, data)):
        first = (c == 0)

        # --- Row 0: NProbe (single recall panel) ---
        ax_np = fig.add_subplot(outer[0, c])
        plot_nprobe(ax_np, ds, d)
        ax_np.set_title(ds.title, fontweight="bold")                       # dataset = column header
        if first:
            ax_np.set_ylabel("Theoretical Recall@10")
            _row_label(ax_np, MODE_TITLES[0])
            ax_np.legend(*_group_handles(LEGEND_COLUMNS[0]), loc="lower right",
                         frameon=False, fontsize=9, handlelength=1.6,
                         labelspacing=0.3, borderaxespad=0.5)

        # --- Row 1: Routing Factor (recall + activation) ---
        gk = outer[1, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax_k_t = fig.add_subplot(gk[0])
        ax_k_b = fig.add_subplot(gk[1], sharex=ax_k_t)
        plt.setp(ax_k_t.get_xticklabels(), visible=False)
        plot_branching(ax_k_t, ax_k_b, ds, d)
        ax_k_b.set_ylim(0, 1.05)
        if first:
            ax_k_t.set_ylabel("Theo. Recall@10")
            ax_k_b.set_ylabel("Activation")
            _row_label(ax_k_t, MODE_TITLES[1])
            ax_k_t.legend(*_group_handles(LEGEND_COLUMNS[1]), loc="lower right",
                          frameon=False, fontsize=9, handlelength=1.6,
                          labelspacing=0.3, borderaxespad=0.5)

        # --- Row 2: Recall Target (recall + activation) ---
        gr = outer[2, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax_r_t = fig.add_subplot(gr[0])
        ax_r_b = fig.add_subplot(gr[1], sharex=ax_r_t)
        plt.setp(ax_r_t.get_xticklabels(), visible=False)
        plot_recall_target(ax_r_t, ax_r_b, ds, d)
        ax_r_t.set_ylim(.45, 1.1)
        ax_r_b.set_ylim(0, 1.05)
        if first:
            ax_r_t.set_ylabel("Theo. Recall@10")
            ax_r_b.set_ylabel("Activation")
            _row_label(ax_r_t, MODE_TITLES[2])
            ax_r_t.legend(*_group_handles(LEGEND_COLUMNS[2]), loc="lower right",
                          frameon=False, fontsize=9, handlelength=1.6,
                          labelspacing=0.3, borderaxespad=0.5)

        if share_recall_ylim:
            ax_np.set_ylim(*ds.recall_ylim)
            ax_k_t.set_ylim(*ds.recall_ylim)
            ax_r_t.set_ylim(*ds.recall_ylim)

    return fig


# --------------------------------------------------------------------------- #
# Self-test: synthesize schema-matching CSVs so the layout can be checked
# without the cluster's /dataset files.
# --------------------------------------------------------------------------- #
def _write_synthetic(results_dir, gpann_path, seed=0):
    os.makedirs(results_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    npr = np.arange(1, 10)
    rec = 1 - 0.4 * np.exp(-npr / 2.0) + rng.normal(0, 0.004, npr.size)
    K = np.array([1, 2, 5, 10, 20, 30, 40, 50])
    krec = 1 - 0.35 * np.exp(-K / 6.0)
    kact = np.clip(0.1 + 0.018 * K, 0, 1)
    tgt = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    trec = np.clip(tgt + 0.05, 0, 1.0)
    tact = np.clip(0.12 + 0.6 * tgt, 0, 1)
    pd.concat([
        pd.DataFrame({"mode": "nprobe", "param": npr, "recall": rec,
                      "activation": npr / 10.0, "imbalance": 0.1}),
        pd.DataFrame({"mode": "branching_factor", "param": K, "recall": krec,
                      "activation": kact, "imbalance": 0.1}),
        pd.DataFrame({"mode": "recall_target", "param": tgt, "recall": trec,
                      "activation": tact, "imbalance": 0.1}),
    ], ignore_index=True).to_csv(os.path.join(results_dir, "routing_metrics.csv"), index=False)

    npr10 = np.arange(1, 11)
    pd.DataFrame({"nprobe": npr10,
                  "mean_recall": np.clip(1 - 0.4 * np.exp(-npr10 / 2.0) + 0.02, 0, 1)}
                 ).to_csv(os.path.join(results_dir, "oracle_nprobe_results.csv"), index=False)
    pd.DataFrame({"recall_target": tgt, "mean_recall": np.clip(trec + 0.01, 0, 1),
                  "mean_partitions": np.clip(tact * 10 - 0.6, 0.5, 10)}
                 ).to_csv(os.path.join(results_dir, "oracle_target_recall_results.csv"), index=False)
    pd.DataFrame({"branching_factor": K, "oracle_recall": np.clip(krec + 0.03, 0, 1),
                  "activation": np.clip(kact - 0.03, 0, 1)}
                 ).to_csv(os.path.join(results_dir, "oracle_branching_results.csv"), index=False)

    ns = np.arange(1, 11)
    pd.DataFrame({"num_shards_searched": ns,
                  "route_456_HNSW_HierKMeans": np.clip(1 - 0.45 * np.exp(-ns / 2.5), 0, 1)}
                 ).to_csv(gpann_path, index=False)


def make_selftest_datasets(tmp):
    # 4 demo datasets to exercise the 4-column layout; 500M columns omit GP-ANN.
    demo = [("SIFT-100M (demo)", (0.67, 1.03), True),
            ("MSTuring-100M (demo)", (0.55, 1.03), True),
            ("SIFT-500M (demo)", (0.67, 1.03), False),
            ("MSTuring-500M (demo)", (0.55, 1.03), False)]
    specs = []
    for i, (title, ylim, has_gpann) in enumerate(demo):
        rdir = os.path.join(tmp, f"ds{i}")
        gpath = os.path.join(tmp, f"gpann{i}.csv")
        _write_synthetic(rdir, gpath, seed=i)
        specs.append(DatasetSpec(title=title, results_dir=rdir,
                                 gpann_csv=gpath if has_gpann else None,
                                 recall_ylim=ylim))
    return specs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="static_routing_quality.pdf")
    ap.add_argument("--selftest", action="store_true",
                    help="render from synthetic data (no /dataset files needed)")
    ap.add_argument("--share-ylim", action="store_true",
                    help="force a common recall y-range per dataset row across modes "
                         "(note: may clip the recall-target curve at low targets)")
    args = ap.parse_args()

    set_style()

    if args.selftest:
        with tempfile.TemporaryDirectory() as tmp:
            datasets = make_selftest_datasets(tmp)
            data = [load_dataset(ds) for ds in datasets]
            fig = build_figure(datasets, data, share_recall_ylim=args.share_ylim)
            fig.savefig(args.output, bbox_inches="tight", dpi=150)
    else:
        data = [load_dataset(ds) for ds in DATASETS]
        fig = build_figure(DATASETS, data, share_recall_ylim=args.share_ylim)
        fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()