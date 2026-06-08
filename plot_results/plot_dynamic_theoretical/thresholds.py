#!/usr/bin/env python3
"""
plot_threshold_grid.py

Maintenance-threshold sweep figure. Shows how the rebuild threshold tau trades
recall against maintenance frequency: tau = 0 rebuilds on every drift (always
fresh), tau = 1 never rebuilds (always stale), and intermediate values rebuild
only when the fraction of reassigned centroids exceeds tau.

    rows    = datasets   (MSTuring-100M, SIFT-100M)
    columns = runbook    (Clustered, Shift)        # uniform omitted -- no maintenance needed
    each cell: theoretical recall@10 vs runbook step, one line per threshold,
               with an X marker at every step where a rebuild actually fired.

Thresholds use a sequential colormap (viridis) so they do NOT collide with
SURGE's categorical routing-mode colors (C0/C1/C2) used elsewhere.

Data (mirrors plot_figure4.ipynb concat_results; lives under DATA_ROOT):
  <cluster_history_dir>/full_results_NProbe_t{t}[_with_oracle].csv
        columns: step, recall, mode, param, did_rebuild   (t added on load)

Usage:
  python plot_threshold_grid.py -o threshold_sweep.pdf
  python plot_threshold_grid.py --selftest -o demo.png      # synthetic data
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def set_style():
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": "lightgray",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


# Thresholds to draw (t in 0..100; t/100 = tau). Subset of the figure4 sweep.
THRESHOLDS = [0, 40, 60, 80, 100]
CMAP = plt.get_cmap("viridis")        # sequential -> no clash with routing colors
REBUILD_MARKER = "x"                   # thin x so dense rebuilds don't blob together
REBUILD_SIZE = 11                      # scatter point area
REBUILD_LW = 0.8                       # marker stroke width
REBUILD_COLOR = "0.25"                 # neutral grey for the rebuild-marker legend key

MODE = "NProbe"                        # routing mode used to measure recall (per figure4)


def is_reference(t):
    """The always-rebuild (min) and never-rebuild (max) thresholds are bounds."""
    return t == THRESHOLDS[0] or t == THRESHOLDS[-1]


# Only the intermediate thresholds carry color (spread wide across the colormap
# so they stay distinct); the two reference bounds are faded grey dashed lines.
MAIN_THRESHOLDS = [t for t in THRESHOLDS if not is_reference(t)]
REFERENCE_COLORS = {THRESHOLDS[0]: "0", THRESHOLDS[-1]: "0.55"}


def color_for(t):
    if is_reference(t):
        return REFERENCE_COLORS[t]
    xs = (np.linspace(0.12, 0.85, len(MAIN_THRESHOLDS)) if len(MAIN_THRESHOLDS) > 1
          else [0.5])
    return CMAP(xs[MAIN_THRESHOLDS.index(t)])


def line_style(t):
    if is_reference(t):
        return dict(ls="--", alpha=0.55, lw=1.3)   # faded dashed reference bound
    return dict(ls="-", alpha=1.0, lw=1.8)


def tau_label(t):
    if t == 0:
        return r"$\tau=0.0$ (always)"
    if t == 100:
        return r"$\tau=1.0$ (never)"
    return rf"$\tau={t/100:.1f}$"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_ROOT = "/dataset/big-ann-benchmarks/data"     # overridden by --selftest


@dataclass
class DatasetSpec:
    title: str
    key: str            # "bigann" or "msturing"
    nprobe_param: int   # SIFT=3, MSTuring=5


@dataclass
class RunbookSpec:
    title: str          # "Clustered" / "Shift"
    key: str            # "clustered" / "shift"
    n_steps: int
    nprobe_has_oracle: bool = True     # shift NProbe files have no "_with_oracle" suffix


DATASETS = [
    DatasetSpec("MSTuring-100M", "msturing", nprobe_param=5),
    DatasetSpec("SIFT-100M", "bigann", nprobe_param=3),
]
RUNBOOKS = [
    RunbookSpec("Clustered", "clustered", n_steps=1280, nprobe_has_oracle=True),
    RunbookSpec("Shift", "shift", n_steps=616, nprobe_has_oracle=False),
]


# --------------------------------------------------------------------------- #
# Paths (mirror plot_figure4.ipynb / plot_figure2.ipynb naming)
# --------------------------------------------------------------------------- #
def cluster_history_dir(ds, rb):
    if ds.key == "bigann":
        return os.path.join(DATA_ROOT, f"bigann-{rb.key}",
                            f"cluster_history_bigann-100M-{rb.key}_10000_10")
    return os.path.join(DATA_ROOT, f"MSTuring-100M-{rb.key}",
                        f"cluster_history_msturing-100M-{rb.key}_10000_10")


def load_threshold(ds, rb):
    base = cluster_history_dir(ds, rb)
    frames = []
    for t in THRESHOLDS:
        path = None
        for suffix in (("_with_oracle", "") if rb.nprobe_has_oracle else ("",)):
            cand = os.path.join(base, f"full_results_{MODE}_t{t}{suffix}.csv")
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            continue
        df = pd.read_csv(path)
        df["t"] = t
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_cell(ax, ds, rb, df):
    for t in THRESHOLDS:
        color = color_for(t)
        sub = df[(df["t"] == t) & (df["mode"] == MODE)
                 & (df["param"] == ds.nprobe_param)].sort_values("step")
        if not len(sub):
            continue
        ax.plot(sub["step"], sub["recall"], color=color, zorder=2, **line_style(t))
        if is_reference(t):
            continue                                  # no rebuild markers on the bounds
        rb_pts = sub[sub["did_rebuild"] == True]      # noqa: E712 (pandas mask)
        if len(rb_pts):
            ax.scatter(rb_pts["step"], rb_pts["recall"], marker=REBUILD_MARKER,
                       color=color, s=REBUILD_SIZE, linewidths=REBUILD_LW, zorder=4)
    ax.grid(True, which="both")


def build_figure(datasets, runbooks, loaded):
    nrows, ncols = len(datasets), len(runbooks)
    fig = plt.figure(figsize=(3.7 * ncols, 2.7 * nrows + 0.4))
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex="col")

    for r, ds in enumerate(datasets):
        for c, rb in enumerate(runbooks):
            ax = axes[r][c]
            plot_cell(ax, ds, rb, loaded[(ds.title, rb.title)])
            if r == 0:
                ax.set_title(rb.title)
            if c == 0:
                ax.set_ylabel("Theoretical Recall@10")
                ax.text(-0.30, 0.5, f"{ds.title}\n({MODE} = {ds.nprobe_param})",
                        transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontweight="bold", fontsize=11)
            if r == nrows - 1:
                ax.set_xlabel("Runbook Step")

    # shared legend: one entry per threshold (dashed/faded for the bounds) + rebuild key
    handles, labels = [], []
    for t in THRESHOLDS:
        st = line_style(t)
        handles.append(Line2D([0], [0], color=color_for(t), ls=st["ls"],
                              alpha=st["alpha"], lw=max(st["lw"], 2.0)))
        labels.append(tau_label(t))
    handles.append(Line2D([0], [0], color=REBUILD_COLOR, marker=REBUILD_MARKER,
                          ls="none", markersize=6, markeredgewidth=1.0))
    labels.append("rebuild")
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.96), columnspacing=1.4, handletextpad=0.5)
    return fig


# --------------------------------------------------------------------------- #
# Self-test: synthesize schema-matching CSVs so the layout can be checked
# without the cluster's /dataset tree.
# --------------------------------------------------------------------------- #
def _sim_threshold(step, n_steps, t, drift, rng):
    top = 0.97
    floor = 0.93 - 0.16 * drift
    if t >= 100:
        rebuilds = np.array([], dtype=int)
    else:
        interval = int(150 * (1 + t / 30.0))          # smaller t -> more frequent rebuilds
        rebuilds = np.arange(interval, n_steps, interval)
    since = np.array([s - (rebuilds[rebuilds <= s].max() if (rebuilds <= s).any() else 0)
                      for s in step], dtype=float)
    rate = 0.00065 + 0.00065 * drift
    rec = np.maximum(floor, top - rate * since) + rng.normal(0, 0.003, step.size)
    did = np.zeros(step.size, dtype=bool)
    for rbk in rebuilds:
        did[int(np.argmin(np.abs(step - rbk)))] = True
    return np.clip(rec, 0, 1), did


def selftest_setup(tmp):
    global DATA_ROOT
    DATA_ROOT = tmp
    for ds in DATASETS:
        for drift, rb in enumerate(RUNBOOKS, start=1):   # clustered=1, shift=2
            base = cluster_history_dir(ds, rb)
            os.makedirs(base, exist_ok=True)
            rng = np.random.default_rng(abs(hash((ds.key, rb.key))) % 2**32)
            step = np.arange(0, rb.n_steps, 4)
            for t in THRESHOLDS:
                rec, did = _sim_threshold(step, rb.n_steps, t, drift, rng)
                df = pd.DataFrame({"step": step, "recall": rec, "mode": MODE,
                                   "param": ds.nprobe_param, "did_rebuild": did})
                suffix = "_with_oracle" if rb.nprobe_has_oracle else ""
                df.to_csv(os.path.join(base, f"full_results_{MODE}_t{t}{suffix}.csv"),
                          index=False)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="threshold_sweep.pdf")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    set_style()
    if args.selftest:
        with tempfile.TemporaryDirectory() as tmp:
            selftest_setup(tmp)
            loaded = {(ds.title, rb.title): load_threshold(ds, rb)
                      for ds in DATASETS for rb in RUNBOOKS}
            fig = build_figure(DATASETS, RUNBOOKS, loaded)
            fig.savefig(args.output, bbox_inches="tight", dpi=150)
    else:
        loaded = {(ds.title, rb.title): load_threshold(ds, rb)
                  for ds in DATASETS for rb in RUNBOOKS}
        fig = build_figure(DATASETS, RUNBOOKS, loaded)
        fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()