#!/usr/bin/env python3
"""
end_to_end_grid.py
 
End-to-end recall / throughput / update-throughput figure for the SURGE paper,
formatted as a single grid. This is the grid version of plot_figure7.ipynb: the
notebook drew one tall 3-panel figure per (dataset, runbook); here all four of
those figures become columns of a shared grid with ONE shared legend instead of
a per-figure legend.
 
    columns = (dataset, runbook)   MSTuring-clustered | MSTuring-shift |
                                   SIFT-clustered     | SIFT-shift
    rows (per column, stacked, height ratios 3:1:1, hspace=0):
        row 0  Recall@10        vs step   (measured + theoretical, all systems)
        row 1  QPS              vs step   (log scale)
        row 2  (update) ops/s   vs step   (log scale, insert/delete by phase)
 
In addition to the notebook content, every column now draws *rebuild markers*
on all three panels: an "x" at each step where the with-maintenance run actually
fired a maintenance rebuild (operation == "rebuild"), exactly like the rebuild
markers in plot_dynamic_theoretical/thresholds.py. The marker sits on the
with-maintenance curve in each panel.
 
Data (identical paths/!schema to plot_figure7.ipynb):
  SURGE   results.csv  with columns: step, operation, mode, param,
                       recall@10, theoretical_recall@10, throughput, ...
            - no-maintenance run  == ..._t10000-rebuildsbroken/results.csv
            - with-maintenance run == ..._t6000/results.csv (fires rebuilds)
  GP-ANN  *_runbook_results_nprobe_with_theo_sweep*.csv  with columns:
                       step, operation, nprobe, recall@10, theoretical_recall,
                       qps_or_throughput
  Postgres JSONL (msturing-clustered only): operation, elapsed_s, count,
                       recall_at_k, step  (read as one JSON object per line)
 
Usage:
  python end_to_end_grid.py -o end_to_end_grid.pdf
  python end_to_end_grid.py --selftest -o demo.png    # synthetic data, no cluster
"""
 
import argparse
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List
 
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
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": "lightgray",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })
 
 
# Single source of truth for every series' style + label, so the shared legend
# and every panel agree. (Fixes the notebook, which mislabeled several of the
# "no maintenance" theoretical curves as "with maintenance, theoretical".)
no_rebuilds_color = "firebrick"
rebuilds_color = "C0"
gpann_color = "C8"
postgres_color = "C4"
 
STYLE = {
    "no_maint":      dict(color=no_rebuilds_color, ls="-",  label="no maintenance"),
    "no_maint_theo": dict(color=no_rebuilds_color, ls="--", label="no maintenance, theoretical"),
    "with_maint":    dict(color=rebuilds_color,    ls="-",  label="with maintenance"),
    "with_maint_theo": dict(color=rebuilds_color,  ls="--", label="with maintenance, theoretical"),
    "gpann":         dict(color=gpann_color,       ls="-",  label="GP-ANN"),
    "gpann_theo":    dict(color=gpann_color,       ls="--", label="GP-ANN, theoretical"),
    "postgres":      dict(color=postgres_color,    ls="-",  label="PostgreSQL"),
}
 
# Order of keys in the shared legend (postgres + rebuild appended dynamically).
LEGEND_ORDER = ["no_maint", "no_maint_theo", "with_maint", "with_maint_theo",
                "gpann", "gpann_theo"]
 
# Rebuild-marker styling -- copied from thresholds.py so the two figures match.
REBUILD_MARKER = "x"      # thin x so dense rebuilds don't blob together
REBUILD_SIZE = 15         # scatter point area
REBUILD_LW = 0.8          # marker stroke width
REBUILD_COLOR = "0.25"    # neutral grey for the rebuild-marker legend key
 
# Insert/delete phase shading.
PHASE_INSERT_COLOR = "green"
PHASE_DELETE_COLOR = "red"
PHASE_ALPHA = 0.05
 
 
# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SURGE_ROOT = "/users/dkhimey/surge/results"          # overridden by --selftest
 
# Clustered runbooks: 10 alternating insert/delete phases of 128 steps each.
CLUSTERED_BOUNDARIES = [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]
# Shift runbooks: irregular insert/delete phases (from the notebook).
SHIFT_BOUNDARIES = [0, 208, 235, 338, 364, 468, 494, 590, 616]
 
 
@dataclass
class ColumnSpec:
    title: str
    no_maint_csv: str          # SURGE t10000-rebuildsbroken (no maintenance)
    with_maint_csv: str        # SURGE t6000 (with maintenance -> rebuilds)
    gpann_csv: str
    phase_boundaries: List[int]
    postgres_json: Optional[str] = None    # only msturing-clustered in the paper
    mode: str = "RecallTarget"             # routing mode used for recall/QPS lines
    param: float = 0.9                     # routing param for both SURGE runs
    gpann_param: int = 5                   # GP-ANN nprobe selected for the lines
 
 
DATASETS = [
    ColumnSpec(
        # /users/dkhimey/surge/results/msturing-100M-clustered_t10000/results.csv
        title="MSTuring-100M-clustered",
        no_maint_csv=f"{SURGE_ROOT}/msturing-100M-clustered_t10000/results.csv",
        with_maint_csv=f"{SURGE_ROOT}/msturing-100M-clustered_t6000/results.csv",
        gpann_csv="/dataset/gp-ann-inserts/results/msturing_ef200/"
                  "msturing100Mclustered_runbook_results_nprobe_with_theo_sweep.csv",
        postgres_json="/dataset/postgres/results/"
                      "msturing-100M-clustered-newest-results-32threads-withmaintenancetiming.json",
        phase_boundaries=CLUSTERED_BOUNDARIES,
    ),
    ColumnSpec(
        title="MSTuring-100M-shift",
        no_maint_csv=f"{SURGE_ROOT}/msturing-100M-shift_t10000/results.csv",
        with_maint_csv=f"{SURGE_ROOT}/msturing-100M-shift_t6000/results.csv",
        gpann_csv="/dataset/gp-ann-inserts/results/msturing_ef200/"
                  "msturing100Mshift_runbook_results_nprobe_with_theo_sweep.csv",
        phase_boundaries=SHIFT_BOUNDARIES,
    ),
    ColumnSpec(
        title="SIFT-100M-clustered",
        no_maint_csv=f"{SURGE_ROOT}/bigann-100M-clustered_t10000-rebuildsbroken/results.csv",
        with_maint_csv=f"{SURGE_ROOT}/bigann-100M-clustered_t6000/results.csv",
        gpann_csv="/dataset/big-ann-benchmarks/data/bigann-clustered/gpann_partitions/"
                  "bigann100Mclustered_runbook_results_nprobe_with_theo_sweep_final.csv",
        phase_boundaries=CLUSTERED_BOUNDARIES,
    ),
    ColumnSpec(
        title="SIFT-100M-shift",
        no_maint_csv=f"{SURGE_ROOT}/bigann-100M-shift_t10000-rebuildsbroken/results.csv",
        with_maint_csv=f"{SURGE_ROOT}/bigann-100M-shift_t6000/results.csv",
        gpann_csv="/dataset/big-ann-benchmarks/data/bigann-shift/gpann_partitions/"
                  "bigann100Mshift_runbook_results_nprobe_with_theo_sweep.csv",
        phase_boundaries=SHIFT_BOUNDARIES,
    ),
]
 
 
# --------------------------------------------------------------------------- #
# Loading / cleaning (mirrors the notebook helpers)
# --------------------------------------------------------------------------- #
def read_postgres_results(json_path):
    """Read a JSONL (one JSON object per line) Postgres result file."""
    records = []
    if not os.path.exists(json_path):
        return pd.DataFrame()
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)
 
 
def remove_weird_throughput_measurements(df, mode, param):
    """Drop the lower of two wildly different throughput points for a (mode, param)."""
    subset = df[(df["mode"] == mode) & (df["param"] == param)]
    if len(subset) <= 1:
        return df
    mx, mn = subset["throughput"].max(), subset["throughput"].min()
    if mx > 1.5 * mn:
        df = df.drop(subset[subset["throughput"] == mn].index)
    return df
 
 
def remove_weird_throughput_measurements_gpann(df, nprobe):
    subset = df[df["nprobe"] == nprobe]
    if len(subset) <= 1:
        return df
    mx, mn = subset["qps_or_throughput"].max(), subset["qps_or_throughput"].min()
    if mx > 1.5 * mn:
        df = df.drop(subset[subset["qps_or_throughput"] == mn].index)
    return df
 
 
def remove_duplicate_steps(df):
    """Keep the last record per (step, operation, param, mode)."""
    df = df.sort_values(by=["step", "operation", "param", "mode"],
                        ascending=[True, True, True, True])
    df = df.drop_duplicates(subset=["step", "operation", "param", "mode"], keep="last")
    return df
 
 
def load_column(cfg: ColumnSpec) -> dict:
    no_maint = pd.read_csv(cfg.no_maint_csv)
    with_maint = pd.read_csv(cfg.with_maint_csv)
    gpann = pd.read_csv(cfg.gpann_csv)
 
    with_maint = remove_weird_throughput_measurements(with_maint, cfg.mode, cfg.param)
    no_maint = remove_weird_throughput_measurements(no_maint, cfg.mode, cfg.param)
    gpann = remove_weird_throughput_measurements_gpann(gpann, cfg.gpann_param)
 
    with_maint = remove_duplicate_steps(with_maint)
    no_maint = remove_duplicate_steps(no_maint)
 
    postgres = None
    if cfg.postgres_json:
        postgres = read_postgres_results(cfg.postgres_json)
        if not postgres.empty and "count" in postgres and "elapsed_s" in postgres:
            postgres["qps"] = postgres["count"] / postgres["elapsed_s"]
 
    return {"no_maint": no_maint, "with_maint": with_maint,
            "gpann": gpann, "postgres": postgres}
 
 
# --------------------------------------------------------------------------- #
# Small plotting helpers
# --------------------------------------------------------------------------- #
def _style(key):
    return {k: v for k, v in STYLE[key].items() if k != "label"}
 
 
def _surge_search(df, cfg):
    """SURGE search rows for the configured routing (mode, param), sorted by step."""
    return df[(df["mode"] == cfg.mode) & (df["param"] == cfg.param)].sort_values("step")
 
 
def _gpann_search(df, cfg):
    return df[df["nprobe"] == cfg.gpann_param].sort_values("step")
 
 
def add_phase_spans(ax, boundaries):
    """Shade alternating insert (green) / delete (red) phases."""
    for i in range(len(boundaries) - 1):
        color = PHASE_INSERT_COLOR if i % 2 == 0 else PHASE_DELETE_COLOR
        ax.axvspan(boundaries[i], boundaries[i + 1], color=color, alpha=PHASE_ALPHA)
 
 
def _rebuild_steps(with_maint):
    """Steps at which the with-maintenance run actually fired a rebuild."""
    if "operation" not in with_maint.columns:
        return np.array([])
    r = with_maint[with_maint["operation"] == "rebuild"]
    return np.sort(r["step"].unique())
 
 
def _markers_on_line(line_x, line_y, steps):
    """Interpolate the with-maintenance line's y at each rebuild step so the
    rebuild 'x' sits on the curve (as in thresholds.py, where the marker lands
    on the recall line at the step that rebuilt)."""
    line_x = np.asarray(line_x, dtype=float)
    line_y = np.asarray(line_y, dtype=float)
    steps = np.asarray(steps, dtype=float)
    if line_x.size == 0 or steps.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(line_x)
    lx, ly = line_x[order], line_y[order]
    keep = (steps >= lx.min()) & (steps <= lx.max())
    steps = steps[keep]
    return steps, np.interp(steps, lx, ly)
 
 
def add_rebuild_markers(ax, line_x, line_y, steps):
    sx, sy = _markers_on_line(line_x, line_y, steps)
    if sx.size:
        ax.scatter(sx, sy, marker=REBUILD_MARKER, color=rebuilds_color,
                   s=REBUILD_SIZE, linewidths=REBUILD_LW, zorder=4)
 
 
# --------------------------------------------------------------------------- #
# Per-panel plotting
# --------------------------------------------------------------------------- #
def plot_recall(ax, cfg, d, rebuild_steps):
    nm = _surge_search(d["no_maint"], cfg)
    wm = _surge_search(d["with_maint"], cfg)
    gp = _gpann_search(d["gpann"], cfg)
 
    ax.plot(nm["step"], nm["recall@10"], **_style("no_maint"))
    ax.plot(nm["step"], nm["theoretical_recall@10"], **_style("no_maint_theo"))
    ax.plot(wm["step"], wm["recall@10"], **_style("with_maint"))
    ax.plot(wm["step"], wm["theoretical_recall@10"], **_style("with_maint_theo"))
    ax.plot(gp["step"], gp["recall@10"], **_style("gpann"))
    ax.plot(gp["step"], gp["theoretical_recall"], **_style("gpann_theo"))
 
    if d["postgres"] is not None and not d["postgres"].empty:
        pg = d["postgres"][d["postgres"]["operation"] == "search"].sort_values("step")
        ax.plot(pg["step"], pg["recall_at_k"], **_style("postgres"))
 
    add_rebuild_markers(ax, wm["step"], wm["recall@10"], rebuild_steps)
    # ax.set_ylim(0, 1.03)
 
 
def plot_qps(ax, cfg, d, rebuild_steps):
    nm = _surge_search(d["no_maint"], cfg)
    wm = _surge_search(d["with_maint"], cfg)
    gp = _gpann_search(d["gpann"], cfg)
 
    ax.plot(nm["step"], nm["throughput"], **_style("no_maint"))
    ax.plot(wm["step"], wm["throughput"], **_style("with_maint"))
    ax.plot(gp["step"], gp["qps_or_throughput"], **_style("gpann"))
 
    if d["postgres"] is not None and not d["postgres"].empty:
        pg = d["postgres"][d["postgres"]["operation"] == "search"].sort_values("step")
        ax.plot(pg["step"], pg["qps"], **_style("postgres"))
 
    add_rebuild_markers(ax, wm["step"], wm["throughput"], rebuild_steps)
    ax.set_yscale("log")
 
 
def _plot_update_phases(ax, df, boundaries, op_insert, op_delete, x, y, key):
    """Plot insert (in green phases) and delete (in red phases) throughput,
    broken at phase boundaries so lines don't connect across phases."""
    s = _style(key)
    # inserts -> phases [b_i, b_{i+1})
    for i in range(len(boundaries) - 1):
        seg = df[(df["operation"] == op_insert) &
                 (df["step"] >= boundaries[i]) & (df["step"] < boundaries[i + 1])]
        if not seg.empty:
            ax.plot(seg[x], seg[y], **s)
    # deletes -> offset by one phase
    for i in range(len(boundaries) - 2):
        seg = df[(df["operation"] == op_delete) &
                 (df["step"] >= boundaries[i + 1]) & (df["step"] < boundaries[i + 2])]
        if not seg.empty:
            ax.plot(seg[x], seg[y], **s)
 
 
def plot_update(ax, cfg, d, rebuild_steps):
    b = cfg.phase_boundaries
    _plot_update_phases(ax, d["no_maint"], b, "insert", "delete",
                        "step", "throughput", "no_maint")
    _plot_update_phases(ax, d["with_maint"], b, "insert", "delete",
                        "step", "throughput", "with_maint")
    _plot_update_phases(ax, d["gpann"], b, "INSERT", "DELETE",
                        "step", "qps_or_throughput", "gpann")
    if d["postgres"] is not None and not d["postgres"].empty:
        _plot_update_phases(ax, d["postgres"], b, "insert", "delete",
                            "step", "qps", "postgres")
 
    # rebuild markers sit on the with-maintenance update-throughput curve
    upd = d["with_maint"][d["with_maint"]["operation"].isin(["insert", "delete"])]
    upd = upd.sort_values("step")
    add_rebuild_markers(ax, upd["step"], upd["throughput"], rebuild_steps)
    ax.set_yscale("log")
 
 
# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def build_figure(datasets, data):
    ncols = len(datasets)
    fig = plt.figure(figsize=(5.0 * ncols, 5.0))
    outer = fig.add_gridspec(1, ncols, wspace=0.18)
 
    any_postgres = any(d["postgres"] is not None and not d["postgres"].empty
                       for d in data)
 
    for c, (cfg, d) in enumerate(zip(datasets, data)):
        first = (c == 0)
        rebuild_steps = _rebuild_steps(d["with_maint"])
 
        inner = outer[0, c].subgridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.0)
        ax1 = fig.add_subplot(inner[0])
        ax2 = fig.add_subplot(inner[1], sharex=ax1)
        ax3 = fig.add_subplot(inner[2], sharex=ax1)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
 
        plot_recall(ax1, cfg, d, rebuild_steps)
        plot_qps(ax2, cfg, d, rebuild_steps)
        plot_update(ax3, cfg, d, rebuild_steps)
 
        for ax in (ax1, ax2, ax3):
            add_phase_spans(ax, cfg.phase_boundaries)
            ax.grid(True, which="both")
 
        ax1.set_title(cfg.title, fontweight="bold")
        ax3.set_xlabel("Step")
        if first:
            ax1.set_ylabel("Recall@10")
            ax2.set_ylabel("QPS")
            ax3.set_ylabel("(update)\noperations/s")
 
    # ---- shared legend ----
    legend_keys = list(LEGEND_ORDER)
    if any_postgres:
        legend_keys.append("postgres")
    handles = [Line2D([0], [0], **_style(k)) for k in legend_keys]
    labels = [STYLE[k]["label"] for k in legend_keys]
    handles.append(Line2D([0], [0], color=REBUILD_COLOR, marker=REBUILD_MARKER,
                          ls="none", markersize=6, markeredgewidth=1.0))
    labels.append("rebuild")
 
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.96),
               columnspacing=1.2, handletextpad=0.5)
    return fig
 
 
# --------------------------------------------------------------------------- #
# Self-test: synthesize schema-matching files so the layout can be checked
# without the cluster's /dataset and /users trees.
# --------------------------------------------------------------------------- #
def _phase_op(step, boundaries):
    """insert in even phases, delete in odd phases (matches the shading)."""
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= step < boundaries[i + 1]:
            return "insert" if i % 2 == 0 else "delete"
    return "insert"
 
 
def _write_surge(path, boundaries, mode, param, rng, with_rebuilds, drift):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = boundaries[-1]
    steps = np.arange(0, n, 4)
 
    # rebuilds (only the with-maintenance run); recall recovers at each rebuild.
    if with_rebuilds:
        rebuild_steps = np.arange(120, n, 130)
    else:
        rebuild_steps = np.array([], dtype=int)
 
    def since_rebuild(s):
        prior = rebuild_steps[rebuild_steps <= s]
        return s - (prior.max() if prior.size else 0)
 
    rows = []
    floor = 0.78 - 0.12 * drift
    for s in steps:
        decay = 0.0009 * (since_rebuild(s) if with_rebuilds else s)
        theo = float(np.clip(0.99 - decay, floor, 0.99))
        meas = float(np.clip(theo - 0.02 + rng.normal(0, 0.004), 0, 1))
        rows.append({"step": int(s), "operation": "search", "mode": mode,
                     "param": param, "recall@10": meas,
                     "theoretical_recall@10": theo,
                     "throughput": float(2000 + rng.normal(0, 80))})
        op = _phase_op(s, boundaries)
        rows.append({"step": int(s), "operation": op, "mode": np.nan,
                     "param": np.nan, "recall@10": -1.0,
                     "theoretical_recall@10": -1.0,
                     "throughput": float(9000 + rng.normal(0, 400))})
    for rs in rebuild_steps:
        rows.append({"step": int(rs), "operation": "rebuild", "mode": np.nan,
                     "param": np.nan, "recall@10": -1.0,
                     "theoretical_recall@10": -1.0, "throughput": np.nan})
    pd.DataFrame(rows).to_csv(path, index=False)
 
 
def _write_gpann(path, boundaries, nprobe, rng):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = boundaries[-1]
    steps = np.arange(0, n, 4)
    rows = []
    for s in steps:
        theo = float(np.clip(0.95 - 0.0007 * s, 0.6, 0.95))
        rows.append({"step": int(s), "operation": "SEARCH", "nprobe": nprobe,
                     "recall@10": float(np.clip(theo - 0.03, 0, 1)),
                     "theoretical_recall": theo,
                     "qps_or_throughput": float(1500 + rng.normal(0, 60))})
        op = "INSERT" if _phase_op(s, boundaries) == "insert" else "DELETE"
        rows.append({"step": int(s), "operation": op, "nprobe": nprobe,
                     "recall@10": np.nan, "theoretical_recall": np.nan,
                     "qps_or_throughput": float(6000 + rng.normal(0, 300))})
    pd.DataFrame(rows).to_csv(path, index=False)
 
 
def _write_postgres(path, boundaries, rng):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = boundaries[-1]
    steps = np.arange(0, n, 4)
    with open(path, "w") as f:
        f.write(json.dumps({"type": "initialization"}) + "\n")
        for s in steps:
            f.write(json.dumps({
                "operation": "search", "step": int(s),
                "elapsed_s": 1.0, "count": int(500 + rng.normal(0, 20)),
                "recall_at_k": float(np.clip(0.9 - 0.0004 * s, 0.6, 0.95))}) + "\n")
            op = _phase_op(s, boundaries)
            f.write(json.dumps({
                "operation": op, "step": int(s),
                "elapsed_s": 1.0, "count": int(3000 + rng.normal(0, 150)),
                "recall_at_k": -1}) + "\n")
 
 
def make_selftest_datasets(tmp):
    global SURGE_ROOT
    SURGE_ROOT = os.path.join(tmp, "surge")
    demo = [
        ("MSTuring-100M-clustered", CLUSTERED_BOUNDARIES, True),
        ("MSTuring-100M-shift", SHIFT_BOUNDARIES, False),
        ("SIFT-100M-clustered", CLUSTERED_BOUNDARIES, False),
        ("SIFT-100M-shift", SHIFT_BOUNDARIES, False),
    ]
    specs = []
    for i, (title, boundaries, has_pg) in enumerate(demo):
        rng = np.random.default_rng(i)
        nm = os.path.join(tmp, f"surge/ds{i}_t10000-rebuildsbroken/results.csv")
        wm = os.path.join(tmp, f"surge/ds{i}_t6000/results.csv")
        gp = os.path.join(tmp, f"gpann/ds{i}.csv")
        _write_surge(nm, boundaries, "RecallTarget", 0.9, rng, False, drift=i)
        _write_surge(wm, boundaries, "RecallTarget", 0.9, rng, True, drift=i)
        _write_gpann(gp, boundaries, 5, rng)
        pg = None
        if has_pg:
            pg = os.path.join(tmp, f"postgres/ds{i}.json")
            _write_postgres(pg, boundaries, rng)
        specs.append(ColumnSpec(title=title, no_maint_csv=nm, with_maint_csv=wm,
                                gpann_csv=gp, postgres_json=pg,
                                phase_boundaries=boundaries))
    return specs
 
 
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="end_to_end_grid.pdf")
    ap.add_argument("--selftest", action="store_true",
                    help="render from synthetic data (no cluster files needed)")
    args = ap.parse_args()
 
    set_style()
    if args.selftest:
        with tempfile.TemporaryDirectory() as tmp:
            datasets = make_selftest_datasets(tmp)
            data = [load_column(cfg) for cfg in datasets]
            fig = build_figure(datasets, data)
            fig.savefig(args.output, bbox_inches="tight", dpi=150)
    else:
        data = [load_column(cfg) for cfg in DATASETS]
        fig = build_figure(DATASETS, data)
        fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")
 
 
if __name__ == "__main__":
    main()
