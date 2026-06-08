#!/usr/bin/env python3
"""
plot_runbook_grid.py

Consolidates the per-(dataset, runbook, mode) runbook plots of figure2 into two
full-width grids, one per routing mode:

    rows    = datasets      (SIFT-100M, MSTuring-100M)
    columns = runbook type  (Uniform -> Clustered -> Shift, by increasing drift)
    cell    = theoretical recall@10 (top)  over  a secondary strip (bottom)

    recall-target figure:  secondary strip = Partition Activation
                           series: with/no maintenance + their two oracles
    nprobe figure:         secondary strip = Partition Imbalance (Gini)
                           series: with/no maintenance + their two oracles + GP-ANN

Each figure has ONE shared legend on top and light insert/delete phase shading.
Intended to be included as a full-width figure* (\\textwidth) in a 2-column paper.

Data (mirrors plot_figure2.ipynb; lives on the cluster under DATA_ROOT):
  <cluster_history_dir>/full_results_RecallTarget_t{0,100}.csv
  <cluster_history_dir>/full_results_NProbe_t{0,100}[_with_oracle].csv
        columns: step, recall, activation, mode, param
        mode in {RecallTarget, Oracle, NProbe, NProbeOracle}
        t = 0  -> with maintenance (rebuild always);  t = 100 -> no maintenance
  <cluster_history_dir>/full_real_imbalance_{rebuilding,no_rebuilding}_results.csv
        columns: stepNum, count   (one row per shard per step; Gini taken per step)
  <gpann_dir>/<key>100M<runbook>_runbook_results_nprobe_with_theo_sweep.csv
        columns: operation, nprobe, step, theoretical_recall, shard_0_active..shard_9_active

Usage:
  python plot_runbook_grid.py -o runbook            # writes runbook_recalltarget.pdf + runbook_nprobe.pdf
  python plot_runbook_grid.py --mode recalltarget -o rt.pdf
  python plot_runbook_grid.py --selftest -o demo    # synthetic data, no cluster files
"""

import argparse
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def set_style():
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "grid.color": "lightgray",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


# Single source of truth for every series' style + label.
STYLE = {
    "with_maint":  dict(color="C0",        ls="-",  label="SURGE, fresh routing"),
    "no_maint":    dict(color="firebrick", ls="-",  label="SURGE, stale routing"),
    "oracle_maint": dict(color="C0",        ls="--", alpha=0.6, label="Oracle, fresh"),
    "oracle_no":   dict(color="firebrick", ls="--", alpha=0.6, label="Oracle, stale"),
    "gpann":       dict(color="C8",        ls="-",  label="GP-ANN"),
}
LEGEND_KEYS = {
    "recalltarget": ["with_maint", "no_maint", "oracle_maint", "oracle_no"],
    "nprobe":       ["with_maint", "no_maint", "oracle_maint", "oracle_no", "gpann"],
}
SECONDARY_YLABEL = {"recalltarget": "Partition\nActivation", "nprobe": "Partition\nImbalance"}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_ROOT = "/dataset/big-ann-benchmarks/data"     # overridden by --selftest

RECALL_TARGET_PARAM = 0.9


@dataclass
class DatasetSpec:
    title: str          # display name, e.g. "SIFT-100M"
    key: str            # path key: "bigann" or "msturing"
    nprobe_param: int   # nprobe used for this dataset (SIFT=3, MSTuring=5)
    # Recall y-limits, per mode, shared across this dataset's runbook columns so
    # workload differences stay comparable. SIFT and MSTuring get different ranges.
    recall_ylim: dict = field(default_factory=dict)


@dataclass
class RunbookSpec:
    title: str          # display name: "Uniform" / "Clustered" / "Shift"
    key: str            # path key: "random" / "clustered" / "shift"
    n_steps: int
    phase_period: Optional[int] = 128          # regular alternating ins/del blocks
    phases: Optional[list] = None              # explicit [(start, end, 'ins'|'del')]


DATASETS = [
    DatasetSpec("SIFT-100M", "bigann", nprobe_param=3,
                recall_ylim={"recalltarget": (0.835, 1.01), "nprobe": (0.68, 1.005)}),
    DatasetSpec("MSTuring-100M", "msturing", nprobe_param=5,
                recall_ylim={"recalltarget": (0.55, 1.005), "nprobe": (0.48, 1.005)}),
]

# Columns ordered by increasing distribution drift.
RUNBOOKS = [
    RunbookSpec("Uniform", "random", n_steps=1280, phase_period=128),
    RunbookSpec("Clustered", "clustered", n_steps=1280, phase_period=128),
    RunbookSpec("Shifting", "shift", n_steps=616, phase_period=None, phases=[
        (0, 208, "ins"), (208, 245, "del"), (245, 338, "ins"), (338, 364, "del"),
        (364, 468, "ins"), (468, 494, "del"), (494, 590, "ins"), (590, 616, "del"),
    ]),
]


# --------------------------------------------------------------------------- #
# Path construction (mirrors plot_figure2.ipynb naming)
# --------------------------------------------------------------------------- #
def _dataset_dir(ds, rb):
    if ds.key == "bigann":
        return os.path.join(DATA_ROOT, f"bigann-{rb.key}")
    return os.path.join(DATA_ROOT, f"MSTuring-100M-{rb.key}")


def cluster_history_dir(ds, rb):
    sub = (f"cluster_history_bigann-100M-{rb.key}_10000_10" if ds.key == "bigann"
           else f"cluster_history_msturing-100M-{rb.key}_10000_10")
    return os.path.join(_dataset_dir(ds, rb), sub)


def gpann_csv(ds, rb):
    name = f"{ds.key}100M{rb.key}_runbook_results_nprobe_with_theo_sweep.csv"
    return os.path.join(_dataset_dir(ds, rb), "gpann_partitions", name)


SHARD_COLS = [f"shard_{i}_active" for i in range(10)]


def gini(counts):
    arr = np.sort(np.asarray(counts, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n


def _gini_per_step(df):
    """df has columns stepNum, count (one row per shard). Return (steps, gini)."""
    steps, vals = [], []
    for step, grp in df.groupby("stepNum"):
        steps.append(step)
        vals.append(gini(grp["count"].values))
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(vals)[order]


def _read_surge(base, mode):
    """Read both t (with/no maintenance) CSVs for a mode, trying the oracle suffix."""
    frames = []
    for t in (0, 100):
        path = None
        for suffix in (("_with_oracle", "") if mode == "NProbe" else ("",)):
            cand = os.path.join(base, f"full_results_{mode}_t{t}{suffix}.csv")
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            continue
        df = pd.read_csv(path)
        df["t"] = t
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_runbook(ds, rb):
    base = cluster_history_dir(ds, rb)
    surge = pd.concat([_read_surge(base, "RecallTarget"), _read_surge(base, "NProbe")],
                      ignore_index=True)
    d = {"surge": surge}

    # imbalance (Gini per step): rebuilding = with maintenance (t0); no_rebuilding = t100
    reb = os.path.join(base, "full_real_imbalance_rebuilding_results.csv")
    noreb = os.path.join(base, "full_real_imbalance_no_rebuilding_results.csv")
    d["imb_t0"] = _gini_per_step(pd.read_csv(reb)) if os.path.exists(reb) else (None, None)
    d["imb_t100"] = _gini_per_step(pd.read_csv(noreb)) if os.path.exists(noreb) else (None, None)

    gp = gpann_csv(ds, rb)
    if os.path.exists(gp):
        g = pd.read_csv(gp)
        g["gini"] = g[SHARD_COLS].apply(lambda r: gini(r.values), axis=1)
        d["gpann"] = g
    else:
        d["gpann"] = None
    return d


# --------------------------------------------------------------------------- #
# Recall-gap analysis
#
# Same-step recall gaps that the paper quotes. The baseline is always the stale
# (no-maintenance, t=100) routing line; the reference is one of:
#
#   routing       fresh routing (t=0)  -- "stale falls short of a maintained system
#                                          by up to X pp"      (rebuild vs no rebuild)
#   oracle-stale  stale oracle  (t=100) -- "stale routing is X pp below the oracle":
#                                          recall lost purely to the routing decision,
#                                          holding the (stale) partition layout fixed
#   oracle-fresh  fresh oracle  (t=0)   -- recall lost vs the best a maintained layout
#                                          could achieve (layout + routing combined)
#
# Every delta is taken ONLY at runbook steps present in BOTH series (inner join on
# `step`), so it is always a same-step comparison -- we never subtract recall at
# step i from recall at step j. If the two series are sampled on different step
# grids, the unmatched steps are counted and flagged rather than interpolated.
# --------------------------------------------------------------------------- #
# (label, which series, t) for the reference line; baseline is always stale routing.
GAP_KINDS = {
    "routing":      ("rebuild vs no rebuild (fresh routing - stale routing)", "routing", 0),
    "oracle-stale": ("stale oracle - stale routing (routing-decision loss, fixed layout)", "oracle", 100),
    "oracle-fresh": ("fresh oracle - stale routing (loss vs best maintained layout)", "oracle", 0),
}


def _recall_line(surge, mode, ds, t, which):
    """[step, recall] for a SURGE series, sorted and de-duped on step.
    which in {'routing','oracle'};  t: 0 = fresh/maintained, 100 = stale."""
    if mode == "recalltarget":
        m, p = ("RecallTarget" if which == "routing" else "Oracle"), RECALL_TARGET_PARAM
    else:
        m, p = ("NProbe" if which == "routing" else "NProbeOracle"), ds.nprobe_param
    s = surge[(surge["t"] == t) & (surge["mode"] == m) & (surge["param"] == p)]
    return (s[["step", "recall"]]
            .dropna()
            .drop_duplicates("step")
            .sort_values("step")
            .reset_index(drop=True))


def _same_step_gap(ref, base):
    """Max/mean of (ref.recall - base.recall) over steps present in BOTH series."""
    if ref.empty or base.empty:
        return None
    m = ref.merge(base, on="step", suffixes=("_ref", "_base"))   # inner join => same step
    if m.empty:
        return None
    m["delta"] = m["recall_ref"] - m["recall_base"]
    i = int(m["delta"].idxmax())
    n_union = len(set(ref["step"]) | set(base["step"]))
    return {
        "matched_steps": len(m),
        "unmatched_steps": n_union - len(m),
        "max_gap": float(m.loc[i, "delta"]),
        "at_step": int(m.loc[i, "step"]),
        "ref_at_max": float(m.loc[i, "recall_ref"]),
        "base_at_max": float(m.loc[i, "recall_base"]),
        "mean_gap": float(m["delta"].mean()),
    }


def recall_gap(d, ds, mode, kind="routing"):
    """Same-step recall gap for one cell. baseline = stale routing (t=100);
    reference chosen by `kind` (see GAP_KINDS). Returns a summary dict or None."""
    surge = d["surge"]
    _, ref_which, ref_t = GAP_KINDS[kind]
    base = _recall_line(surge, mode, ds, t=100, which="routing")
    ref = _recall_line(surge, mode, ds, t=ref_t, which=ref_which)
    return _same_step_gap(ref, base)


def report_recall_gaps(loaded, modes, datasets, runbooks,
                       kinds=("routing", "oracle-stale", "oracle-fresh"), out_csv=None):
    """Print a per-cell table (and overall max) for each requested gap kind. Returns
    a combined DataFrame; if out_csv is set, all kinds are written with a `kind` column."""
    all_rows = []
    for kind in kinds:
        desc, _, _ = GAP_KINDS[kind]
        rows = []
        for mode in modes:
            for ds in datasets:
                for rb in runbooks:
                    dd = loaded.get((ds.title, rb.title))
                    if dd is None:
                        continue
                    g = recall_gap(dd, ds, mode, kind)
                    if g is not None:
                        rows.append({"kind": kind, "mode": mode, "dataset": ds.title,
                                     "runbook": rb.title, **g})
        if not rows:
            print(f"\n[{kind}] no data found")
            continue
        df = pd.DataFrame(rows)
        all_rows.extend(rows)

        print(f"\n=== Max recall gap [{kind}]: {desc}, same-step ===")
        hdr = (f"{'mode':<14}{'dataset':<15}{'runbook':<11}{'max gap':>9}{'pp':>7}"
               f"{'@step':>7}{'ref':>7}{'base':>7}{'mean':>8}{'matched':>9}{'unmatched':>10}")
        print(hdr)
        print("-" * len(hdr))
        for _, r in df.sort_values("max_gap", ascending=False).iterrows():
            warn = "  <-- step grids differ" if r["unmatched_steps"] else ""
            print(f"{r['mode']:<14}{r['dataset']:<15}{r['runbook']:<11}"
                  f"{r['max_gap']:>9.4f}{100 * r['max_gap']:>7.1f}{int(r['at_step']):>7}"
                  f"{r['ref_at_max']:>7.3f}{r['base_at_max']:>7.3f}{r['mean_gap']:>8.4f}"
                  f"{int(r['matched_steps']):>9}{int(r['unmatched_steps']):>10}{warn}")
        top = df.loc[df["max_gap"].idxmax()]
        print(f"Largest [{kind}] gap overall: {100 * top['max_gap']:.1f} pp "
              f"({top['dataset']}, {top['runbook']}, {top['mode']} mode) at step "
              f"{int(top['at_step'])} [ref {top['ref_at_max']:.3f} vs stale {top['base_at_max']:.3f}]")

    if not all_rows:
        return None
    out = pd.DataFrame(all_rows)
    if out_csv:
        out.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _line(ax, x, y, key):
    s = {k: v for k, v in STYLE[key].items() if k != "label"}
    ax.plot(x, y, **s)


def _row_label(ax, text):
    ax.text(-0.35, 0.3, text, transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontweight="bold", fontsize=10)



def _phases(rb):
    if rb.phases is not None:
        return rb.phases
    out, s, kind = [], 0, "ins"
    while s < rb.n_steps:
        e = min(s + rb.phase_period, rb.n_steps)
        out.append((s, e, kind))
        s, kind = e, ("del" if kind == "ins" else "ins")
    return out


def plot_cell(ax_top, ax_bot, ds, rb, d, mode):
    surge = d["surge"]

    if mode == "recalltarget":
        p = RECALL_TARGET_PARAM
        sel = lambda m, t: surge[(surge["t"] == t) & (surge["mode"] == m)
                                 & (surge["param"] == p)].sort_values("step")
        for t, key in ((0, "with_maint"), (100, "no_maint")):
            s = sel("RecallTarget", t)
            _line(ax_top, s["step"], s["recall"], key)
            _line(ax_bot, s["step"], s["activation"], key)
        for t, key in ((0, "oracle_maint"), (100, "oracle_no")):
            s = sel("Oracle", t)
            if len(s):
                _line(ax_top, s["step"], s["recall"], key)
                _line(ax_bot, s["step"], s["activation"], key)
        ax_bot.set_ylim(0, 0.705)

    else:  # nprobe
        p = ds.nprobe_param
        sel = lambda m, t: surge[(surge["t"] == t) & (surge["mode"] == m)
                                 & (surge["param"] == p)].sort_values("step")
        for t, key in ((0, "with_maint"), (100, "no_maint")):
            s = sel("NProbe", t)
            _line(ax_top, s["step"], s["recall"], key)
        for t, key in ((0, "oracle_maint"), (100, "oracle_no")):
            s = sel("NProbeOracle", t)
            if len(s):
                _line(ax_top, s["step"], s["recall"], key)
        # imbalance strip
        for (steps, vals), key in ((d["imb_t0"], "with_maint"), (d["imb_t100"], "no_maint")):
            if steps is not None:
                _line(ax_bot, steps, vals, key)
        # GP-ANN (recall + imbalance)
        g = d["gpann"]
        if g is not None:
            gg = g[g["operation"] == "SEARCH"] if "operation" in g.columns else g
            gg = gg[gg["nprobe"] == p].sort_values("step")
            _line(ax_top, gg["step"], gg["theoretical_recall"], "gpann")
            _line(ax_bot, gg["step"], gg["gini"], "gpann")
        ax_bot.set_ylim(0, 0.7)


def _stacked_columns(groups):
    """Flatten [[(h,l),...], ...] into legend (handles, labels, ncol) so that each
    inner group renders as its own stacked column. matplotlib fills column-major,
    so we pad each group to equal height with blank entries and concatenate."""
    blank = Line2D([], [], color="none", marker="none", linestyle="none")
    nrow = max(len(g) for g in groups)
    handles, labels = [], []
    for g in groups:
        for j in range(nrow):
            h, l = g[j] if j < len(g) else (blank, "")
            handles.append(h)
            labels.append(l)
    return handles, labels, len(groups)


def build_figure(mode, datasets, runbooks, loaded):
    nrows, ncols = len(datasets), len(runbooks)
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows - 2))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.15, wspace=0.2)

    for r, ds in enumerate(datasets):
        for c, rb in enumerate(runbooks):
            d = loaded[(ds.title, rb.title)]
            cell = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
            ax_top = fig.add_subplot(cell[0])
            ax_bot = fig.add_subplot(cell[1], sharex=ax_top)
            plt.setp(ax_top.get_xticklabels(), visible=False)

            plot_cell(ax_top, ax_bot, ds, rb, d, mode)
            ylim = ds.recall_ylim.get(mode)
            if ylim:
                ax_top.set_ylim(*ylim)

            for (s, e, kind) in _phases(rb):
                col = "green" if kind == "ins" else "red"
                ax_top.axvspan(s, e, color=col, alpha=0.05)
                ax_bot.axvspan(s, e, color=col, alpha=0.05)

            if r == 0:
                ax_top.set_title(rb.title)
            if c == 0:
                ax_top.set_ylabel("Theoretical\nRecall@10", fontsize=10)
                ax_bot.set_ylabel(SECONDARY_YLABEL[mode], fontsize=10)
                row_text = ds.title
                if mode == "nprobe":
                    row_text = f"{ds.title} (NProbe = {ds.nprobe_param})"
                if mode == "recalltarget":
                    row_text = f"{ds.title} (Target = {RECALL_TARGET_PARAM})"
                _row_label(ax_top, row_text)
            if r == nrows - 1:
                ax_bot.set_xlabel("Runbook Step")

    # Legend grouped into stacked columns so it stays narrower than the figure.
    def entry(key):
        return (Line2D([0], [0], **{k: v for k, v in STYLE[key].items() if k != "label"}),
                STYLE[key]["label"])

    routing = [entry("with_maint"), entry("no_maint")]
    oracles = [entry("oracle_maint"), entry("oracle_no")]
    phases = [(Patch(facecolor="green", alpha=0.25), "Insert phase"),
              (Patch(facecolor="red", alpha=0.25), "Delete phase")]
    groups = [routing, oracles]
    if mode == "nprobe":
        groups.append([entry("gpann")])     # single-entry column
    groups.append(phases)

    handles, labels, ncol = _stacked_columns(groups)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.91), columnspacing=1.8, handletextpad=0.5)
    return fig


# --------------------------------------------------------------------------- #
# Self-test: synthesize schema-matching files so the layout can be checked
# without the cluster's /dataset tree.
# --------------------------------------------------------------------------- #
def _synth_runbook(ds, rb, drift):
    """drift in {0,1,2} = uniform/clustered/shift; writes all CSVs for (ds, rb)."""
    base = cluster_history_dir(ds, rb)
    os.makedirs(base, exist_ok=True)
    rng = np.random.default_rng(hash((ds.key, rb.key)) % 2**32)
    step = np.arange(0, rb.n_steps, 4)
    saw = 0.03 * np.sin(step / 60.0)

    def recalls(base_no):
        rec_with = np.clip(0.96 + 0.005 * np.sin(step / 80.) + rng.normal(0, .003, step.size), 0, 1)
        rec_no = np.clip(base_no - 0.10 * drift + saw + rng.normal(0, .004, step.size), 0, 1)
        orc_with = np.clip(0.99 + 0.0 * step, 0, 1)
        orc_no = np.clip(0.965 - 0.005 * drift + 0 * step, 0, 1)
        return rec_with, rec_no, orc_with, orc_no

    # RecallTarget files (mode RecallTarget + Oracle), bottom metric = activation
    rw, rn, ow, on = recalls(0.97)
    act_with = np.clip(0.25 + 0.03 * drift + 0 * step, 0, 1)
    act_no = np.clip(0.45 + 0.0 * step, 0, 1)
    act_ow, act_on = np.clip(0.18 + 0 * step, 0, 1), np.clip(0.27 + 0 * step, 0, 1)
    for t, (rec, act, orc, oact) in ((0, (rw, act_with, ow, act_ow)),
                                     (100, (rn, act_no, on, act_on))):
        df = pd.concat([
            pd.DataFrame({"step": step, "recall": rec, "activation": act,
                          "mode": "RecallTarget", "param": RECALL_TARGET_PARAM}),
            pd.DataFrame({"step": step, "recall": orc, "activation": oact,
                          "mode": "Oracle", "param": RECALL_TARGET_PARAM}),
        ], ignore_index=True)
        df.to_csv(os.path.join(base, f"full_results_RecallTarget_t{t}.csv"), index=False)

    # NProbe files (mode NProbe + NProbeOracle)
    rw, rn, ow, on = recalls(0.95)
    for t, (rec, orc) in ((0, (rw, ow)), (100, (rn, on))):
        df = pd.concat([
            pd.DataFrame({"step": step, "recall": rec, "activation": ds.nprobe_param / 10.,
                          "mode": "NProbe", "param": ds.nprobe_param}),
            pd.DataFrame({"step": step, "recall": orc, "activation": ds.nprobe_param / 10.,
                          "mode": "NProbeOracle", "param": ds.nprobe_param}),
        ], ignore_index=True)
        df.to_csv(os.path.join(base, f"full_results_NProbe_t{t}_with_oracle.csv"), index=False)

    # imbalance files: 10 shard counts per step
    def imbalance_file(path, base_gini):
        rows = []
        for st in step:
            spread = base_gini + 0.0
            counts = np.abs(rng.normal(1000, 1000 * spread, 10))
            for sh, ct in enumerate(counts):
                rows.append({"stepNum": st, "shard": sh, "count": ct})
        pd.DataFrame(rows).to_csv(path, index=False)
    imbalance_file(os.path.join(base, "full_real_imbalance_rebuilding_results.csv"), 0.15 + 0.05 * drift)
    imbalance_file(os.path.join(base, "full_real_imbalance_no_rebuilding_results.csv"), 0.20 + 0.22 * drift)

    # GP-ANN (only 100M; here always present in selftest)
    gp = gpann_csv(ds, rb)
    os.makedirs(os.path.dirname(gp), exist_ok=True)
    grec = np.clip(0.93 - 0.09 * drift + saw, 0, 1)
    gdf = {"operation": "SEARCH", "nprobe": ds.nprobe_param, "step": step,
           "theoretical_recall": grec}
    gcounts = np.abs(rng.normal(1000, 1000 * (0.18 + 0.18 * drift), (step.size, 10)))
    for i in range(10):
        gdf[f"shard_{i}_active"] = gcounts[:, i]
    pd.DataFrame(gdf).to_csv(gp, index=False)


def selftest_setup(tmp):
    global DATA_ROOT
    DATA_ROOT = tmp
    for ds in DATASETS:
        for drift, rb in enumerate(RUNBOOKS):
            _synth_runbook(ds, rb, drift)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="runbook",
                    help="output path; for --mode both, used as a prefix")
    ap.add_argument("--mode", choices=["recalltarget", "nprobe", "both"], default="both")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gap-csv", default=None,
                    help="also write the per-cell recall-gap table to this CSV")
    ap.add_argument("--gap-kind", default="all",
                    choices=["routing", "oracle-stale", "oracle-fresh", "all"],
                    help="which same-step gap(s) to report (default: all)")
    args = ap.parse_args()

    set_style()
    modes = ["recalltarget", "nprobe"] if args.mode == "both" else [args.mode]
    gap_kinds = (["routing", "oracle-stale", "oracle-fresh"]
                 if args.gap_kind == "all" else [args.gap_kind])

    def render(loaded, ext):
        for m in modes:
            fig = build_figure(m, DATASETS, RUNBOOKS, loaded)
            if args.mode == "both":
                out = f"{os.path.splitext(args.output)[0]}_{m}{ext}"
            else:
                out = args.output
            fig.savefig(out, bbox_inches="tight", dpi=150)
            print(f"wrote {out}")

    if args.selftest:
        with tempfile.TemporaryDirectory() as tmp:
            selftest_setup(tmp)
            loaded = {(ds.title, rb.title): load_runbook(ds, rb)
                      for ds in DATASETS for rb in RUNBOOKS}
            render(loaded, ".png")
            report_recall_gaps(loaded, modes, DATASETS, RUNBOOKS,
                                kinds=gap_kinds, out_csv=args.gap_csv)
    else:
        loaded = {(ds.title, rb.title): load_runbook(ds, rb)
                  for ds in DATASETS for rb in RUNBOOKS}
        render(loaded, ".pdf")
        report_recall_gaps(loaded, modes, DATASETS, RUNBOOKS,
                           kinds=gap_kinds, out_csv=args.gap_csv)


if __name__ == "__main__":
    main()