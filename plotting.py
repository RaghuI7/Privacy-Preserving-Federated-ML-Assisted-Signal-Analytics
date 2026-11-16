"""Figures drawn ONLY from the run logs (Work Order section 1 rule 1 and section 13).

The rejected submission plotted from hardcoded result dictionaries and from generators
that fabricated curves and pseudo-membership-attacks. Nothing here can accept numbers
that are not in results/logs/*.jsonl: every entry point starts by calling
runlog.read_logs(), and a figure whose runs have not been logged is reported as
*skipped*, with the filter it looked for, instead of being drawn.

Each figure is written as PDF (vector, for the paper) and PNG (300 dpi, for the response
letter) under config.FIG_DIR and is returned in a manifest recording the stage, the
number of runs behind it and the exact log filter used, so every panel can be traced back
to the runs that produced it.

Field names come from harness._record(): the config block contributes `arch`,
`architecture`, `cell`, `method`, `seed`, `partition_id`, `z`, `C`, `K`, `T`, `delta`,
`split_kind`, `variant`, `channel_mode`, `feature_set`, `alpha`, `byz_attack`,
`byz_aggregator`, `byz_fraction`, `dp_level`, `mia_attack`; the result block contributes
`accuracy`, `f1`, `f1_macro`, `precision`, `recall`, `balanced_accuracy`,
`macro_specificity`, `kappa`, `roc_auc_ovr`, `nll`, `f1_per_class`,
`precision_per_class`, `recall_per_class`, `specificity_per_class`, `class_support`,
`confusion`, `eps`, `eps_total`, `eps_updates`, `eps_clip_fraction`, `alpha_star`,
`release_snr_mean`, `per_round_test_acc`, `per_round_precision`, `per_round_recall`,
`per_round_macro_f1`, `per_round_eps_cum`, `per_round_release_snr`, `eps_per_round`,
`clip_history`, `communication_payload_bytes`, `partition_heterogeneity` and, for the cost
stage, the nested `latency_cpu_batch1`, `memory_gpu`, `memory_cpu` blocks.
"""

import os

import numpy as np

from . import config, multiseed, runlog

# ── style ─────────────────────────────────────────────────────────────────────
FIGSIZE_1COL = 3.5
FIGSIZE_2COL = 7.16

# Okabe-Ito: colour-blind safe, and stable across figures (same method, same colour).
METHOD_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
                 "#000000", "#F0E442", "#66C2A5", "#FC8D62"]
_GRID = dict(linestyle=":", alpha=0.45)
_CAPSIZE = 2.0
# The three release supports the DP sweep measures. They pay the SAME certificate at a given
# z, which is why a privacy-utility figure must not group by epsilon alone: three mechanisms
# at one eps would average into a single marker and the comparison the resubmission rests on
# would vanish from the figure that is supposed to show it.
DP_SUPPORTS = ("all", "released-noise", "head")
DP_SUPPORT_LABELS = {"all": "whole model (as published)",
                     "released-noise": "sparse release",
                     "head": "head-only release"}
DP_SUPPORT_STYLES = ("-", "--", "-.")
DP_SUPPORT_MARKERS = ("o", "s", "^")

_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.labelsize": 9,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "errorbar.capsize": _CAPSIZE,
}

STATE = {"applied": False, "inline": None, "plt": None, "mpl": None}


def apply_style():
    """Set the figure style once per session. Returns False when matplotlib is absent."""
    if STATE["applied"]:
        return True
    try:
        import matplotlib
        try:
            from IPython import get_ipython
            in_notebook = get_ipython() is not None
        except Exception:
            in_notebook = False
        if not in_notebook:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("  plotting unavailable (%s)" % exc)
        return False
    plt.rcParams.update(_RC)
    STATE.update(applied=True, plt=plt, mpl=matplotlib,
                 inline=bool(in_notebook) if STATE["inline"] is None else STATE["inline"])
    return True


def _pyplot():
    if not STATE["applied"] and not apply_style():
        raise RuntimeError("matplotlib is not installed")
    return STATE["plt"]


def _show(fig, name):
    """Save PDF + 300 dpi PNG and, in a notebook, display the figure inline."""
    plt = _pyplot()
    config.ensure_dirs()
    pdf = os.path.join(config.FIG_DIR, name + ".pdf")
    png = os.path.join(config.FIG_DIR, name + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    if STATE["inline"] is None:
        try:
            from IPython import get_ipython
            STATE["inline"] = get_ipython() is not None
        except Exception:
            STATE["inline"] = False
    if STATE["inline"]:
        try:
            from IPython.display import display
            display(fig)
            plt.show()
        except Exception:
            plt.show()
    plt.close(fig)
    return pdf, png


def _finish(fig, name, info):
    """Attach the paths and the traceability fields to one figure record."""
    pdf, png = _show(fig, name)
    rec = dict(figure=name, path_pdf=pdf, path_png=png)
    rec.update(info)
    print("  [fig] %-34s <- %s" % (name, rec.get("log_filter", "")))
    return rec


# ── log access helpers ────────────────────────────────────────────────────────
def _arch_of(rec):
    return str(rec.get("architecture") or rec.get("arch") or "?")


def _cell_of(rec):
    return str(rec.get("cell") or rec.get("method") or "?")


def _support_of(rec):
    """Which coordinates a DP run released: the whole model, the sparse set, or the head.

    Read from the mechanism fields, never from the cell name, because the name is what a
    method is called and these are what it did.
    """
    s = rec.get("dp_support")
    if s in DP_SUPPORTS:
        return str(s)
    if rec.get("dp_head_only"):
        return "head"
    sp = rec.get("dp_sparsity")
    try:
        if sp is not None and float(sp) < 1.0:
            return "released-noise"
    except (TypeError, ValueError):
        pass
    return "all"


def _val(rec, metric):
    """A metric as float, or None when this record does not carry it."""
    v = rec.get(metric)
    if v is None or isinstance(v, (dict, list, tuple, bool, str)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _mean(vals):
    """Arithmetic mean of the values that exist; 0.0 for an empty group.

    Used for ordering only, so a cell with no logged reading sorts last rather than
    raising, and no plotted number comes from here.
    """
    v = [float(x) for x in vals if x is not None]
    return sum(v) / len(v) if v else 0.0


def _nested(rec, block, key):
    d = rec.get(block)
    if isinstance(d, dict):
        return _val(d, key)
    return None


def _summary(runs, key_fields, metric="accuracy"):
    """Group logged runs by `key_fields` and summarise the seed replicates."""
    groups = {}
    for r in runs:
        k = tuple(str(r.get(f, "?")) for f in key_fields)
        groups.setdefault(k, []).append(r)
    rows = []
    for k, g in sorted(groups.items()):
        vals = [_val(x, metric) for x in g]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        s = multiseed.summarise(vals, metric_name=metric)
        s["group"] = k
        s["metric"] = metric
        s.update(dict(zip(key_fields, k)))
        s["_runs"] = g
        rows.append(s)
    return rows


def _ci_bars(axes, xs, rows, metric="accuracy", color=None, width=0.62,
             show_points=True, label=None, alpha=0.85):
    """Bars of the seed mean with 95% CI whiskers; every individual reading as a dot.

    The dots are the reviewers' request (R2-12) made visible: a bar of means cannot show
    that a method won five seeds and lost one badly.
    """
    means = [row["mean"] for row in rows]
    low_err = [row["mean"] - row["ci95_low"] for row in rows]
    high_err = [row["ci95_high"] - row["mean"] for row in rows]
    col = color or METHOD_COLORS[0]
    bars = axes.bar(xs, means, color=col, width=width, alpha=alpha, label=label)
    axes.errorbar(xs, means, yerr=[low_err, high_err], fmt="none", ecolor="black",
                  elinewidth=0.9, capsize=_CAPSIZE, capthick=0.9, zorder=3)
    if show_points:
        rng = np.random.RandomState(0)
        for i, row in enumerate(rows):
            vals = [_val(r, metric) for r in row["_runs"]]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            jitter = np.clip(rng.uniform(-0.5, 0.5, len(vals)) * width * 0.75,
                             -width * 0.4, width * 0.4)
            axes.plot(np.asarray(xs, float)[i] + jitter, vals, "o", ms=3.0,
                      color="black", alpha=0.75, zorder=4)
    return bars, means


def _label_axis(axes, labels, title, ylabel, xlabel=None, ylim=None):
    axes.set_xticks(list(range(len(labels))))
    axes.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    axes.set_title(title, fontsize=9)
    axes.set_ylabel(ylabel)
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylim:
        axes.set_ylim(*ylim)
    axes.grid(axis="y", **_GRID)


def _skip(name, reason):
    print("  [fig] %-34s SKIPPED: %s" % (name, reason))
    return dict(figure=name, skipped=reason, n_runs=0)


def _n(runs):
    return len({(r.get("dataset"), r.get("stage"), _cell_of(r),
                 int(r.get("partition_id", 0)), int(r.get("seed", 0)))
               for r in runs})


ALL_DATASETS_DEFAULT = ("UCI HAR", "PAMAP2", "WISDM v2", "HARTH")


# ══ Task 4 - main results, every reading, curves, paired effects ══════════════
def fig_main_results(dataset, stages=("main",), metric="accuracy"):
    """Every method x architecture with 95% CI bars and per-seed dots (Tables 5, 6)."""
    plt = _pyplot()
    runs = [r for st in stages for r in runlog.read_logs(dataset=dataset, stage=st)]
    runs = [r for r in runs if _val(r, metric) is not None]
    if not runs:
        return _skip("fig_main_%s" % dataset.replace(" ", "_"),
                     "no logged runs for %s stage=%s" % (dataset, ",".join(stages)))
    rows = _summary(runs, ["architecture", "cell"], metric=metric)
    rows.sort(key=lambda r: -r["mean"])
    arch_labels = {}
    for r in rows:
        arch_labels.setdefault(str(r.get("architecture") or r.get("arch")),
                               len(arch_labels))
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.3))
    colors = [METHOD_COLORS[arch_labels[str(r.get("architecture")
                                          or r.get("arch"))] % len(METHOD_COLORS)]
              for r in rows]
    for i, row in enumerate(rows):
        _ci_bars(axes, [i], [row], metric=metric, color=colors[i], show_points=True)
    for arch, idx in sorted(arch_labels.items(), key=lambda kv: kv[1]):
        axes.plot([], [], "s", ms=7, color=METHOD_COLORS[idx % len(METHOD_COLORS)],
                  label=str(arch))
    axes.plot([], [], "o", ms=3, color="black", label="seed reading")
    axes.plot([], [], "|", ms=9, mec="black", color="none", label="95% CI (t, n seeds)")
    _label_axis(axes, ["%s\n%s" % (r.get("architecture") or r.get("arch"), r["cell"])
                       for r in rows],
                "%s: %s per method, mean over %d logged runs"
                % (dataset, metric, _n(runs)), metric, ylim=(0, 1.02))
    axes.legend(fontsize=7, frameon=False, ncol=4, loc="lower center",
                bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    return _finish(fig, "fig_main_%s" % dataset.replace(" ", "_"), dict(
        stage=",".join(stages), n_runs=_n(runs), cells=len(rows),
        log_filter="dataset=%s stage=%s metric=%s" % (dataset, ",".join(stages), metric)))


def fig_all_readings(dataset, metric="accuracy"):
    """Per-seed readings of every method, in one strip chart per cell."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset)
            if _val(r, metric) is not None and r.get("seed") is not None]
    per_cell = {}
    for r in runs:
        key = "%s | %s" % (_arch_of(r), _cell_of(r))
        per_cell.setdefault(key, {})[int(r["seed"])] = _val(r, metric)
    keep = {k: v for k, v in sorted(per_cell.items()) if len(v) >= 2}
    name = "fig_all_readings_%s" % dataset.replace(" ", "_")
    if not keep:
        return _skip(name, "no cell of %s has >= 2 logged seeds" % dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.4))
    keys = list(keep)
    half = max(0.28, min(0.45, 6.0 / max(1, len(keys))))
    for i, k in enumerate(keys):
        vals = list(keep[k].values())
        axes.add_patch(plt.Rectangle((i - half, np.percentile(vals, 25)), 2 * half,
                                     np.percentile(vals, 75) - np.percentile(vals, 25),
                                     facecolor=METHOD_COLORS[i % len(METHOD_COLORS)],
                                     edgecolor="none", alpha=0.35, zorder=1))
        axes.plot([i - half, i + half], [np.median(vals)] * 2, color="black", lw=1.1,
                  zorder=3)
        rng = np.random.RandomState(int(i))
        xs = i + np.clip(rng.uniform(-half, half, len(vals)), -half * 0.85, half * 0.85)
        axes.plot(xs, vals, "o", ms=3.4, color="black", alpha=0.8, zorder=4)
        axes.annotate("n=%d" % len(vals), (i, 0.02), xycoords=("data", "axes fraction"),
                      fontsize=6, ha="center", color="0.4")
    axes.set_xticks(list(range(len(keys))))
    axes.set_xticklabels(keys, rotation=50, ha="right", fontsize=7)
    axes.set_ylabel(metric)
    axes.set_title("%s: every logged reading (box = IQR, line = median, dot = seed)"
                   % dataset, fontsize=9)
    axes.grid(axis="y", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(n_runs=sum(len(v) for v in keep.values()),
                                   cells=len(keys),
                                   log_filter="dataset=%s all stages" % dataset))


def fig_training_curves(dataset, max_curves=40):
    """Per-round test accuracy for every logged run - the curves the old code faked."""
    plt = _pyplot()
    runs = runlog.read_logs(dataset=dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.0))
    drawn, seen = 0, set()
    for r in runs:
        curve = r.get("per_round_test_acc") or []
        curve = [c for c in curve if c is not None]
        if len(curve) < 2:
            continue
        key = (_arch_of(r), _cell_of(r), int(r.get("partition_id", 0)),
               int(r.get("seed", 0)))
        if key in seen or drawn >= max_curves:
            continue
        seen.add(key)
        axes.plot(range(1, len(curve) + 1), curve, lw=0.9,
                  color=METHOD_COLORS[drawn % len(METHOD_COLORS)],
                  label="%s/%s s%d" % (_arch_of(r), _cell_of(r), r.get("seed", 0)))
        drawn += 1
    name = "fig_curves_%s" % dataset.replace(" ", "_")
    if not drawn:
        return _skip(name, "no logged run for %s carries a per_round_test_acc curve"
                     % dataset)
    axes.set_xlabel("communication round")
    axes.set_ylabel("test accuracy")
    axes.set_title("%s: per-round test accuracy, %d logged runs%s"
                   % (dataset, drawn, " (capped)" if drawn >= max_curves else ""),
                   fontsize=9)
    axes.grid(True, **_GRID)
    if drawn <= 14:
        axes.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    return _finish(fig, name, dict(n_runs=drawn,
                                   log_filter="dataset=%s, per_round_test_acc logged"
                                   % dataset))


def fig_paired_forest(dataset):
    """Forest plot of the declared comparison family: paired mean difference and CI.

    Drawn from exp_main.paired_tests(), which recomputes the statistics from the logs,
    so the figure and the CSV cannot disagree.
    """
    plt = _pyplot()
    try:
        from .experiments import exp_main
    except Exception as exc:
        return _skip("fig_forest_%s" % dataset.replace(" ", "_"),
                     "exp_main unavailable (%s)" % exc)
    rows = [r for r in exp_main.paired_tests(dataset)
            if r.get("mean_diff") is not None]
    name = "fig_forest_%s" % dataset.replace(" ", "_")
    if not rows:
        return _skip(name, "no comparison has >= 2 paired seeds for %s" % dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 0.34 * len(rows) + 1.5))
    ys = np.arange(len(rows))[::-1]
    for y, row in zip(ys, rows):
        lo = row["mean_diff"] - row["diff_ci95_low"]
        hi = row["diff_ci95_high"] - row["mean_diff"]
        col = METHOD_COLORS[1] if row["mean_diff"] > 0 else METHOD_COLORS[0]
        axes.errorbar(row["mean_diff"], y, xerr=[[lo], [hi]], fmt="o", ms=5, lw=1.2,
                      color=col, capsize=3)
        axes.annotate("n=%d  p=%.3g%s" % (row["n_seeds"], row.get("holm_p", row["p_raw"]),
                                          " *" if (row.get("holm_p") or 1) < 0.05 else ""),
                      (row["diff_ci95_high"] + 0.004, y), va="center", fontsize=6.5)
    axes.axvline(0.0, color="black", lw=0.9, ls="--")
    axes.set_yticks(list(ys))
    axes.set_yticklabels([r["comparison"] for r in rows], fontsize=7)
    axes.set_xlabel("paired mean difference in test accuracy (first minus second)")
    axes.set_title("%s: comparison family, Holm-adjusted p (*)" % dataset, fontsize=9)
    axes.grid(axis="x", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(n_runs=len(rows),
                                   log_filter="dataset=%s stage=main paired_tests()"
                                   % dataset))


# ══ Task 1 / 2 - privacy, clipping ════════════════════════════════════════════
_METRIC_PRETTY = {"accuracy": "test accuracy", "f1": "weighted F1",
                  "f1_macro": "macro F1", "precision": "weighted precision",
                  "recall": "weighted recall", "balanced_accuracy": "balanced accuracy",
                  "macro_specificity": "macro specificity", "kappa": "Cohen's kappa",
                  "roc_auc_ovr": "one-vs-rest ROC-AUC", "nll": "negative log-likelihood"}


def _pretty(metric):
    return _METRIC_PRETTY.get(metric, metric)


def fig_privacy_utility(datasets=None, stage="privacy", metric="accuracy"):
    """A metric versus the certified cumulative epsilon, one panel per dataset x arch.

    Work Order line 65 asks for accuracy AND F1 against cumulative epsilon for all four
    datasets and all three architectures, so the metric is a parameter and the file name
    follows it: `fig_privacy_utility` for accuracy, `fig_privacy_utility_<metric>` for any
    other, so the two readings are never drawn over each other.
    """
    plt = _pyplot()
    datasets = list(datasets or ALL_DATASETS_DEFAULT)
    runs = [r for ds in datasets for r in runlog.read_logs(dataset=ds, stage=stage)]
    usable = [r for r in runs if _val(r, metric) is not None]
    name = ("fig_privacy_utility" if metric == "accuracy"
            else "fig_privacy_utility_%s" % metric)
    if not usable:
        return _skip(name, "no %s-stage runs logged for %s" % (stage, datasets))
    cells, baselines, zlabels = {}, {}, {}
    for r in usable:
        e = _val(r, "eps")
        key = (r["dataset"], _arch_of(r))
        if e is None:
            # no noise -> no finite certificate; keep it as the non-private reference line
            baselines.setdefault(key, []).append(_val(r, metric))
            continue
        # series are (release support, eps), not eps: see DP_SUPPORTS
        cells.setdefault(key, {}).setdefault((_support_of(r), e), []).append(
            _val(r, metric))
        zlabels.setdefault((key, _support_of(r), e), set()).add(
            "z=%g" % float(r["z"]) if r.get("z") is not None else "?")
    if not cells:
        return _skip(name, "privacy runs were logged without an eps certificate")
    keys = sorted(cells)
    ncol = 2
    nrow = int(np.ceil(len(keys) / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIGSIZE_2COL, 2.35 * nrow),
                             squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        supports = [s for s in DP_SUPPORTS if any(k[0] == s for k in cells[key])]
        for si, sup in enumerate(supports):
            pts = sorted(e for (s, e) in cells[key] if s == sup)
            means = [float(np.mean(cells[key][(sup, e)])) for e in pts]
            errs = []
            for e in pts:
                summ = multiseed.summarise(cells[key][(sup, e)], metric_name=metric)
                errs.append([summ["mean"] - summ["ci95_low"],
                             summ["ci95_high"] - summ["mean"]])
            ax.errorbar(pts, means, yerr=np.asarray(errs).T,
                        marker=DP_SUPPORT_MARKERS[si % len(DP_SUPPORT_MARKERS)],
                        ms=4, lw=1.2, ls=DP_SUPPORT_STYLES[si % len(DP_SUPPORT_STYLES)],
                        color=METHOD_COLORS[si % len(METHOD_COLORS)],
                        label=DP_SUPPORT_LABELS.get(sup, sup))
            for x, m in zip(pts, means):
                ax.annotate("/".join(sorted(zlabels[(key, sup, x)])), (x, m), fontsize=5.5,
                            xytext=(0, 5), textcoords="offset points", ha="center")
        bl = [_val(r, metric) for r in usable
              if (r["dataset"], _arch_of(r)) == key and _val(r, "eps") is None]
        if bl:
            ax.axhline(float(np.mean(bl)), ls="--", lw=0.9, color="black")
            ax.annotate("non-private %.3f" % float(np.mean(bl)),
                        (ax.get_xlim()[1], float(np.mean(bl))), fontsize=6, ha="right",
                        va="bottom", xytext=(0, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.invert_xaxis()      # stronger privacy to the right
        ax.set_xlabel("cumulative $\\varepsilon$ ($\\delta=10^{-5}$)")
        ax.set_ylabel(_pretty(metric))
        ax.set_title("%s / %s" % (key[0], key[1]), fontsize=8)
        ax.grid(True, **_GRID)
        if len(supports) > 1:
            ax.legend(fontsize=5.5, frameon=False)
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("%s versus privacy-utility from logged certificates (%d runs, %d points)"
                 % (_pretty(metric).capitalize(), _n(usable),
                    sum(len(v) for v in cells.values())), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, name, dict(stage=stage, metric=metric, n_runs=_n(usable),
                                   panels=len(keys),
                                   log_filter="stage=%s, eps certificate logged" % stage))


def fig_cross_dataset(datasets=None, stage="main", metric="accuracy"):
    """One cell, four datasets, one axis: the figure that makes the four comparable.

    Grouped bars per (architecture, cell), one bar per dataset, all four panels of the
    paper's argument on a single shared y-scale. This is the picture that answers the
    sheet's line 12 visually: the broken-pipeline WISDM sweeps and the fixed-pipeline
    Table 5 cannot sit side by side without the gap showing. A dataset with no logged run
    for a cell is a missing bar, never a bar of height zero.

    The dotted line is chance for the harmonised label set (1/N_CLASS), because the whole
    DP result is a statement about how far above chance a noisy release still gets.
    """
    plt = _pyplot()
    datasets = list(datasets or ALL_DATASETS_DEFAULT)
    runs_by_ds, by_ds = {}, {}
    for ds in datasets:
        runs = [r for r in runlog.read_logs(dataset=ds, stage=stage)
                if _val(r, metric) is not None]
        runs_by_ds[ds] = runs
        rows = _summary(runs, ["architecture", "cell"], metric=metric)
        # a cell can be present under both `architecture` spellings of an old log, so key
        # on the pair the summary already resolved
        by_ds[ds] = {(r["architecture"], r["cell"]): r for r in rows}
    present = [ds for ds in datasets if by_ds[ds]]
    n_runs = _n([r for ds in present for r in runs_by_ds[ds]])
    name = ("fig_cross_dataset" if metric == "accuracy"
            else "fig_cross_dataset_%s" % metric)
    if not present:
        return _skip(name, "no %s-stage run with a logged %s for %s"
                     % (stage, metric, datasets))
    cells = sorted({k for ds in present for k in by_ds[ds]})
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.4))
    width = min(0.8 / len(present), 0.2)
    for i, ds in enumerate(present):
        xs, rows = [], []
        for j, cell in enumerate(cells):
            row = by_ds[ds].get(cell)
            if row is None:
                continue
            xs.append(j + (i - (len(present) - 1) / 2.0) * width)
            rows.append(row)
        _ci_bars(axes, xs, rows, metric=metric, color=METHOD_COLORS[i], width=width,
                 show_points=False, label="%s (%d runs)" % (ds, _n(runs_by_ds[ds])))
    axes.axhline(1.0 / config.N_CLASS, ls=":", lw=1.0, color="black")
    axes.annotate("chance 1/%d" % config.N_CLASS, (axes.get_xlim()[1], 1.0 / config.N_CLASS),
                  fontsize=6, ha="right", va="bottom", xytext=(0, 2),
                  textcoords="offset points")
    axes.set_xticks(list(range(len(cells))))
    axes.set_xticklabels(["%s\n%s" % c for c in cells], rotation=0, ha="center", fontsize=6)
    axes.set_ylabel(_pretty(metric))
    axes.set_ylim(0, 1)
    axes.grid(True, axis="y", **_GRID)
    axes.legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("%s by dataset, one pipeline (%s stage, %d logged runs)"
                 % (_pretty(metric).capitalize(), stage, n_runs), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, name, dict(stage=stage, metric=metric, n_runs=n_runs,
                                   cells=len(cells), datasets=",".join(present),
                                   log_filter="stage=%s, dataset in %s, %s logged"
                                   % (stage, datasets, metric)))


def fig_cross_dataset_eps(datasets=None, stages=("privacy", "clip"), metric="accuracy"):
    """Cumulative epsilon versus a metric with the four datasets on one axis, per arch.

    `fig_privacy_utility` gives one panel per dataset, which is the right way to read a
    single dataset's mechanism comparison and the wrong way to compare two datasets: a
    reader cannot tell a level shift from an axis shift. Here every dataset shares the axes
    of its architecture panel, in the dataset's colour, and the release support is carried
    by the line style for the same reason as in `fig_privacy_utility` - the three supports
    pay the same certificate at a given z and must not average into one marker.
    """
    plt = _pyplot()
    datasets = list(datasets or ALL_DATASETS_DEFAULT)
    stages = list(stages or ["privacy"])
    runs = [r for st in stages for ds in datasets
            for r in runlog.read_logs(dataset=ds, stage=st)
            if _val(r, metric) is not None and _val(r, "eps") is not None]
    name = ("fig_cross_dataset_eps" if metric == "accuracy"
            else "fig_cross_dataset_eps_%s" % metric)
    if not runs:
        return _skip(name, "no %s-stage runs with both %s and an eps certificate for %s"
                     % (",".join(stages), metric, datasets))
    arches = sorted({_arch_of(r) for r in runs})
    ds_colors = {ds: METHOD_COLORS[i % len(METHOD_COLORS)] for i, ds in enumerate(datasets)}
    ncol = 2
    nrow = int(np.ceil(len(arches) / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIGSIZE_2COL, 2.6 * nrow),
                             squeeze=False)
    for i, arch in enumerate(arches):
        ax = axes[i // ncol][i % ncol]
        for ds in datasets:
            cells = {}
            for r in runs:
                if _arch_of(r) != arch or str(r.get("dataset")) != ds:
                    continue
                cells.setdefault((_support_of(r), _val(r, "eps")), []).append(
                    _val(r, metric))
            if not cells:
                continue
            for si, sup in enumerate(DP_SUPPORTS):
                pts = sorted(e for (s, e) in cells if s == sup)
                if not pts:
                    continue
                means, errs = [], []
                for e in pts:
                    summ = multiseed.summarise(cells[(sup, e)], metric_name=metric)
                    means.append(summ["mean"])
                    errs.append([summ["mean"] - summ["ci95_low"],
                                 summ["ci95_high"] - summ["mean"]])
                ax.errorbar(pts, means, yerr=np.asarray(errs).T,
                            marker=DP_SUPPORT_MARKERS[si % len(DP_SUPPORT_MARKERS)],
                            ms=4, lw=1.2, ls=DP_SUPPORT_STYLES[si % len(DP_SUPPORT_STYLES)],
                            color=ds_colors[ds],
                            label="%s %s" % (ds, DP_SUPPORT_LABELS.get(sup, sup)))
        ax.axhline(1.0 / config.N_CLASS, ls=":", lw=0.9, color="black")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("cumulative $\\varepsilon$ ($\\delta=10^{-5}$)")
        ax.set_ylabel(_pretty(metric))
        ax.set_ylim(0, 1)
        ax.set_title(arch, fontsize=8)
        ax.grid(True, **_GRID)
        ax.legend(fontsize=5, frameon=False, loc="lower left")
    for j in range(len(arches), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("%s versus cumulative $\\varepsilon$, all datasets on one scale"
                 " (%d certified runs)" % (_pretty(metric).capitalize(), _n(runs)),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, name, dict(stage=",".join(stages), metric=metric,
                                   n_runs=_n(runs), panels=len(arches),
                                   log_filter="stage in %s, eps certificate logged,"
                                   " %s logged" % (stages, metric)))


def fig_clip_sweep(dataset, stage="clip", metric="accuracy"):
    """Fixed-C sweep plus the adaptive and fixed arms at the same z."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _val(r, metric) is not None]
    name = "fig_clip_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no clip-stage runs logged for %s" % dataset)
    rows = _summary(runs, ["architecture", "cell"], metric=metric)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.1))
    colors = [METHOD_COLORS[0] if "| C=" in r["cell"] else
              (METHOD_COLORS[1] if r["cell"].endswith("adaptive") else METHOD_COLORS[2])
              for r in rows]
    for i, row in enumerate(rows):
        _ci_bars(axes, [i], [row], metric=metric, color=colors[i])
    _label_axis(axes, ["%s\n%s" % (r.get("architecture") or r.get("arch"), r["cell"])
                       for r in rows],
                "%s: clipping - fixed C sweep, then adaptive vs fixed at z=%.1f (%d runs)"
                % (dataset, config.DP_ARM_Z, _n(runs)), metric, ylim=(0, 1.02))
    axes.plot([], [], "s", ms=7, color=METHOD_COLORS[0], label="fixed C sweep")
    axes.plot([], [], "s", ms=7, color=METHOD_COLORS[1], label="adaptive")
    axes.plot([], [], "s", ms=7, color=METHOD_COLORS[2], label="fixed (default C)")
    axes.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs), cells=len(rows),
                                   log_filter="dataset=%s stage=clip" % dataset))


def fig_clip_history(dataset, stage="clip", arch="MLP"):
    """The logged C_r and clipped-fraction trajectory - raw, never smoothed."""
    plt = _pyplot()
    curves = []
    for r in runlog.read_logs(dataset=dataset, stage=stage):
        hist = r.get("clip_history")
        if isinstance(hist, list) and hist and str(r.get("cell", "")).startswith(arch):
            curves.append((int(r.get("seed", 0)), hist))
    name = "fig_clip_history_%s" % dataset.replace(" ", "_")
    if not curves:
        return _skip(name, "no clip_history logged for %s/%s (written by the adaptive "
                           "and fixed clip arms)" % (dataset, arch))
    curves.sort(key=lambda c: c[0])
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_2COL, 2.7))
    frozen_at = None
    for i, (seed, hist) in enumerate(curves[:8]):
        col = METHOD_COLORS[i % len(METHOD_COLORS)]
        cs = [_val(h, "C") for h in hist]
        fr = [_val(h, "clip_fraction") for h in hist]
        xs = [int(h.get("round", j + 1)) for j, h in enumerate(hist)]
        axes[0].plot(xs, cs, lw=1.1, color=col, label="seed %d" % seed)
        axes[1].plot(xs, fr, lw=1.1, color=col)
        rel = [x for x, h in zip(xs, hist) if h.get("estimator_released")]
        if rel and len(rel) < len(xs):
            frozen_at = max(frozen_at or 0, max(rel))
    if frozen_at is not None:
        # where the clip-fraction estimator stopped releasing; after this the line is a
        # fixed radius chosen by the first rounds, and only those releases are accounted
        axes[0].axvline(frozen_at + 0.5, ls=":", lw=0.9, color="grey")
        axes[0].text(frozen_at + 0.8, axes[0].get_ylim()[1],
                     "estimator stopped", fontsize=6, va="top", color="grey")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("$C_r$")
    axes[0].set_title("%s/%s: clip norm as logged" % (dataset, arch), fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    axes[1].axhline(config.ADAPT_TARGET_Q, ls="--", lw=0.9, color="black")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("clipped fraction")
    axes[1].set_title("fraction clipped (dashed = target $\\gamma$=%.2f)"
                      % config.ADAPT_TARGET_Q, fontsize=8)
    for ax in axes:
        ax.grid(True, **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=len(curves),
                                   log_filter="dataset=%s stage=clip cell startswith %s"
                                   % (dataset, arch)))


def fig_eps_split(dataset, stage="clip"):
    """Budget decomposition: epsilon of the updates vs of the clip-fraction estimator."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _val(r, "eps_total") is not None]
    name = "fig_eps_split_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no adaptive-accounted run logged for %s (adaptive=True "
                            "writes eps_updates/eps_clip_fraction/eps_total)" % dataset)
    rows = _summary(runs, ["architecture", "cell"], metric="eps_total")
    up = [float(np.mean([_val(r, "eps_updates") or np.nan for r in x["_runs"]]))
          for x in rows]
    cl = [float(np.mean([_val(r, "eps_clip_fraction") or np.nan for r in x["_runs"]]))
          for x in rows]
    tot = [x["mean"] for x in rows]
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 2.9))
    xs = list(range(len(rows)))
    axes.bar(xs, up, color=METHOD_COLORS[0], width=0.6, label="$\\varepsilon$ updates")
    axes.bar(xs, cl, bottom=up, color=METHOD_COLORS[1], width=0.6,
             label="$\\varepsilon$ clip-fraction estimator")
    for i, t in enumerate(tot):
        axes.annotate("$\\varepsilon$=%.2f\n$\\alpha^*$=%s"
                      % (t, _fmt_star(rows[i]["_runs"])),
                      (i, (up[i] + cl[i]) * 1.03), ha="center", va="bottom", fontsize=7)
    axes.set_xticks(xs)
    axes.set_xticklabels(["%s\n%s" % (r.get("architecture") or r.get("arch"), r["cell"])
                          for r in rows], rotation=40, ha="right", fontsize=7)
    axes.set_ylabel("cumulative $\\varepsilon$")
    axes.set_title("%s: adaptive accounting, composed at the shared $\\alpha^*$ "
                   "(%d runs)" % (dataset, _n(runs)), fontsize=9)
    axes.legend(frameon=False, fontsize=8)
    axes.grid(axis="y", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs),
                                   log_filter="dataset=%s stage=clip eps_total logged"
                                   % dataset))


def _fmt_star(runs):
    vals = [_val(r, "alpha_star") for r in runs]
    vals = [v for v in vals if v is not None]
    return "%.2f" % float(np.mean(vals)) if vals else "n/a"


# ══ Task 3 - splits ═══════════════════════════════════════════════════════════
def fig_leakage(datasets=None, stage="splits", metric="accuracy"):
    """Window-level (leaking) vs subject-disjoint accuracy, paired per seed."""
    plt = _pyplot()
    datasets = list(datasets or ALL_DATASETS_DEFAULT)
    cells = {}
    total = 0
    for ds in datasets:
        for r in runlog.read_logs(dataset=ds, stage=stage):
            cell = _cell_of(r)
            if not (cell.endswith("| window-split") or cell.endswith("| subject-split")):
                continue
            if _val(r, metric) is None:
                continue
            kind = "window" if cell.endswith("window-split") else "subject"
            arch = cell.split("|")[0].strip()
            cells.setdefault((ds, arch), {}).setdefault(kind, {})[int(r["seed"])] = \
                _val(r, metric)
            total += 1
    name = "fig_leakage"
    pairs = {k: v for k, v in cells.items() if len(v.get("window", {})) and
             len(v.get("subject", {}))}
    if not pairs:
        return _skip(name, "no dataset has both window-split and subject-split runs")
    keys = sorted(pairs)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 0.5 * len(keys) + 1.6))
    ys = np.arange(len(keys))[::-1]
    for y, k in zip(ys, keys):
        w, s = pairs[k]["window"], pairs[k]["subject"]
        seeds = sorted(set(w) & set(s))
        gap = np.array([w[x] - s[x] for x in seeds])
        summ = multiseed.summarise(gap, metric_name="gap")
        col = METHOD_COLORS[1] if gap.mean() > 0 else METHOD_COLORS[0]
        axes.errorbar(gap.mean(), y,
                      xerr=[[summ["mean"] - summ["ci95_low"]],
                            [summ["ci95_high"] - summ["mean"]]],
                      fmt="o", ms=5, color=col, capsize=3)
        for x in seeds:
            axes.plot(w[x] - s[x], y + 0.12, "|", ms=7, color="black", alpha=0.6)
        axes.annotate("n=%d seeds" % len(seeds), (0.985, y),
                      xycoords=("axes fraction", "data"), fontsize=6, ha="right",
                      color="0.4")
    axes.axvline(0.0, color="black", lw=0.9, ls="--")
    axes.set_yticks(list(ys))
    axes.set_yticklabels(["%s / %s" % (k[0], k[1]) for k in keys], fontsize=7)
    axes.set_xlabel("accuracy(window-level) - accuracy(subject-disjoint)")
    axes.set_title("Split leakage gap: positive means the window split flattered "
                   "the model (%d runs)" % total, fontsize=9)
    axes.grid(axis="x", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=total,
                                   log_filter="stage=%s cell endswith window-split / "
                                              "subject-split" % stage))


def fig_loso(dataset, stage="splits", metric="accuracy"):
    """Leave-one-subject-out: held-out accuracy distribution per architecture."""
    plt = _pyplot()
    groups = {}
    for r in runlog.read_logs(dataset=dataset, stage=stage):
        cell = _cell_of(r)
        if "loso" not in cell or _val(r, metric) is None:
            continue
        groups.setdefault(cell.split("|")[0].strip(), []).append(_val(r, metric))
    name = "fig_loso_%s" % dataset.replace(" ", "_")
    if not groups:
        return _skip(name, "no loso-subjN cells logged for %s" % dataset)
    keys = sorted(groups)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 2.8))
    bp = axes.boxplot([groups[k] for k in keys], widths=0.5, patch_artist=True,
                      medianprops=dict(color="black"))
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(METHOD_COLORS[i % len(METHOD_COLORS)])
        p.set_alpha(0.4)
    rng = np.random.RandomState(1)
    for i, k in enumerate(keys):
        axes.plot((i + 1) + rng.uniform(-0.16, 0.16, len(groups[k])), groups[k], "o",
                  ms=3.2, color="black", alpha=0.8)
    axes.set_xticks(list(range(1, len(keys) + 1)))
    axes.set_xticklabels(["%s\n(n=%d folds)" % (k, len(groups[k])) for k in keys],
                         rotation=30, ha="right", fontsize=7)
    axes.set_ylabel("held-out subject %s" % metric)
    axes.set_title("%s: leave-one-subject-out, one point per unseen subject" % dataset,
                   fontsize=9)
    axes.grid(axis="y", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=sum(len(v) for v in groups.values()),
                                   log_filter="dataset=%s stage=%s cell contains loso"
                                   % (dataset, stage)))


def fig_matched(dataset, stage="splits", metric="accuracy"):
    """Architecture-matched and representation-matched arms side by side."""
    plt = _pyplot()
    rows = _summary([r for r in runlog.read_logs(dataset=dataset, stage=stage)
                     if _val(r, metric) is not None and
                     ("matched" in _cell_of(r))],
                    ["architecture", "cell"], metric=metric)
    name = "fig_matched_%s" % dataset.replace(" ", "_")
    if not rows:
        return _skip(name, "no arch-matched / rep-matched runs logged for %s" % dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.0))
    for i, row in enumerate(rows):
        _ci_bars(axes, [i], [row], metric=metric,
                 color=METHOD_COLORS[i % len(METHOD_COLORS)])
    _label_axis(axes, ["%s\n%s" % (r.get("architecture"), r["cell"]) for r in rows],
                "%s: matched comparisons (%d runs; same rounds, epochs, batch, lr)"
                % (dataset, _n([r for r in runlog.read_logs(dataset=dataset, stage=stage)
                                if "matched" in _cell_of(r)])), metric, ylim=(0, 1.02))
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_cells=len(rows),
                                   log_filter="dataset=%s stage=%s cell contains matched"
                                   % (dataset, stage)))


# ══ Task 5 - non-IID ═════════════════════════════════════════════════════════
def fig_noniid(dataset, stage="noniid", metric="accuracy"):
    """Accuracy vs Dirichlet alpha per method; spread is over independent partitions."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _val(r, metric) is not None]
    name = "fig_noniid_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no noniid runs logged for %s" % dataset)
    def arm(r):
        # The DP and non-DP arms of Task 5 share a `method` string; without the suffix
        # they would be averaged into one curve.
        return "%s%s" % (r.get("method"), "+DP" if r.get("z") else "")
    methods = sorted({arm(r) for r in runs})
    order = [a for a in ("0.05", "0.1", "0.3", "0.5", "1.0", "5.0", "IID")
             if any(str(r.get("alpha")) == a for r in runs)]
    order += sorted({str(r.get("alpha")) for r in runs} - set(order))
    if not order:
        return _skip(name, "the %s logs carry no alpha column" % dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 2.9))
    xi = {a: i for i, a in enumerate(order)}
    for j, meth in enumerate(methods):
        means, mins, maxs, ns = [], [], [], []
        for a in order:
            vals = [_val(r, metric) for r in runs
                    if str(r.get("alpha")) == a and arm(r) == meth]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            s = multiseed.summarise(vals, metric_name=metric)
            means.append((xi[a], s["mean"]))
            mins.append(min(vals))
            maxs.append(max(vals))
            ns.append(len(vals))
        if not means:
            continue
        xs = [m[0] for m in means]
        col = METHOD_COLORS[j % len(METHOD_COLORS)]
        axes.fill_between(xs, mins, maxs, alpha=0.15, color=col)
        axes.plot(xs, [m[1] for m in means], "-o", ms=4, color=col, label=meth)
        for i, x in enumerate(xs):
            axes.plot([x, x], [mins[i], maxs[i]], "-", lw=0.8, color=col, alpha=0.5)
            axes.annotate("n=%d" % ns[i], (x, maxs[i]), fontsize=5.5, ha="center",
                          va="bottom", color="0.35")
    axes.set_xticks(list(range(len(order))))
    axes.set_xticklabels(["$\\alpha$=" + a for a in order])
    axes.set_ylabel("%s (mean over partitions)" % metric)
    axes.set_title("%s: label heterogeneity, thin line = min-max over partitions\n"
                   "(+DP curves: a Dirichlet client mixes subjects, so its eps is not a "
                   "user-level certificate)" % dataset, fontsize=8)
    axes.legend(fontsize=7, frameon=False, title="aggregation")
    axes.grid(True, **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs),
                                   log_filter="dataset=%s stage=%s" % (dataset, stage)))


# ══ Task 6 - Byzantine ═══════════════════════════════════════════════════════
def fig_byzantine(dataset, stage="byzantine", metric="accuracy"):
    """Clean accuracy vs adversary fraction: one line per attack, one panel per aggregator."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _val(r, metric) is not None and r.get("byz_attack") is not None]
    if not runs:
        return _skip("fig_byzantine_%s" % dataset.replace(" ", "_"),
                     "no byzantine-stage runs logged for %s" % dataset)
    out = []
    cells = [(arch, part) for arch in sorted({_arch_of(r) for r in runs})
             for part in sorted({str(r.get("byz_partition") or "iid") for r in runs})]
    for arch, part in cells:
        # One figure per (architecture, partition kind): an IID curve and a Dirichlet
        # curve pooled together would hide exactly the difference Task 6 asks about.
        sub = [r for r in runs if _arch_of(r) == arch
               and str(r.get("byz_partition") or "iid") == part]
        aggs = list(config.BYZ_AGGREGATORS)
        aggs = [a for a in aggs if any(str(r.get("byz_aggregator")) == a for r in sub)] or \
            sorted({str(r.get("byz_aggregator")) for r in sub})
        attacks = [a for a in config.BYZ_ATTACKS
                   if any(str(r.get("byz_attack")) == a for r in sub)]
        fig, axes = plt.subplots(1, len(aggs), figsize=(FIGSIZE_2COL, 2.7),
                                 squeeze=False, sharey=True)
        drawn = 0
        for i, agg in enumerate(aggs):
            ax = axes[0][i]
            for j, atk in enumerate(attacks):
                pts = {}
                for r in sub:
                    if str(r.get("byz_aggregator")) != agg or str(r.get("byz_attack")) != atk:
                        continue
                    pts.setdefault(float(r.get("byz_fraction") or 0.0), []).append(
                        _val(r, metric))
                if not pts:
                    continue
                xs = sorted(pts)
                ms = [float(np.mean(pts[x])) for x in xs]
                sd = [float(np.std(pts[x], ddof=1)) if len(pts[x]) > 1 else 0.0
                      for x in xs]
                ax.errorbar(xs, ms, yerr=sd, marker="o", ms=3.5, lw=1.1,
                            color=METHOD_COLORS[j % len(METHOD_COLORS)], label=atk)
                drawn += sum(len(v) for v in pts.values())
            ax.set_xlabel("adversary fraction")
            ax.set_title(str(agg), fontsize=8)
            ax.set_xticks(list(config.BYZ_FRACTIONS))
            ax.set_xticklabels(["%.1f" % f for f in config.BYZ_FRACTIONS])
            ax.grid(True, **_GRID)
        axes[0][0].set_ylabel("%s (mean $\\pm$ std over seeds)" % metric)
        axes[0][0].legend(fontsize=6, frameon=False, title="attack", ncol=2)
        fig.suptitle("%s / %s / %s: Byzantine robustness, Krum assumes f=%d (%d runs)"
                     % (dataset, arch, part, config.BYZ_ASSUMED_F, drawn), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        name = "fig_byzantine_%s_%s_%s" % (dataset.replace(" ", "_"),
                                           arch.replace(" ", ""), part)
        out.append(_finish(fig, name, dict(stage=stage, n_runs=drawn, arch=arch,
                                           partition=part,
                                           log_filter="dataset=%s stage=%s arch=%s "
                                                      "byz_partition=%s"
                                                      % (dataset, stage, arch, part))))
    return out


# ══ Task 7 - resampling and channels ═════════════════════════════════════════
def fig_resampling(datasets=None, stage="resampling", metric="accuracy"):
    """50 Hz / native / band-limited arms, per dataset, non-DP and DP side by side."""
    plt = _pyplot()
    datasets = [d for d in (datasets or list(ALL_DATASETS_DEFAULT))]
    per_ds = {}
    total = 0
    for ds in datasets:
        flat = [dict(r, dp_flag=_dp_of(r))
                for r in runlog.read_logs(dataset=ds, stage=stage)
                if _val(r, metric) is not None and not r.get("channel_mode")]
        rows = _summary(flat, ["architecture", "variant", "dp_flag"], metric=metric)
        if rows:
            per_ds[ds] = rows
            total += _n(flat)
    name = "fig_resampling"
    if not per_ds:
        return _skip(name, "no resampling-stage runs logged (variant column empty)")
    keys = list(per_ds)
    fig, axes = plt.subplots(1, len(keys), figsize=(FIGSIZE_2COL, 2.9), squeeze=False)
    for i, ds in enumerate(keys):
        ax = axes[0][i]
        rows = per_ds[ds]
        variants = list(dict.fromkeys(r["variant"] for r in rows))
        arms = list(dict.fromkeys("%s / %s" % (r["architecture"], r["dp_flag"])
                                 for r in rows))
        width = 0.8 / max(1, len(arms))
        for j, arm in enumerate(arms):
            for k, v in enumerate(variants):
                hit = [r for r in rows if r["variant"] == v and
                       "%s / %s" % (r["architecture"], r["dp_flag"]) == arm]
                if not hit:
                    continue
                x = j + (k - (len(variants) - 1) / 2.0) * width
                _ci_bars(ax, [x], [hit[0]], metric=metric,
                         color=METHOD_COLORS[j % len(METHOD_COLORS)], width=width * 0.9,
                         show_points=False,
                         label=arm if k == 0 else None)
        ax.set_xticks(list(range(len(arms))))
        ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel(metric)
        ax.set_title("%s: %s" % (ds, ", ".join(variants)), fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        ax.grid(axis="y", **_GRID)
    fig.suptitle("Task 7 sampling-rate ablation (%d runs)" % total, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, name, dict(stage=stage, n_runs=total,
                                   log_filter="stage=%s variant logged, channel_mode "
                                              "absent" % stage))


def _dp_of(r):
    return "DP z=%g" % float(r.get("z") or 0) if r.get("z") else "non-DP"


def fig_channel_treatment(dataset="WISDM v2", stage="resampling", metric="accuracy"):
    """Zero-pad vs accelerometer-only vs channel mask, non-DP and at one z."""
    plt = _pyplot()
    runs = [dict(r, dp_flag=_dp_of(r)) for r in runlog.read_logs(dataset=dataset,
                stage=stage) if r.get("channel_mode") and _val(r, metric) is not None]
    name = "fig_channels_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no channel_mode runs logged for %s (exp_resampling."
                            "channel_ablation writes them on raw windows)" % dataset)
    rows = _summary(runs, ["architecture", "channel_mode", "dp_flag"], metric=metric)
    modes = list(dict.fromkeys(r["channel_mode"] for r in rows))
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.1))
    for i, row in enumerate(rows):
        _ci_bars(axes, [i], [row], metric=metric,
                 color=METHOD_COLORS[modes.index(row["channel_mode"])
                                     % len(METHOD_COLORS)])
    _label_axis(axes, ["%s\n%s / %s" % (r.get("architecture"), r["channel_mode"],
                                        r["dp_flag"]) for r in rows],
                "%s: missing-gyroscope treatment on raw windows (%d runs)"
                % (dataset, _n(runs)), metric, ylim=(0, 1.02))
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs), cells=len(rows),
                                   log_filter="dataset=%s stage=%s channel_mode logged"
                                   % (dataset, stage)))


# ══ Task 8 - feature families ════════════════════════════════════════════════
def fig_feature_families(dataset, stage="features", metric="accuracy"):
    """Leave-one-family-out and single-family accuracy against the all-families arm."""
    plt = _pyplot()
    runs = [dict(r, dp_flag=_dp_of(r)) for r in runlog.read_logs(dataset=dataset,
                stage=stage) if r.get("feature_set") and _val(r, metric) is not None]
    name = "fig_features_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no feature-stage runs logged for %s (feature_set column "
                           "empty)" % dataset)
    rows = _summary(runs, ["feature_set", "dp_flag"], metric=metric)
    dps = sorted({r["dp_flag"] for r in rows})
    base = {r["dp_flag"]: r["mean"] for r in rows if r["feature_set"] == "all_families"}
    panels = [("leave-one-family-out",
               [r for r in rows if r["feature_set"].startswith("lofo_")]),
              ("single family only",
               [r for r in rows if r["feature_set"].startswith("only_")])]
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_2COL, 3.2))
    for ax, (title, sel) in zip(axes, panels):
        fams = list(dict.fromkeys(_short_fam(r["feature_set"]) for r in sel))
        for j, dp in enumerate(dps):
            hits = [r for r in sel if r["dp_flag"] == dp]
            for k, r in enumerate(hits):
                x = (fams.index(_short_fam(r["feature_set"]))
                     + (j - (len(dps) - 1) / 2.0) * 0.36)
                _ci_bars(ax, [x], [r], metric=metric,
                         color=METHOD_COLORS[j % len(METHOD_COLORS)], width=0.32,
                         show_points=False, label=dp if k == 0 else None)
            if dp in base:
                ax.axhline(base[dp], ls="--", lw=0.8,
                           color=METHOD_COLORS[j % len(METHOD_COLORS)])
        ax.set_xticks(list(range(len(fams))))
        ax.set_xticklabels(fams, rotation=55, ha="right", fontsize=6.5)
        ax.set_title("%s (dashed = all families, same privacy arm)" % title, fontsize=8)
        ax.set_ylabel(metric)
        ax.grid(axis="y", **_GRID)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("%s: feature families, %d runs (n_features is logged per run)"
                 % (dataset, _n(runs)), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs),
                                   n_families=len({r["feature_set"] for r in rows}),
                                   log_filter="dataset=%s stage=%s feature_set logged"
                                   % (dataset, stage)))


def _short_fam(fs):
    return str(fs).replace("lofo_", "w/o ").replace("only_", "")


# ══ Task 9 - deployment cost ═════════════════════════════════════════════════
def fig_compute_cost(dataset, stage="cost"):
    """Parameters / MACs / latency / memory / payload for CNN-BiLSTM vs LightTCN."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _arch_of(r) != "?"]
    name = "fig_cost_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no cost-stage runs logged for %s" % dataset)
    metrics = [
        ("parameters", lambda r: _val(r, "params")),
        ("MACs / window", lambda r: _val(r, "macs")),
        ("CPU latency (ms, median)", lambda r: _nested(r, "latency_cpu_batch1",
                                                       "latency_ms_median")),
        ("host RSS (kB)", lambda r: _nested(r, "memory_cpu", "host_rss_delta_kb")),
        ("payload / round (bytes)", lambda r: _val(r, "uplink_downlink_bytes_per_round")),
        ("local epoch (s)", lambda r: _val(r, "local_epoch_time_s")),
    ]
    have = [(lab, get) for lab, get in metrics if any(get(r) is not None for r in runs)]
    if not have:
        return _skip(name, "cost logs carry none of the measured columns")
    arches = sorted({_arch_of(r) for r in runs})
    fig, axes = plt.subplots(1, len(have), figsize=(FIGSIZE_2COL, 2.6), squeeze=False)
    vals_by_metric = {}
    for i, (lab, get) in enumerate(have):
        ax = axes[0][i]
        vals = []
        for a in arches:
            v = [get(r) for r in runs if _arch_of(r) == a and get(r) is not None]
            vals.append(float(np.mean(v)) if v else np.nan)
        vals_by_metric[lab] = dict(zip(arches, vals))
        good = [(j, v) for j, v in enumerate(vals) if v == v and v > 0]
        ax.bar([j for j, _ in good], [v for _, v in good],
               color=[METHOD_COLORS[j % len(METHOD_COLORS)] for j, _ in good])
        ax.set_yscale("log")
        ax.set_xticks(list(range(len(arches))))
        ax.set_xticklabels(arches, fontsize=6.5, rotation=30, ha="right")
        ax.set_title(lab, fontsize=7.5)
        for j, v in good:
            ax.annotate("%.3g" % v, (j, v), ha="center", va="bottom", fontsize=6)
    red = _reduction(vals_by_metric)
    fig.suptitle("%s: measured deployment cost - %s (paper claims ~30%% fewer params)"
                 % (dataset, red or "no CNN/TCN pair logged"), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _finish(fig, name, dict(stage=stage, n_runs=len(runs),
                                   reductions=vals_by_metric,
                                   log_filter="dataset=%s stage=%s" % (dataset, stage)))


def _reduction(vals_by_metric):
    for lab in ("parameters", "MACs / window"):
        d = vals_by_metric.get(lab) or {}
        a, b = d.get("CNN-BiLSTM"), d.get("LightTCN")
        if a and b and a > 0:
            return "%s: %.1f%% fewer in LightTCN" % (lab, 100.0 * (1.0 - b / a))
    return None


# ══ Task 10 - membership inference ═══════════════════════════════════════════
def fig_mia(dataset, stage="mia"):
    """Attack advantage and AUC per attack, by privacy level - the metric MIA reports."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if r.get("mia_attack") is not None]
    name = "fig_mia_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no mia-stage runs logged for %s" % dataset)
    adv = _summary(runs, ["dp_level", "mia_attack"], metric="advantage")
    auc = _summary(runs, ["dp_level", "mia_attack"], metric="auc")
    if not adv:
        return _skip(name, "mia runs for %s carry no advantage column" % dataset)
    attacks = sorted({r["mia_attack"] for r in adv})
    levels = sorted({r["dp_level"] for r in adv})
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_2COL, 2.9))
    for ax, rows, metric, ylab in ((axes[0], adv, "advantage",
                                   "membership advantage"),
                                   (axes[1], auc, "auc", "attack AUC")):
        for j, lev in enumerate(levels):
            for r in rows:
                if r["dp_level"] != lev:
                    continue
                x = attacks.index(r["mia_attack"]) + (j - (len(levels) - 1) / 2.0) * 0.36
                _ci_bars(ax, [x], [r], metric=metric,
                         color=METHOD_COLORS[j % len(METHOD_COLORS)], width=0.32,
                         show_points=False, label="%s%s" % (lev, _eps_tag(r)))
        if metric == "advantage":
            ax.axhline(0.0, color="black", lw=0.9, ls="--")
            ax.annotate("random guess", (axes[0].get_xlim()[1], 0.0), fontsize=6,
                        ha="right", va="bottom")
        else:
            ax.axhline(0.5, color="black", lw=0.9, ls="--")
        ax.set_xticks(list(range(len(attacks))))
        ax.set_xticklabels([_short_attack(a) for a in attacks], rotation=50, ha="right",
                           fontsize=6.5)
        ax.set_ylabel(ylab)
        ax.set_title(metric, fontsize=8)
        ax.legend(fontsize=6, frameon=False, title="target")
        ax.grid(axis="y", **_GRID)
    fig.suptitle("%s: membership inference, threshold chosen on shadow data (%d runs)"
                 % (dataset, len(runs)), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, name, dict(stage=stage, n_runs=len(runs),
                                   log_filter="dataset=%s stage=%s mia_attack logged"
                                   % (dataset, stage)))


def _eps_tag(r):
    e = _val(r["_runs"][0], "target_eps") if r.get("_runs") else None
    e = e if e is not None else (_val(r["_runs"][0], "eps") if r.get("_runs") else None)
    return "" if e is None else " ($\\varepsilon$=%.2f)" % e


def _short_attack(a):
    s = str(a).split("[")[0]
    return {"yeom_loss_threshold": "Yeom loss", "confidence": "Confidence",
            "shokri_shadow": "Shokri shadow", "label_only": "Label-only",
            "lira": "LiRA"}.get(s, s)


# ── the whole set ─────────────────────────────────────────────────────────────
PANEL_METRICS = ("accuracy", "precision", "recall", "f1_macro", "balanced_accuracy",
                 "macro_specificity", "kappa", "roc_auc_ovr")
PANEL_LABELS = {"accuracy": "accuracy", "precision": "precision (w)",
                "recall": "recall (w)", "f1_macro": "macro F1",
                "balanced_accuracy": "balanced acc.",
                "macro_specificity": "macro specificity", "kappa": "Cohen's kappa",
                "roc_auc_ovr": "ROC-AUC (ovr)"}


def _cells_for(dataset, stage, arch):
    """Logged runs of one dataset/stage/architecture, grouped as {cell: [runs]}."""
    runs = runlog.read_logs(dataset=dataset, stage=stage)
    cells = {}
    for r in runs:
        if arch and _arch_of(r).upper() != arch.upper():
            continue
        cells.setdefault(_cell_of(r), []).append(r)
    return cells


def fig_metric_panels(dataset, stage="main", arch="MLP"):
    """Every headline metric, one panel each, for all methods of one architecture.

    The accuracy-only figure cannot show the failure mode a privacy paper has to answer:
    a model that keeps accuracy by collapsing onto the majority class shows it here as
    recall and macro-F1 falling while accuracy does not. Panels are only drawn for metrics
    the logs carry, so a run logged before a column existed leaves that panel absent rather
    than plotting a number nobody measured.
    """
    plt = _pyplot()
    name = "fig_metrics_%s_%s" % (dataset.replace(" ", "_"), arch.replace("-", ""))
    cells = _cells_for(dataset, stage, arch)
    if not cells:
        return _skip(name, "no logged runs for %s stage=%s arch=%s"
                            % (dataset, stage, arch))
    ordered = sorted(cells, key=lambda c: -_mean(
        [_val(r, "accuracy") for r in cells[c]]))
    have = [m for m in PANEL_METRICS
            if any(_val(r, m) is not None for c in ordered for r in cells[c])]
    if not have:
        return _skip(name, "logs carry none of the panel metrics")
    ncol = 4
    nrow = -(-len(have) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIGSIZE_2COL, 2.5 * nrow),
                             squeeze=False)
    for i, metric in enumerate(have):
        ax = axes[i // ncol][i % ncol]
        rows = [r for r in _summary([r for c in ordered for r in cells[c]],
                                    ["cell"], metric=metric)
                if r["mean"] is not None]
        if not rows:
            ax.axis("off")
            continue
        xs = list(range(len(rows)))
        _ci_bars(ax, xs, rows, metric=metric, color=METHOD_COLORS[i % len(METHOD_COLORS)])
        ax.set_xticks(xs)
        ax.set_xticklabels([r["cell"] for r in rows], rotation=60, ha="right", fontsize=5.5)
        ax.set_title(PANEL_LABELS.get(metric, metric), fontsize=8)
        ax.set_ylim(0, 1.02 if metric != "roc_auc_ovr" else 1.0)
        ax.grid(axis="y", **_GRID)
    for j in range(len(have), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("%s / %s: full metric set, %d logged runs (bars = seed mean, "
                 "whiskers = 95%% t-CI, dots = each seed)"
                 % (dataset, arch, sum(len(v) for v in cells.values())), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, name, dict(stage=stage, n_runs=sum(len(v) for v in cells.values()),
                                   metrics=list(have),
                                   log_filter="dataset=%s stage=%s architecture=%s"
                                   % (dataset, stage, arch)))


def fig_per_class(dataset, stage="main", arch="MLP", top=6):
    """Class-wise macro picture: mean F1 per class, one line per method.

    A per-class panel is what shows which activity a private model gave up. Only methods
    whose runs logged `f1_per_class` appear, and the class axis is the integer label index
    the log carries - the class names come from the Work Order's mapping table, not here.
    """
    plt = _pyplot()
    name = "fig_per_class_%s_%s" % (dataset.replace(" ", "_"), arch.replace("-", ""))
    cells = {c: rs for c, rs in _cells_for(dataset, stage, arch).items()
             if any(isinstance(r.get("f1_per_class"), list) for r in rs)}
    if not cells:
        return _skip(name, "no logged run for %s stage=%s arch=%s has f1_per_class"
                            % (dataset, stage, arch))
    ordered = sorted(cells, key=lambda c: -_mean(
        [_val(r, "accuracy") for r in cells[c]]))[:top]
    n_class = max(len(r["f1_per_class"]) for c in ordered for r in cells[c]
                  if isinstance(r.get("f1_per_class"), list))
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE_2COL, 2.3))
    for panel, (ax, key, title) in enumerate(zip(
            axes, ("f1_per_class", "precision_per_class", "recall_per_class"),
            ("per-class F1", "per-class precision", "per-class recall"))):
        for i, c in enumerate(ordered):
            series = {}
            for r in cells[c]:
                lst = r.get(key)
                if not isinstance(lst, list):
                    continue
                for k, v in enumerate(lst[:n_class]):
                    if v is not None:
                        series.setdefault(k, []).append(float(v))
            if not series:
                continue
            xs = sorted(series)
            means = [sum(series[x]) / len(series[x]) for x in xs]
            ax.plot(xs, means, "-", marker="o", ms=3.5, lw=1.1,
                    color=METHOD_COLORS[i % len(METHOD_COLORS)], label=c)
        ax.set_xticks(list(range(n_class)))
        ax.set_xlabel("class index", fontsize=7)
        ax.set_ylabel(title.split()[-1], fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", **_GRID)
        if panel == 0 and ordered:
            ax.legend(fontsize=5.5, frameon=False, ncol=2)
    fig.suptitle("%s / %s: per-class readings for the %d logged methods with the highest "
                 "mean accuracy (seed mean)" % (dataset, arch, len(ordered)), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _finish(fig, name, dict(stage=stage, n_runs=sum(len(cells[c]) for c in ordered),
                                   cells=list(ordered), n_class=n_class,
                                   log_filter="dataset=%s stage=%s architecture=%s "
                                              "f1_per_class logged"
                                              % (dataset, stage, arch)))


def fig_confusion(dataset, stage="main", arch="MLP", top=4):
    """Row-normalised confusion heatmaps, averaged over the seeds that logged a matrix.

    The rejected submission's figures carried no confusion evidence at all. These are drawn
    only from the `confusion` arrays in the run log; a method with no logged matrix is
    absent from the figure.
    """
    plt = _pyplot()
    name = "fig_confusion_%s_%s" % (dataset.replace(" ", "_"), arch.replace("-", ""))
    cells = {c: rs for c, rs in _cells_for(dataset, stage, arch).items()
             if any(isinstance(r.get("confusion"), list) for r in rs)}
    if not cells:
        return _skip(name, "no logged run for %s stage=%s arch=%s has a confusion matrix"
                            % (dataset, stage, arch))
    ordered = sorted(cells, key=lambda c: -_mean(
        [_val(r, "accuracy") for r in cells[c]]))[:top]
    n = max(len(r["confusion"]) for c in ordered for r in cells[c]
            if isinstance(r.get("confusion"), list))
    ncol = min(4, len(ordered))
    nrow = -(-len(ordered) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIGSIZE_2COL, 2.1 * nrow),
                             squeeze=False)
    for i, c in enumerate(ordered):
        ax = axes[i // ncol][i % ncol]
        mats = [r["confusion"] for r in cells[c] if isinstance(r.get("confusion"), list)]
        # arithmetic mean of the logged counts, cell by cell, so the panel is checkable
        # against confusion_table
        tot_counts = [[0.0 for _ in range(n)] for _ in range(n)]
        for m in mats:
            for a in range(n):
                for b in range(n):
                    tot_counts[a][b] += float(m[a][b]) if (
                        a < len(m) and b < len(m[a])) else 0.0
        counts = [[v / max(1, len(mats)) for v in row] for row in tot_counts]
        rate = []
        for a in range(n):
            row_sum = sum(counts[a])
            rate.append([v / row_sum if row_sum > 0 else 0.0 for v in counts[a]])
        im = ax.imshow(rate, cmap="Blues", vmin=0.0, vmax=1.0)
        for a in range(n):
            for b in range(n):
                ax.text(b, a, "%.2f" % rate[a][b], ha="center", va="center",
                        fontsize=5,
                        color="white" if rate[a][b] > 0.6 else "black")
        ax.set_xticks(list(range(n))); ax.set_yticks(list(range(n)))
        ax.set_xlabel("pred", fontsize=6.5); ax.set_ylabel("true", fontsize=6.5)
        ax.set_title("%s (n=%d seeds)" % (c, len(mats)), fontsize=7)
        ax.set_xticklabels([str(k) for k in range(n)], fontsize=5.5)
        ax.set_yticklabels([str(k) for k in range(n)], fontsize=5.5)
    for j in range(len(ordered), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("%s / %s: confusion, row-normalised, averaged over the seeds that logged "
                 "a matrix" % (dataset, arch), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish(fig, name, dict(stage=stage, n_runs=sum(len(cells[c]) for c in ordered),
                                   cells=list(ordered),
                                   log_filter="dataset=%s stage=%s architecture=%s "
                                              "confusion logged"
                                              % (dataset, stage, arch)))


def fig_epsilon_curve(dataset, stages=("privacy", "clip", "main")):
    """Epsilon spent per round, beside the measured release SNR per round.

    Left panel: the accountant's cumulative curve for each DP level that was logged, which
    is the quantity the paper's privacy claims are stated in. Right panel: the same runs'
    measured signal-to-noise of the released update (mean over seeds). Read together they
    are the honest version of "why is accuracy flat" - the curve on the left is what the
    mechanism guarantees, the one on the right is what it leaves usable.
    """
    plt = _pyplot()
    name = "fig_eps_curve_%s" % dataset.replace(" ", "_")
    runs = [r for st in stages for r in runlog.read_logs(dataset=dataset, stage=st)]
    runs = [r for r in runs if isinstance(r.get("eps_per_round"), list) and r["eps_per_round"]]
    if not runs:
        return _skip(name, "no logged run for %s in stages=%s carries eps_per_round"
                            % (dataset, ",".join(stages)))
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_2COL, 2.6))
    groups = {}
    for r in runs:
        groups.setdefault("%s | %s" % (_arch_of(r), _cell_of(r)), []).append(r)
    for i, (key, rs) in enumerate(sorted(groups.items())):
        curves = [r["eps_per_round"] for r in rs if isinstance(r.get("eps_per_round"), list)]
        if not curves:
            continue
        n_max = max(len(c) for c in curves)
        # the curve is identical across seeds by construction (eps depends on z, T, delta
        # only), so the mean is taken to average away nothing but to stay honest if a run
        # stopped early
        xs = list(range(1, n_max + 1))
        eps_mean, snr_mean = [], []
        for x in xs:
            e = [c[x - 1] for c in curves if len(c) >= x]
            eps_mean.append(sum(e) / len(e) if e else None)
        col = METHOD_COLORS[i % len(METHOD_COLORS)]
        pts = [(x, v) for x, v in zip(xs, eps_mean) if v is not None]
        if pts:
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col,
                         lw=1.3, marker="o", ms=2.5, label=key)
        snr = [r.get("per_round_release_snr") for r in rs
               if isinstance(r.get("per_round_release_snr"), list)]
        if snr:
            n_s = max(len(s) for s in snr)
            for x in range(1, n_s + 1):
                vals = [s[x - 1] for s in snr if len(s) >= x and s[x - 1] is not None]
                if vals:
                    snr_mean.append((x, sum(vals) / len(vals)))
            if snr_mean:
                axes[1].plot([p[0] for p in snr_mean], [p[1] for p in snr_mean], "-",
                             color=col, lw=1.3, marker="s", ms=2.5, label=key)
    axes[0].set_xlabel("round", fontsize=7); axes[0].set_ylabel("$\\varepsilon$ spent",
                                                               fontsize=7)
    axes[0].set_title("cumulative privacy cost (accountant)", fontsize=8)
    axes[0].grid(**_GRID)
    axes[1].axhline(1.0, color="black", lw=0.9, ls="--")
    axes[1].annotate("signal = noise", (1.0, 1.0), fontsize=5.5, ha="left", va="bottom")
    axes[1].set_xlabel("round", fontsize=7)
    axes[1].set_ylabel("release SNR (measured)", fontsize=7)
    axes[1].set_title("utility the noise leaves behind", fontsize=8)
    axes[1].grid(**_GRID)
    axes[0].legend(fontsize=5.0, frameon=False, ncol=2)
    fig.suptitle("%s: epsilon and measured release SNR per round, %d DP runs"
                 % (dataset, len(runs)), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _finish(fig, name, dict(stage=",".join(stages), n_runs=len(runs),
                                   cells=len(groups),
                                   log_filter="dataset=%s stages=%s eps_per_round logged"
                                   % (dataset, ",".join(stages))))


def _per_dataset_fns():
    return [("fig_main", fig_main_results), ("fig_all_readings", fig_all_readings),
            ("fig_metrics", fig_metric_panels), ("fig_per_class", fig_per_class),
            ("fig_confusion", fig_confusion),
            ("fig_curves", fig_training_curves), ("fig_forest", fig_paired_forest),
            ("fig_eps_curve", fig_epsilon_curve),
            ("fig_clip", fig_clip_sweep), ("fig_clip_history", fig_clip_history),
            ("fig_eps_split", fig_eps_split), ("fig_loso", fig_loso),
            ("fig_matched", fig_matched), ("fig_noniid", fig_noniid),
            ("fig_features", fig_feature_families), ("fig_cost", fig_compute_cost),
            ("fig_mia", fig_mia)]


def all_figures(datasets=None, inline=None):
    """Regenerate every figure the logs can support. Returns the manifest list.

    A figure whose data has not been logged is reported as skipped with the filter it
    looked for, so an empty panel is never mistaken for a result and nothing here can
    invent a value.
    """
    if not apply_style():
        return [dict(figure="*", skipped="matplotlib unavailable")]
    if inline is not None:
        STATE["inline"] = bool(inline)
    datasets = list(datasets) if datasets else list(ALL_DATASETS_DEFAULT)
    built = []
    for ds in datasets:
        print(" figures for %s" % ds)
        for _, fn in _per_dataset_fns():
            try:
                built.append(fn(ds))
            except Exception as exc:
                built.append(_skip(fn.__name__, "%s(%s) raised %s: %s"
                                  % (fn.__name__, ds, type(exc).__name__, exc)))
        for fn in (fig_byzantine,):
            try:
                got = fn(ds)
                built.extend(got if isinstance(got, list) else [got])
            except Exception as exc:
                built.append(_skip(fn.__name__, "%s(%s) raised %s: %s"
                                  % (fn.__name__, ds, type(exc).__name__, exc)))
        if ds in ("WISDM v2", "PAMAP2"):
            try:
                built.append(fig_channel_treatment(ds))
            except Exception as exc:
                built.append(_skip("fig_channels", "raised %s: %s"
                                  % (type(exc).__name__, exc)))
    # The cross-dataset and privacy sweeps: accuracy AND F1, because line 65 of the sheet
    # asks for both against cumulative epsilon on all four datasets and all three
    # architectures, and one shared-axis view of the four datasets, because a per-dataset
    # panel cannot show whether they are still on the same scale.
    cross = [(fig_privacy_utility, dict(datasets=datasets)),
             (fig_privacy_utility, dict(datasets=datasets, metric="f1")),
             (fig_cross_dataset, dict(datasets=datasets)),
             (fig_cross_dataset, dict(datasets=datasets, metric="f1")),
             (fig_cross_dataset_eps, dict(datasets=datasets)),
             (fig_cross_dataset_eps, dict(datasets=datasets, metric="f1")),
             (fig_leakage, dict(datasets=datasets)),
             (fig_resampling, dict(datasets=datasets))]
    for fn, kwargs in cross:
        metric = kwargs.get("metric", "accuracy")
        label = fn.__name__ if metric == "accuracy" else "%s_%s" % (fn.__name__, metric)
        try:
            built.append(fn(**kwargs))
        except Exception as exc:
            built.append(_skip(label, "raised %s: %s" % (type(exc).__name__, exc)))
    write_manifest(built)
    return built


def write_manifest(built):
    import json
    config.ensure_dirs()
    path = os.path.join(config.FIG_DIR, "figure_manifest.json")
    rows = [{k: v for k, v in b.items() if k != "_runs"} for b in built]
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    ok = [b for b in built if not b.get("skipped")]
    print("figures: %d drawn, %d skipped -> %s" % (len(ok), len(built) - len(ok),
                                                   config.FIG_DIR))
    for b in built:
        if b.get("skipped"):
            print("   skipped %-26s %s" % (b.get("figure"), b["skipped"]))
    with open(os.path.join(config.FIG_DIR, "figure_manifest.csv"), "w", newline="") as fh:
        import csv as _csv
        cols = ["figure", "path_png", "path_pdf", "stage", "n_runs", "cells",
                "log_filter", "skipped"]
        w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for b in rows:
            w.writerow({k: b.get(k, "") for k in cols})
    return path


if __name__ == "__main__":      # pragma: no cover
    print("figure manifest from the logs that exist:")
    for rec in all_figures():
        print("  %s" % rec.get("figure"))
