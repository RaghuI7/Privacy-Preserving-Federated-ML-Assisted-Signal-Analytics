"""Multi-seed aggregation and paired significance testing (Work Order Task 4).

The submission reported one seed (42) for everything. With 5 paired samples the
Wilcoxon signed-rank test cannot go below p = 0.0625, so the family of comparisons
here is evaluated on >= 6 seeds and Holm-corrected across the declared family.
"""

import numpy as np
from scipy import stats

MIN_SEEDS_FOR_WILCOXON = 6


def _ci_t(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    if n < 2:
        return (float(vals[0]), float(vals[0])) if n else (float("nan"), float("nan"))
    m = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(n))
    crit = float(stats.t.ppf(0.975, n - 1))
    return m - crit * se, m + crit * se


def summarise(vals, metric_name="accuracy"):
    vals = np.asarray(vals, float)
    lo, hi = _ci_t(vals)
    return dict(metric=metric_name, n=int(len(vals)), mean=float(np.mean(vals)),
                std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                ci95_low=float(lo), ci95_high=float(hi),
                min=float(np.min(vals)), max=float(np.max(vals)))


def cohen_dz(diffs):
    d = np.asarray(diffs, float)
    if len(d) < 2:
        return float("nan")
    s = float(d.std(ddof=1))
    return float(d.mean() / s) if s > 1e-12 else float("nan")


def compare(a, b, name_a="A", name_b="B", family=None, min_seeds=MIN_SEEDS_FOR_WILCOXON):
    """Paired comparison over the SAME seeds.

    Returns mean difference, 95% CI, Cohen's dz, paired t-test p, Wilcoxon p and the
    Holm-adjusted p when `family` (a list of the raw p-values in the comparison family)
    is supplied.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired comparison needs equal seed counts (%d vs %d)"
                         % (len(a), len(b)))
    n = len(a)
    d = a - b
    lo, hi = _ci_t(d)
    t_stat, p_t = stats.ttest_rel(a, b) if n >= 2 else (float("nan"), float("nan"))
    if n >= 2 and np.any(np.abs(d) > 1e-15):
        try:
            w_stat, p_w = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, p_w = float("nan"), float("nan")
    else:
        w_stat, p_w = float("nan"), float("nan")
    out = dict(comparison="%s vs %s" % (name_a, name_b), n_seeds=int(n),
               mean_a=float(a.mean()), mean_b=float(b.mean()),
               mean_diff=float(d.mean()), diff_ci95_low=float(lo), diff_ci95_high=float(hi),
               cohen_dz=cohen_dz(d), t_stat=float(t_stat), p_ttest=float(p_t),
               wilcoxon_stat=float(w_stat) if w_stat == w_stat else float("nan"),
               p_wilcoxon=float(p_w), insufficient_seeds=bool(n < min_seeds),
               note=("n=%d < %d: Wilcoxon cannot reach p<0.05" % (n, min_seeds)
                     if n < min_seeds else ""))
    if family is not None:
        out["holm_p"] = float(holm_adjust(list(family), p_w))
    return out


def holm_adjust(pvals, p):
    """Holm step-down adjusted p for the entry whose raw value is `p`."""
    order = np.argsort(list(pvals))
    m = len(pvals)
    adj = [float("nan")] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * float(pvals[idx]))
        running = max(running, val)          # monotonicity enforcement
        adj[idx] = running
    for i, v in enumerate(pvals):
        if abs(float(v) - float(p)) < 1e-15:
            return adj[i]
    return float("nan")


def holm_grouped(rows, family_field, key="p_wilcoxon", out="holm_p_grouped"):
    """Holm within groups defined by `family_field`, pooled across everything else.

    `holm_table` corrects inside one dataset x architecture, which is the family the Work
    Order describes; but the same comparison tag is also run on several datasets, and a
    response letter that says "Holm corrected" invites the stricter reading. This gives
    both: group by the comparison name across every dataset and architecture, correct
    inside that, and write `out` plus the family size so the reader sees which of the two
    corrections a star came from. The per-family `holm_p` column is left untouched.
    """
    def clean(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 1.0
        return f if f == f else 1.0
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(str(r.get(family_field, "")), []).append(i)
    for idxs in groups.values():
        pvals = [clean(rows[i].get(key, 1.0)) for i in idxs]
        order = np.argsort(pvals)
        m = len(pvals)
        running = 0.0
        for rank, pos in enumerate(order):
            val = min(1.0, (m - rank) * pvals[pos])
            running = max(running, val)
            rows[idxs[pos]][out] = running
            rows[idxs[pos]]["holm_group_size"] = m
    for r in rows:
        r.setdefault(out, None)
        r.setdefault("holm_group_size", 0)
    return rows


def holm_table(rows, key="p_wilcoxon"):
    """rows: list of compare() dicts. Adds holm_p to each, in place, and returns them."""
    def clean(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 1.0
        return f if f == f else 1.0          # an undefined test is not significant
    pvals = [clean(r.get(key, 1.0)) for r in rows]
    order = np.argsort(pvals)
    m = len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        rows[idx]["holm_p"] = running
        rows[idx]["p_raw"] = pvals[idx]
    return rows


def group_by_seeds(runs, key_fields, metric="accuracy"):
    """{config tuple: {seed: value}} - the shape every paired test wants."""
    groups = {}
    for r in runs:
        k = tuple(str(r.get(f)) for f in key_fields)
        groups.setdefault(k, {})[int(r["seed"])] = float(r[metric])
    return groups


def paired_from_logs(runs, field, value_a, value_b, pair_keys=("dataset", "partition_id"),
                     metric="accuracy", family=None, name_a=None, name_b=None):
    """Paired comparison of two values of `field`, matched on seed within each
    (pair_keys) cell. `field` is normally 'method' or 'architecture'."""
    A, B = {}, {}
    for r in runs:
        cell = tuple(str(r.get(f)) for f in pair_keys)
        seed = int(r.get("seed", -1))
        if r.get(field) == value_a:
            A[(cell, seed)] = float(r[metric])
        if r.get(field) == value_b:
            B[(cell, seed)] = float(r[metric])
    common = sorted(set(A) & set(B))
    if len(common) < 2:
        return dict(n_paired=len(common), mean_diff=None, p_wilcoxon=1.0, p_ttest=1.0,
                    note="fewer than 2 paired cells available")
    a = [A[c] for c in common]
    b = [B[c] for c in common]
    out = compare(a, b, name_a or str(value_a), name_b or str(value_b), family=family)
    out["n_paired"] = len(common)
    return out
