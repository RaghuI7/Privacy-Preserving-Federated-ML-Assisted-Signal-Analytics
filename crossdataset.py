"""The four datasets in one file, on one scale.

Work Order line 12 forbids the exact situation the previous revision was in: WISDM v2's
sweeps came from an older broken pipeline (~23% accuracy) while Table 5 came from the fixed
script (84.9%), and nothing in the deliverables could tell a reader which was which. Three
artifacts close that. `cross_dataset_runs.csv` is every logged run of every dataset on one
fixed header, so the four per-dataset sessions leave behind four files that stack.
`cross_dataset_comparison.csv` puts the same cell from all four datasets in one row, so a
stale or partial dataset is visible as a gap or a level shift with no arithmetic to do.
`pipeline_identity.csv` compares the settings each dataset actually ran, taken from the
logs rather than from the settings table, and says plainly whether the four are still one
pipeline.

Everything here reads `results/logs/`. No model is run and nothing is invented: a cell with
no logged run is an empty column, and a run logged before a metric existed leaves that
metric blank.

One session cannot hold all four: Kaggle wipes `/kaggle/working` between runs, so each
per-dataset bundle writes the tables for the datasets whose logs it has, and `consolidate`
is the merge step that takes the exported folders (or the four CSVs) back in and writes the
combined file and the cross-dataset figures from them.
"""

import csv
import json
import os

from . import config, multiseed, runlog

# the four datasets the sheet fixes, in the order the paper presents them
DATASETS = ("UCI HAR", "PAMAP2", "WISDM v2", "HARTH")

# the same headline set as exp_main.SUMMARY_METRICS, restated here so this module can be
# imported without pulling the experiment stack
METRICS = ("accuracy", "precision", "recall", "f1", "f1_macro", "balanced_accuracy",
           "macro_specificity", "kappa", "roc_auc_ovr", "nll")

CERT_COLS = ("eps", "alpha_star", "z", "C", "K", "T", "delta", "adjacency", "mechanism")

# The header of `cross_dataset_runs.csv`, in the order a reader wants it: which run this
# is, what it was configured with, every metric, the certificate it paid, then anything
# the stage carries that no other stage does. Fixed, not inferred, so the file written by
# the UCI HAR session and the file written by the HARTH session stack on top of each
# other without a column moving.
RUN_COLUMNS = (
    "dataset", "stage", "architecture", "cell", "label", "method", "seed",
    "partition_id", "reportable", "profile", "config_hash", "logged_at", "wall_time_s",
    "rounds", "T", "local_epochs", "epochs", "lr", "batch", "max_windows", "split_kind",
    "n_clients", "n_windows_train", "n_windows_test", "variant", "channel_mode",
    "feature_set", "attack", "mia_attack", "byz_attack", "byz_aggregator",
    "byz_fraction", "dp_support", "dp_sparsity", "dp_head_only",
) + METRICS + CERT_COLS + (
    "release_snr_mean", "release_snr_min", "release_snr_max", "noise_dominates",
    "accuracy_iter_avg", "f1_iter_avg", "noise_coords", "total_coords",
    "user_level_certificate", "extra_json")
# `T` is both a setting and a certificate column; a header must not repeat a column
RUN_COLUMNS = tuple(dict.fromkeys(RUN_COLUMNS))

# The columns that stay text when a merged CSV is read back: everything else that parses
# as a number is turned back into one, because a comparison of two text "0.850" values is
# a string comparison and a table of them is silently wrong.
TEXT_COLUMNS = ("dataset", "stage", "architecture", "cell", "label", "method", "profile",
                "reportable", "config_hash", "logged_at", "split_kind", "variant",
                "channel_mode", "feature_set", "attack", "mia_attack", "byz_attack",
                "byz_aggregator", "mechanism", "adjacency", "dp_support", "dp_head_only",
                "user_level_certificate", "noise_dominates", "extra_json")

# The settings that must agree across datasets for their rows to be comparable at all.
# `reportable` and `profile` are here because a smoke row mixed into a full table is the
# same class of error as a different round count, and just as invisible in a figure.
# `stage` is deliberately absent: it is the grouping the check runs inside, not a setting,
# so comparing it against itself would flag every table as inconsistent.
# Both `rounds`/`T` and `local_epochs`/`epochs` are listed: the FL loop writes the first of
# each pair and a centralised run the second, and a key a stage never logged is skipped.
PIPELINE_KEYS = ("rounds", "T", "local_epochs", "epochs", "lr", "batch", "z", "C",
                 "delta", "max_windows", "split_kind", "profile", "reportable")

# Properties of the signal, not of the pipeline: four datasets with different subject
# counts cannot share these, and a check that flagged them would flag itself as noise.
# They are still reported, so the table shows what actually differed.
DATASET_KEYS = ("K", "n_clients", "n_windows_train", "n_windows_test")


def _slug(name):
    return str(name).replace(" ", "_").replace("/", "-")


def slug(name):
    """The column suffix this module uses for a dataset name (public: readers of the CSVs
    need to know which column is which dataset without guessing the spelling)."""
    return _slug(name)


def _cell_of(r):
    return str(r.get("cell") or r.get("label") or r.get("method") or "?")


def _arch_of(r):
    return str(r.get("architecture") or r.get("arch") or "?")


def _num(v):
    """A float, or None for anything a CSV column cannot hold."""
    if v is None or isinstance(v, (list, dict, tuple, bool, str)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _csv_num(v):
    """The same, but from CSV text: a merged file carries every number as a string."""
    if v is None or isinstance(v, (list, dict, tuple, bool)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _fmt_val(v):
    """One canonical spelling of a setting value.

    A log holds the round count as the int 50, a merged CSV reads it back as the float
    50.0, and comparing the two texts would report a pipeline difference that does not
    exist. Numbers get the same format from both sources; everything else keeps its text.
    """
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return format(float(v), ".10g")
    f = _csv_num(v)
    return format(f, ".10g") if f is not None else str(v)


def _reference_set(seen):
    """The value-set the most datasets used at this stage: the majority, not the union.

    Flagging against the union marks every dataset guilty when one drifts, which hides
    which one drifted. Ties break on the sorted values so two runs agree.
    """
    counts = {}
    for vals in seen.values():
        counts[frozenset(vals)] = counts.get(frozenset(vals), 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
    return set(ranked[0][0])


def read_all(datasets=None, stages=None, reportable_only=True):
    """Every logged run of every named dataset, as one list. The source of all tables."""
    rows = []
    for ds in (datasets or DATASETS):
        for stage in (stages or [None]):
            try:
                rows.extend(runlog.read_logs(dataset=ds, stage=stage,
                                             reportable_only=reportable_only))
            except Exception:
                # a dataset with no logs at all is a coverage statement, not an error
                continue
    return rows


def _read_log_dir(log_dir):
    """The JSONL logs in one exported folder, in the same shape as `runlog.read_logs`.

    A session's `/kaggle/working` is gone by the next day; what survives is the folder the
    user exported. The file name is still `<dataset>__<stage>.jsonl`, so the dataset and
    stage come from it exactly as they do from the live log directory.
    """
    out = []
    try:
        names = sorted(os.listdir(log_dir))
    except Exception:
        return out
    for fname in names:
        if not fname.endswith(".jsonl"):
            continue
        base = fname[:-len(".jsonl")]
        ds, _, st = base.partition("__")
        with open(os.path.join(log_dir, fname)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                rec.setdefault("dataset", ds)
                rec.setdefault("stage", st)
                out.append(rec)
    return out


def runs_table(rows=None, datasets=None, stages=None, reportable_only=True):
    """One row per logged run, all four datasets sharing ONE fixed column set.

    This is the single file the four sessions can be merged with: every number any figure
    or table is built from, with its dataset, stage, architecture, cell, seed, the full
    metric set, the settings that produced it and the certificate it was bought with.
    The header is fixed rather than the union of whatever the logs happen to carry, so
    the UCI HAR file and the HARTH file have the same columns in the same order and a
    reader can stack the four with one command; a field that is real but not in the
    column list is kept in `extra_json` instead of silently changing the header.
    """
    rows = rows if rows is not None else read_all(datasets, stages, reportable_only)
    known = set(RUN_COLUMNS)
    out = []
    for r in rows:
        row = {k: None for k in RUN_COLUMNS}
        row.update(dataset=r.get("dataset"), stage=r.get("stage"),
                   architecture=_arch_of(r), cell=_cell_of(r), seed=r.get("seed"),
                   partition_id=r.get("partition_id"),
                   reportable=r.get("reportable"), profile=r.get("profile"),
                   config_hash=r.get("config_hash"), logged_at=r.get("logged_at"),
                   wall_time_s=_num(r.get("wall_time_s")))
        for k in METRICS:
            row[k] = _num(r.get(k))
        for k in RUN_COLUMNS:
            if r.get(k) is not None and k not in METRICS:
                row[k] = r[k]
        extras = {k: v for k, v in r.items()
                  if k not in known and not isinstance(v, (list, dict, tuple))}
        if extras:
            row["extra_json"] = json.dumps(extras, sort_keys=True, default=str)
        out.append(row)
    out.sort(key=lambda x: (str(x["dataset"]), str(x["stage"]), str(x["architecture"]),
                            str(x["cell"]), str(x["seed"])))
    return out


def comparison_table(rows=None, datasets=None, stages=None, metrics=None,
                     reportable_only=True):
    """One row per (stage, architecture, cell); each dataset's spread side by side.

    For every metric and dataset the row carries `<metric>_mean__<Dataset>`,
    `<metric>_std__<Dataset>` and `<metric>_ci__<Dataset>` (the interval as one string,
    because a table cell a human reads), plus `<metric>_n__<Dataset>`. `accuracy_spread`
    is max-minus-min of the four means: the number that says whether one dataset is off
    the scale the others are on. A dataset with no logged run in the cell gets
    `missing__<Dataset>` and no numbers - never a zero.
    """
    rows = rows if rows is not None else read_all(datasets, stages, reportable_only)
    datasets = list(datasets or DATASETS)
    metrics = list(metrics or METRICS)
    groups = {}
    for r in rows:
        key = (str(r.get("stage") or "?"), _arch_of(r), _cell_of(r))
        groups.setdefault(key, {}).setdefault(str(r.get("dataset")), []).append(r)
    out = []
    for (stage, arch, cell), by_ds in sorted(groups.items()):
        row = dict(stage=stage, architecture=arch, cell=cell,
                   datasets_present=len(by_ds))
        acc_means = []
        for ds in datasets:
            group = by_ds.get(ds) or []
            slug = _slug(ds)
            for metric in metrics:
                vals = [_num(r.get(metric)) for r in group]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                s = multiseed.summarise(vals, metric_name=metric)
                row["%s_mean__%s" % (metric, slug)] = round(s["mean"], 6)
                row["%s_std__%s" % (metric, slug)] = round(s["std"], 6)
                row["%s_ci__%s" % (metric, slug)] = "[%.4f, %.4f]" % (s["ci95_low"],
                                                                      s["ci95_high"])
                row["%s_n__%s" % (metric, slug)] = s["n"]
                if metric == "accuracy":
                    acc_means.append(s["mean"])
            snr = [_num(r.get("release_snr_mean")) for r in group]
            snr = [v for v in snr if v is not None]
            if snr:
                row["release_snr_mean__%s" % slug] = round(sum(snr) / len(snr), 6)
            eps = [r.get("eps") for r in group if r.get("eps") is not None]
            if eps:
                uniq = sorted({_fmt_val(v) for v in eps})
                row["eps__%s" % slug] = eps[0] if len(uniq) == 1 else "|".join(uniq)
            # the settings that produced these numbers, on the same row: a reader who sees
            # a level shift can check whether the runs were even comparable without
            # opening a second file
            for field in ("rounds", "T", "n_clients"):
                vals = sorted({_fmt_val(r[field]) for r in group
                               if r.get(field) is not None})
                if vals:
                    row["%s__%s" % (field, slug)] = "|".join(vals)
            if not group:
                row["missing__%s" % slug] = "no logged runs"
        row["accuracy_spread"] = (round(max(acc_means) - min(acc_means), 6)
                                  if len(acc_means) > 1 else None)
        row["complete"] = len(acc_means) == len(datasets)
        out.append(row)
    return out


def pipeline_identity(rows=None, datasets=None, reportable_only=True):
    """Do the four datasets still run one pipeline? Answered from the logs, not settings.

    One row per (stage, setting, dataset) holding the distinct values that dataset was
    logged using at that stage. A setting is only comparable when every dataset that ran
    the stage used the SAME SET of values: a single value everywhere agrees, and a grid
    swept identically on all four agrees too. A dataset that used a subset or a different
    value is flagged NO, and that is exactly the defect the sheet's line 12 names - the
    WISDM sweeps from an older broken pipeline next to Table 5 from the fixed script.
    """
    rows = rows if rows is not None else read_all(datasets, None, reportable_only)
    wanted = list(datasets or DATASETS)
    # The pooled multi-dataset arm is a member of the evidence but not one of the four
    # datasets, and it deliberately runs a different window cap - so it is not part of a
    # statement about whether the four are still one pipeline.
    rows = [r for r in rows if str(r.get("dataset")) in set(wanted)]
    out = []
    stages = sorted({str(r.get("stage") or "?") for r in rows})
    for stage in stages:
        sub = [r for r in rows if str(r.get("stage") or "?") == stage]
        # Coverage first, as its own row: the sheet also complains about a dataset missing
        # from a table outright (Table 7 ran without WISDM v2), and a per-setting
        # comparison cannot show that - a missing dataset simply contributes no rows.
        counts = {}
        for r in sub:
            counts[str(r.get("dataset"))] = counts.get(str(r.get("dataset")), 0) + 1
        for ds in wanted:
            n = counts.get(ds, 0)
            out.append(dict(stage=stage, setting="__coverage__", dataset=ds,
                            values="%d logged runs" % n, n_distinct=1 if n else 0,
                            must_be_identical="no - the grid may legitimately differ",
                            agrees_with_other_datasets=(
                                "yes" if n else "NO - no logged run at this stage")))
        for key, required in [(k, True) for k in PIPELINE_KEYS] + \
                             [(k, False) for k in DATASET_KEYS]:
            seen = {}
            for r in sub:
                v = r.get(key)
                if v is None or isinstance(v, (list, dict, tuple)):
                    continue
                seen.setdefault(str(r.get("dataset")), set()).add(_fmt_val(v))
            if not seen:
                continue
            reference = _reference_set(seen)
            for ds in sorted(seen):
                vals = seen[ds]
                same = vals == reference
                out.append(dict(
                    stage=stage, setting=key, dataset=ds,
                    values="|".join(sorted(vals)),
                    n_distinct=len(vals),
                    must_be_identical=("yes" if required else
                                       "no - property of the signal, not the pipeline"),
                    agrees_with_other_datasets=(
                        "yes" if same else
                        ("not required - differs, dataset-intrinsic" if not required
                         else "NO - the other %d datasets used %s"
                              % (len(seen) - 1, "|".join(sorted(reference)))))))
    return out


def identity_violations(identity_rows):
    """The (stage, setting, dataset) triples that break the one-pipeline rule."""
    return sorted({(r["stage"], r["setting"], r["dataset"]) for r in identity_rows
                   if str(r["agrees_with_other_datasets"]).startswith("NO")})


def _write(path, rows, fieldnames=None):
    if not rows:
        return None
    fields = list(fieldnames) if fieldnames else sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def print_summary(rows=None, datasets=None, stages=None, reportable_only=True, limit=14):
    """The comparable numbers, on the console, in the order they will be read.

    Pasted into a session log this is the evidence for line 12: the same cell, four
    datasets, one line each, and the identity verdict underneath. `rows` lets a caller
    that has already merged logs from several sessions print from that merge instead of
    re-reading this session's log directory.
    """
    rows = rows if rows is not None else read_all(datasets, stages, reportable_only)
    # the four columns are always printed, including for a dataset that has not logged
    # anything yet: a blank column is the coverage statement the sheet asks for
    cols = list(datasets or DATASETS)
    if not rows:
        print("  cross-dataset: no logged runs yet for %s - the comparison is drawn from"
              " the logs, so it fills in as the per-dataset sessions finish"
              % ", ".join(cols))
        return
    print("  cross-dataset: %d logged runs, %d of %d datasets with logs"
          % (len(rows), len({str(r.get("dataset")) for r in rows} & set(cols)), len(cols)))
    comp = [c for c in comparison_table(rows=rows, datasets=cols)
            if c.get("stage") == "main"] or comparison_table(rows=rows, datasets=cols)
    hdr = "  %-9s %-12s %-16s" % ("stage", "architecture", "cell")
    line = hdr + "".join(" %-16s" % d for d in cols)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for c in comp[:limit]:
        cells = []
        for d in cols:
            # A metric with no logged value is a blank, never a zero: a 0.000 F1 next to a
            # real accuracy would be a number no run produced.
            parts = []
            for metric in ("accuracy", "f1"):
                v = c.get("%s_mean__%s" % (metric, _slug(d)))
                parts.append("-" if v is None else "%.3f" % v)
            cells.append("/".join(parts))
        print("  %-9s %-12s %-16s%s" % (c["stage"], c["architecture"], c["cell"],
                                        "".join(" %-16s" % v for v in cells)))
    if len(comp) > limit:
        print("  ... %d more cells in cross_dataset_comparison.csv" % (len(comp) - limit))
    ident = pipeline_identity(rows=rows, datasets=cols)
    bad = identity_violations(ident)
    checked = sorted({i["setting"] for i in ident if i["setting"] != "__coverage__"})
    print("  settings compared within each stage: %s" % ", ".join(checked))
    if bad:
        print("  PIPELINE NOT IDENTICAL - %d of the %d checks disagree (see"
              " pipeline_identity.csv; the rows of the datasets that disagree are not"
              " comparable):" % (len(bad), len(ident)))
        for stage, setting, ds in bad[:12]:
            mine = [i["values"] for i in ident
                    if i["stage"] == stage and i["setting"] == setting
                    and i["dataset"] == ds]
            others = sorted({i["values"] for i in ident
                             if i["stage"] == stage and i["setting"] == setting
                             and i["dataset"] != ds})
            print("     %-11s %-14s %-9s used %s; the others used %s"
                  % (stage, setting, ds, mine[0], ", ".join(others)))
        if len(bad) > 12:
            print("     ... %d more in pipeline_identity.csv" % (len(bad) - 12))
    else:
        print("  pipeline identity: every setting checked is identical on all %d datasets"
              " with logs, and every stage has runs from all of them" % len(cols))


def write_outputs(datasets=None, stages=None, results_dir=None, reportable_only=True):
    """Write the three cross-dataset CSVs and return their paths (absent where empty)."""
    config.ensure_dirs()
    results_dir = results_dir or config.RESULTS_DIR
    rows = read_all(datasets, stages, reportable_only)
    paths = {}
    for fname, data, cols in (
            ("cross_dataset_runs.csv", runs_table(rows=rows), RUN_COLUMNS),
            ("cross_dataset_comparison.csv", comparison_table(rows=rows), None),
            ("pipeline_identity.csv", pipeline_identity(rows=rows), None)):
        p = _write(os.path.join(results_dir, fname), data, fieldnames=cols)
        if p:
            paths[fname] = p
            print("  cross-dataset: %-30s %4d rows -> %s"
                  % (fname, len(data), p))
        else:
            print("  cross-dataset: %-30s  skipped - nothing logged yet" % fname)
    return paths


def session_block(datasets=None, stages=None, results_dir=None):
    """What the driver calls at the end of a session: write the files, print the reading."""
    paths = write_outputs(datasets=datasets, stages=stages, results_dir=results_dir)
    print_summary(datasets=datasets, stages=stages)
    return paths


def collect_run_files(paths):
    """Stack several `cross_dataset_runs.csv` on one header, and prove they match.

    The four per-dataset bundles each write the same columns, so merging them is a
    concatenation - but only if the headers really are the same, which is checked rather
    than assumed. A file whose header is not `RUN_COLUMNS` is refused and named: silently
    dropping its columns is how one dataset ends up compared on fewer settings than
    another.

    Returns (rows, notes). `rows` is in the shape the tables here take, so the CSV text
    values are turned back into numbers for the metric columns.
    """
    rows, notes = [], []
    for p in (paths or []):
        try:
            with open(p, newline="") as fh:
                got = list(csv.DictReader(fh))
        except Exception as exc:
            notes.append("%s could not be read (%s: %s) - skipped"
                         % (p, type(exc).__name__, exc))
            continue
        fields = tuple(got[0].keys()) if got else ()
        if fields and fields != tuple(RUN_COLUMNS):
            notes.append("%s has %d columns that differ from the %d-column standard"
                         " header - skipped; re-run that bundle with the current code so"
                         " its file stacks with the others"
                         % (p, len(fields), len(RUN_COLUMNS)))
            continue
        for rec in got:
            row = {}
            for k, v in rec.items():
                if v == "" or v is None:
                    row[k] = None
                elif k in TEXT_COLUMNS:
                    row[k] = v
                else:
                    f = _csv_num(v)
                    row[k] = v if f is None else f
            if row.get("dataset") is None:
                notes.append("%s has a row with no dataset - skipped" % p)
                continue
            rows.append(row)
        notes.append("%s: %d runs from %s" % (os.path.basename(p), len(got),
                                              sorted({str(r["dataset"]) for r in got})))
    return rows, notes


def consolidate(inputs=None, datasets=None, results_dir=None, make_figures=True):
    """Merge what the four per-dataset sessions left behind into one comparable set.

    `/kaggle/working` is wiped between sessions, so no single bundle ever holds all four
    datasets' logs. This is the merge step: point `inputs` at the folders or files the
    earlier sessions exported (attached as Kaggle outputs/datasets, or downloaded), and it
    writes the three cross-dataset CSVs and the cross-dataset figures from the combined
    runs - same code path, same columns, same figures as a session that had everything.

    Accepts a log folder, a `results` folder, a `cross_dataset_runs.csv` or a mixture; a
    directory is searched for both the JSONL logs and the CSV.
    """
    inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
    log_dirs, csvs, notes = [], [], []
    for item in inputs:
        if os.path.isdir(item):
            # a session exports results/logs next to the CSVs; accept the notebook output
            # root, a results folder or the log folder itself
            for d in (item, os.path.join(item, "results", "logs"),
                      os.path.join(item, "logs")):
                if os.path.isdir(d) and any(f.endswith(".jsonl") for f in os.listdir(d)):
                    log_dirs.append(d)
            for root, _, files in os.walk(item):
                for f in files:
                    if f == "cross_dataset_runs.csv":
                        csvs.append(os.path.join(root, f))
        elif item.endswith(".csv"):
            csvs.append(item)
        elif item.endswith(".jsonl"):
            log_dirs.append(os.path.dirname(item))
        else:
            notes.append("%s is not a folder, a .csv or a .jsonl - skipped" % item)
    rows, seen = [], set()

    def _key(r):
        return (str(r.get("dataset")), str(r.get("stage")), _arch_of(r),
                _cell_of(r), str(r.get("seed")), str(r.get("config_hash")))

    for d in sorted(set(log_dirs)):
        for r in _read_log_dir(d):
            if _key(r) in seen:
                continue
            seen.add(_key(r))
            rows.append(r)
    from_logs = len(rows)
    extra, extra_notes = collect_run_files(csvs)
    notes.extend(extra_notes)
    for rec in extra:
        if _key(rec) in seen:
            continue
        seen.add(_key(rec))
        rows.append(rec)
    if not rows:
        print("  consolidate: nothing found in %s - pass the folders the four sessions"
              " exported (their results/logs, or their cross_dataset_runs.csv)" % inputs)
        return dict(rows=0, paths={}, notes=notes)
    present = sorted({str(r.get("dataset")) for r in rows})
    print("  consolidate: %d runs (%d from logs, %d from the runs CSVs) covering %s"
          % (len(rows), from_logs, len(rows) - from_logs, ", ".join(present)))
    results_dir = results_dir or config.RESULTS_DIR
    config.ensure_dirs()
    paths = {}
    for fname, data, cols in (
            ("cross_dataset_runs.csv", runs_table(rows=rows), RUN_COLUMNS),
            ("cross_dataset_comparison.csv", comparison_table(rows=rows), None),
            ("pipeline_identity.csv", pipeline_identity(rows=rows, datasets=datasets),
             None)):
        p = _write(os.path.join(results_dir, fname), data, fieldnames=cols)
        if p:
            paths[fname] = p
            print("  consolidate: %-30s %4d rows -> %s" % (fname, len(data), p))
    print_summary(rows=rows, datasets=datasets)
    if make_figures:
        _figures(datasets=present)
    return dict(rows=len(rows), paths=paths, notes=notes)


def _figures(datasets):
    """Draw only the cross-dataset figures, from whatever is now in the logs."""
    try:
        from . import plotting
    except Exception as exc:
        print("  consolidate: figures unavailable (%s: %s) - the CSVs are still written"
              % (type(exc).__name__, exc))
        return
    for fn, kwargs in ((plotting.fig_privacy_utility, dict(datasets=datasets)),
                       (plotting.fig_privacy_utility, dict(datasets=datasets,
                                                           metric="f1")),
                       (plotting.fig_cross_dataset, dict(datasets=datasets)),
                       (plotting.fig_cross_dataset, dict(datasets=datasets,
                                                          metric="f1")),
                       (plotting.fig_cross_dataset_eps, dict(datasets=datasets)),
                       (plotting.fig_cross_dataset_eps, dict(datasets=datasets,
                                                              metric="f1"))):
        try:
            fn(**kwargs)
        except Exception as exc:
            print("  consolidate: %s skipped (%s: %s)"
                  % (fn.__name__, type(exc).__name__, exc))
