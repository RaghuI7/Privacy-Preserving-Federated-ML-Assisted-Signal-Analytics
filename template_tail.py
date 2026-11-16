
'''

# ══════════════════════════ 3. INSTALLER ══════════════════════════════════════════

_FILE_MARK = "#@@FILE "


def _ppml_parse(blob):
    """Split the embedded blob back into {package-relative path: source}."""
    out, cur, buf = {}, None, []
    for line in blob.splitlines():
        if line.startswith(_FILE_MARK):
            if cur:
                out[cur] = "\n".join(buf).rstrip("\n") + "\n"
            cur, buf = line[len(_FILE_MARK):].strip(), []
        elif cur is not None:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).rstrip("\n") + "\n"
    return out


def _ppml_install(base_dir=INSTALL_DIR, verbose=True):
    """Write the embedded package to <base_dir>/ppml/ and return the package module."""
    files = _ppml_parse(_PPML_BLOB)
    root = os.path.join(base_dir, "ppml")
    written = 0
    for rel, src in sorted(files.items()):
        path = os.path.join(base_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        old = None
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                old = fh.read()
        if old != src:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(src)
            written += 1
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    for m in [k for k in list(sys.modules) if k == "ppml" or k.startswith("ppml.")]:
        del sys.modules[m]
    __import__("importlib").invalidate_caches()
    import ppml                                            # noqa: E402
    if verbose:
        print("  package: %d files, %d written (%s)" % (len(files), written, root))
    return ppml


# ══════════════════════════ 4. DRIVER ═════════════════════════════════════════════

_WRITERS = ("write_csv", "write_csvs", "write_crosschecks",
            "write_label_map_table", "write_client_note")

# A function whose only required argument is `dataset` can only read the logs, so it is
# safe to call for a CSV dump. channel_ablation is the one exception: it loads its own
# data and trains, so it is never called implicitly - run it yourself if you need it.
_NOT_TABLES = {"channel_ablation"}


def _call_by_name(fn, h, dataset_name):
    """Call a stage's CSV/report writer, supplying its required args by parameter name."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    kwargs = {}
    for pname, p in params.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not inspect.Parameter.empty:
            continue
        if pname == "dataset":
            kwargs["dataset"] = dataset_name
        elif pname == "h":
            kwargs["h"] = h
        else:
            print("    %s: cannot supply required arg %r - skipped" % (fn.__name__, pname))
            return False
    fn(**kwargs)
    return True


def _write_rows(path, rows):
    import csv
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (("%.6f" % v if isinstance(v, float) else v))
                        for k, v in r.items()})


def _dump_tables(mod, dataset_name, results_dir):
    """Run every table function the stage exposes on `dataset` and save it as CSV.

    The stage modules keep their tables as functions returning list[dict] and only some
    of them also carry a write_csv, so this is what makes a single Kaggle session
    produce the paper's tables instead of raw logs only.
    """
    import types
    for fname in sorted(dir(mod)):
        if fname.startswith("_") or fname in _NOT_TABLES or fname in _WRITERS:
            continue
        fn = getattr(mod, fname)
        if not isinstance(fn, types.FunctionType) or fn.__module__ != mod.__name__:
            continue
        try:
            params = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            continue
        required = [p for p in params if p.default is inspect.Parameter.empty]
        if [p.name for p in required] != ["dataset"]:
            continue
        try:
            rows = fn(dataset_name)
        except Exception as exc:
            print("    %s(%s) failed: %s" % (fname, dataset_name, exc))
            continue
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        out = os.path.join(results_dir, "tables",
                          "%s__%s__%s.csv" % (dataset_name.replace(" ", "_"),
                                              mod.__name__.split(".")[-1], fname))
        try:
            _write_rows(out, rows)
            print("    table %-22s %4d rows -> %s" % (fname, len(rows),
                                                      os.path.basename(out)))
        except Exception as exc:
            print("    %s: could not write %s (%s)" % (fname, out, exc))


def _run_stage(ppml, h, stage, seeds, arches, t_session):
    """One stage for one dataset. Returns False when the session budget stopped it."""
    from ppml import config
    from ppml.experiments import STAGES
    mod, what = STAGES[stage]
    print("\n--- stage %-11s %s" % (stage, what))
    kw = {"verbose": VERBOSE}
    if seeds is not None:
        kw["seeds"] = seeds
    if "arches" in inspect.signature(mod.run).parameters:
        kw["arches"] = arches
    elif arches is not None:
        print("    (%s has a fixed architecture list; ARCHES ignored)" % stage)
    t0 = time.time()
    try:
        mod.run(h, **kw)
    except config.BudgetExceeded as exc:
        print("  STOPPED: %s" % exc)
        print("  completed runs are already in results/logs/ - re-run this file to resume")
        return False
    except Exception as exc:
        print("  stage %s raised %s: %s" % (stage, type(exc).__name__, exc))
        traceback.print_exc()
        print("  continuing with the next stage; fix and re-run this stage to fill the gaps")
    for wname in _WRITERS:
        fn = getattr(mod, wname, None)
        if fn is None or not callable(fn):
            continue
        try:
            if _call_by_name(fn, h, h.bundle.name):
                print("    wrote %s" % wname)
        except Exception as exc:
            print("    %s failed (%s) - usually a stage that is only part-complete"
                  % (wname, exc))
    try:
        _dump_tables(mod, h.bundle.name, config.RESULTS_DIR)
    except Exception as exc:
        print("    table dump failed (%s) - the raw logs are still complete" % exc)
    print("--- stage %-11s done in %.1f min" % (stage, (time.time() - t0) / 60.0))
    return True


def _preflight(ppml):
    from ppml import config, fl, runlog
    config.ensure_dirs()
    from ppml import dp_accountant
    print("preflight")
    print("  profile      : %s (reportable=%s)" % (config.PROFILE, config.is_reportable()))
    # Scope is part of the evidence. Printing the data caps and the local-epoch budget
    # here means the first lines of any pasted log already disclose what was run, which
    # is what makes a reduced-scope session auditable instead of mysterious.
    print("  notebook     : %s  datasets=%s" % (NOTEBOOK_ID, RUN_DATASETS))
    print("  scope        : max_windows=%s max_subjects=%s E_mlp=%s E_cnn=%s "
          "T=%s/%s seeds=%s/%s"
          % (MAX_WINDOWS_BY_DATASET, MAX_SUBJECTS, config.LOCAL_EPOCHS_MLP,
             config.LOCAL_EPOCHS_CNN, config.cfg()["rounds"], config.cfg()["rounds_dp"],
             config.cfg()["seeds"], config.cfg()["seeds_supp"]))
    print("  recipe       : lr %g->%g (%s), class weights=%s, window scaling=%s, "
          "augment=%s, model selection=%s"
          % (config.LR, config.LR_MIN, config.LR_SCHEDULE, config.CLASS_WEIGHTS,
             config.WINDOW_SCALE, config.AUGMENT,
             "best validation round" if config.SELECT_BEST_VAL else "last round"))
    for ds in RUN_DATASETS:
        print("  stages %-9s: %s" % (ds, ", ".join(s for s in RUN_STAGES
                                                   if s in STAGES_BY_DATASET.get(ds, RUN_STAGES))))
    # The DP operating point, stated before any arm runs: the z, the certificate it buys and
    # the sweep's third arm. A session that changed one of these without saying so would
    # produce tables whose epsilon column no longer matches the arm it is attached to.
    arm_eps = dp_accountant.account_central_gaussian(
        config.DP_ARM_Z, config.cfg()["rounds_dp"], config.DELTA)
    print("  dp arms      : z=%s -> eps=%.4f (T=%d, delta=%g)  head-sweep=%s"
          % (config.DP_ARM_Z, arm_eps["eps"], config.cfg()["rounds_dp"],
             config.DELTA, config.PRIVACY_HEAD_SWEEP))
    if config.PRIVACY_HEAD_SWEEP:
        print("                   ^ head release added at every z for %s: the accuracy "
              "question is measured here, at the same eps as the plain arm it is beside"
              % ", ".join(config.PRIVACY_HEAD_ARCHES))
    print("  printing     : per-round metric lines %s (everything printed is also logged)"
          % ("ON" if config.PRINT_ROUNDS else "OFF"))
    print("  work dir     : %s" % config.WORK)
    print("  kaggle       : %s" % config.KAGGLE)
    print("  data mounts  : %s" % (sorted(os.listdir("/kaggle/input"))
                                   if os.path.isdir("/kaggle/input") else "none"))
    print("  versions     : %s" % runlog.versions())
    print("  hardware     : %s" % runlog.hardware())
    ok = dp_accountant.selftest()
    print("  dp accountant selftest: %s" % ("PASSED" if ok else "FAILED"))
    if not ok:
        raise RuntimeError("DP accountant does not reproduce its own closed form; "
                           "no privacy number from this session is trustworthy")
    # The sparsified DP arm is a claim about epsilon, not just about accuracy: it may only
    # be reported if selecting coordinates after the draw demonstrably leaves the release
    # inside the certificate the plain arm pays for.
    ok2 = fl.projection_selftest()
    print("  dp mechanism selftest: %s" % ("PASSED" if ok2 else "FAILED"))
    if not ok2:
        raise RuntimeError("the sparse release is not epsilon-neutral in this build; the "
                           "FL+DP-Sparse arm would be unreportable, so nothing runs")
    ok3 = fl.clipper_selftest()
    print("  adaptive clipper selftest: %s" % ("PASSED" if ok3 else "FAILED"))
    if not ok3:
        raise RuntimeError("the clip estimator cannot count its own releases; the "
                           "adaptive arm's epsilon would be a guess, so nothing runs")
    # per-example gradients on THIS device: vmap must equal the exact per-example loop
    try:
        from ppml import models as _m
        from ppml.experiments import exp_dpsgd as _dps
        _dev = _m.device_for(True)
        for _a in ("MLP-Small", "LightTCN", "CNN-BiLSTM"):
            if _a == "MLP-Small":
                continue
            _e = _dps.selftest_psg(_a, device=_dev)
            print("  per-example gradient selftest %-10s on %s: max diff %.1e %s"
                  % (_a, _dev, _e, "PASSED" if _e < 1e-3 else "FAILED"))
    except Exception as exc:
        print("  per-example gradient selftest could not run (%s: %s) - the DP-SGD stages "
              "fall back to the exact per-example loop" % (type(exc).__name__, exc))
    # independent eps cross-check (Google dp-accounting), printed before any training
    for _z in config.Z_SWEEP:
        _r = dp_accountant.cross_check(_z, config.cfg()["rounds_dp"], config.DELTA)
        print("  eps cross-check z=%-5s ours %.4f  dp-accounting %s  %s"
              % (_z, _r["ours"], "n/a" if _r["theirs"] is None else "%.4f" % _r["theirs"],
                 "ours >= theirs (conservative)" if _r["ok"] else _r.get("error", "CHECK")))
    # the adaptive arm's cost, stated before it runs: the estimator, not the model, is
    # what the certificate pays for at this participation level.
    T_dp = config.cfg()["rounds_dp"]
    W = min(int(config.ADAPT_FREEZE_ROUNDS), int(T_dp))
    ad = dp_accountant.account_adaptive(config.DP_ARM_Z, T_dp, config.DELTA,
                                        sigma_b=config.ADAPT_SIGMA_B, T_clip=W)
    ad_all = dp_accountant.account_adaptive(config.DP_ARM_Z, T_dp, config.DELTA,
                                            sigma_b=config.ADAPT_SIGMA_B)
    print("  adaptive eps   : z=%.1f T=%d -> eps=%.3f with the estimator releasing %d of "
          "%d rounds (eps=%.3f unfrozen, eps=%.3f fixed-C at the same z)"
          % (config.DP_ARM_Z, T_dp, ad["eps_total"], W, T_dp, ad_all["eps_total"],
             dp_accountant.account_central_gaussian(config.DP_ARM_Z, T_dp,
                                                    config.DELTA)["eps"]))
    from ppml.experiments import exp_dpsgd, exp_dplearn
    plan = exp_dpsgd.dp_sgd_plan()
    print("  DP-SGD plan  : example-level (one window), q=%.3f, %d steps per client, "
          "solved BEFORE training: %s"
          % (config.DP_SGD_SAMPLE_RATE, exp_dpsgd.planned_steps(),
             ", ".join("eps=%.1f -> z=%.3f" % pe for pe in plan)))
    cells, target = exp_dplearn.cells()
    print("  dplearn      : user-level target eps=%.4f; %s"
          % (target, "; ".join("%s (z=%s)" % (c[0], "-" if c[1] is None else
                                              "%.3f" % c[1]["z"]) for c in cells)))
    return ppml


def _feasibility(h):
    """User-level DP feasibility, printed BEFORE any DP accuracy exists.

    Predicted release SNR per round at the operating point, K/(z*sqrt(d)), for the three
    released supports. Below ~1 the release is mostly noise; the arm is still run and
    reported, and this line is what explains its accuracy.
    """
    from ppml import config
    import numpy as np
    s0 = h.split(config.SEEDS[0])
    K = len(h.clients(s0, config.SEEDS[0])["client_indices"])
    z = config.DP_ARM_Z
    T = config.cfg()["rounds_dp"]
    rows = []
    for name, d in (("MLP whole update", 71750), ("MLP-Small whole update", 7942),
                    ("head-only release", 390)):
        snr = K / (z * np.sqrt(d))
        rows.append(dict(dataset=h.bundle.name, K=K, support=name, d=d, z=z, T=T,
                         snr_per_round=snr, snr_over_T=snr * np.sqrt(T)))
        print("  feasibility %-9s K=%2d %-24s d=%6d  SNR/round=%.3f  x sqrt(T)=%.2f"
              % (h.bundle.name, K, name, d, snr, snr * np.sqrt(T)))
    return rows


def _import_previous_logs():
    """Resume across Kaggle sessions: copy results/logs/*.jsonl from any attached input
    (a previous session's saved output) into this session's log folder, de-duplicated by
    run id. Runs found there are skipped, not recomputed."""
    from ppml import config
    config.ensure_dirs()
    if not os.path.isdir("/kaggle/input"):
        return 0
    found = glob.glob("/kaggle/input/**/results/logs/*.jsonl", recursive=True)
    n_new = 0
    for src in found:
        dst = os.path.join(config.LOG_DIR, os.path.basename(src))
        seen = set()
        if os.path.exists(dst):
            with open(dst) as fh:
                for line in fh:
                    try:
                        seen.add(json.loads(line).get("config_hash"))
                    except ValueError:
                        pass
        with open(src) as fh, open(dst, "a") as out:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    h_ = json.loads(line).get("config_hash")
                except ValueError:
                    continue
                if h_ in seen:
                    continue
                seen.add(h_)
                out.write(line + "\n")
                n_new += 1
    if found:
        print("  resume       : %d earlier runs imported from %d attached log files"
              % (n_new, len(found)))
    return n_new


def _run_pooled(ppml, run_datasets, stages):
    """Task 7's cross-sensor arm, run once per session instead of once per dataset.

    It pools a dataset that has a gyroscope with one that has not, which is the only
    setting where zero-padding and an explicit mask can differ at all - inside a single
    gyroscope-less dataset both are constants. That needs two datasets loaded, so it does
    not belong in the per-dataset loop. Returns the pseudo-dataset name when it logged
    runs, so the report counts it beside the real ones.
    """
    if "resampling" not in stages:
        return []
    from ppml import config
    from ppml.experiments import exp_resampling as er
    if not er.pooled_available(run_datasets):
        print("\n--- stage %-11s skipped: this session loads %s and the arm needs %s"
              % ("pooled", ", ".join(run_datasets), " and ".join(er.POOLED_SETS)))
        return []
    if config.budget_left_s() <= 0:
        print("\n--- stage %-11s skipped: session budget exhausted" % "pooled")
        return []
    print("\n--- stage %-11s Task 7 pooled channels (gyro + no-gyro, one model)" % "pooled")
    incomplete = False
    try:
        er.pooled_run(available=run_datasets, verbose=VERBOSE)
    except config.BudgetExceeded as exc:
        incomplete = True
        print("  pooled arm stopped by the session budget (%s) - what it finished is "
              "logged below; re-run this file to complete it" % exc)
    except Exception as exc:
        print("  pooled arm raised %s: %s - the per-dataset channel table is unaffected"
              % (type(exc).__name__, exc))
        traceback.print_exc()
    # The CSVs are written from the log here rather than left to pooled_run, which only
    # writes when it logged something new: a resumed session dedupes every run and would
    # otherwise finish with the table missing.
    logged = er.table(er.POOLED_NAME)
    if not logged:
        return []
    try:
        er.write_pooled_csvs()
    except Exception as exc:
        print("  pooled CSVs not written: %s: %s" % (type(exc).__name__, exc))
    if incomplete:
        print("  NOTE pooled table is partial (budget) - %d rows logged so far"
              % len(logged))
    return [er.POOLED_NAME]


def _report(ppml, names):
    from ppml import config, crossdataset, runlog
    print("\n" + "=" * 78)
    print("OUTPUTS under %s" % config.RESULTS_DIR)
    for ds in names:
        rows = runlog.read_logs(dataset=ds, reportable_only=False)
        by_stage = {}
        for r in rows:
            by_stage[r.get("stage", "?")] = by_stage.get(r.get("stage", "?"), 0) + 1
        n_rep = sum(1 for r in rows if r.get("reportable"))
        print("  %-9s %5d logged runs (%d reportable)  %s"
              % (ds, len(rows), n_rep,
                 ", ".join("%s:%d" % kv for kv in sorted(by_stage.items()))))
    print("  logs   : %s/*.jsonl  (one line per run, the only source for figures)"
          % config.LOG_DIR)
    print("  csvs   : %s/*.csv    (per-stage summary tables)" % config.RESULTS_DIR)
    print("  one file: cross_dataset_runs.csv (fixed %d-column header, one row per run),"
          " cross_dataset_comparison.csv (every metric, four datasets side by side),"
          " pipeline_identity.csv (line 12 checked from the logs)"
          % len(__import__("ppml.crossdataset", fromlist=["x"]).RUN_COLUMNS))
    print("  tables : %s         (one CSV per stage table function)"
          % os.path.join(config.RESULTS_DIR, "tables"))
    print("  figures: %s         (PDF + 300 dpi PNG, drawn from the logs)"
          % config.FIG_DIR)
    print("  resume : keep results/ - already-logged runs are skipped on the next run")
    print("=" * 78)


def _cross_dataset(ppml):
    """Print the four datasets side by side from whatever this session has logged.

    One session is one dataset, so the other three columns read blank - that blank is the
    coverage statement the Work Order asks for, not a zero. The files are written by
    `ppml.reporting.run_report` just below; this is the console evidence for the session log.
    """
    from ppml import crossdataset
    print("\n--- the four datasets in one file (Work Order line 12)")
    try:
        crossdataset.print_summary()
    except Exception as exc:
        print("  cross-dataset block failed (%s: %s) - the logs are unaffected"
              % (type(exc).__name__, exc))
        return
    print("  Kaggle wipes /kaggle/working between sessions, so stack the four files in")
    print("  one merge afterwards:")
    print("      from ppml import crossdataset")
    print("      crossdataset.consolidate([<results folder of each per-dataset session>])")


def _finalise(ppml, names):
    """Figures from the logs + the section 13 run report and contradiction flags."""
    from ppml import config, reporting
    print("\n--- run report and figures")
    try:
        payload = reporting.run_report(datasets=names, write_figures=MAKE_FIGURES)
    except Exception as exc:
        print("  report failed (%s: %s) - the logs are still complete; run "
              "ppml.reporting.run_report() in a later session"
              % (type(exc).__name__, exc))
        traceback.print_exc()
        return None
    print("  report   : %s" % os.path.join(config.RESULTS_DIR, "run_report.md"))
    print("  claims   : %s" % os.path.join(config.RESULTS_DIR, "contradictions.csv"))
    print("  figures  : %s (+ figure_manifest.csv)" % config.FIG_DIR)
    flags = [c for c in payload["contradictions"]
             if c["verdict"] in ("CONTRADICTED", "NOT SUPPORTED", "UNEXPECTED")]
    if flags:
        print("\n  EVERY CLAIM THE MEASUREMENTS DO NOT SUPPORT (send this to the authors):")
        for c in flags:
            print("   - [%s] %s\n     %s" % (c["verdict"], c["paper_claim"],
                                             c["evidence"]))
    return payload


class _Tee(object):
    """Write everything printed to the console AND to run_log.txt, so the full log is one
    of the files in the results zip (Kaggle's log viewer is not a file you can attach)."""

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        self.stream.write(text)
        try:
            self.fh.write(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        self.stream.flush()
        try:
            self.fh.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _zip_outputs():
    """One file to download: results/, figures/ and run_log.txt in PPML_results_<NB>.zip."""
    import zipfile
    out = os.path.join(INSTALL_DIR, "PPML_results_%s.zip" % NOTEBOOK_ID)
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for sub in ("results", "figures"):
                base = os.path.join(INSTALL_DIR, sub)
                for dp, dn, fn in os.walk(base):
                    for f in fn:
                        full = os.path.join(dp, f)
                        z.write(full, os.path.relpath(full, INSTALL_DIR))
            log = os.path.join(INSTALL_DIR, "run_log.txt")
            if os.path.exists(log):
                z.write(log, "run_log.txt")
        print("\nALL OUTPUTS IN ONE FILE: %s (%.1f MB) - download it from the Output tab"
              % (out, os.path.getsize(out) / 1e6))
    except Exception as exc:
        print("zip failed (%s) - results/ and figures/ are still in the Output tab" % exc)


def main():
    os.makedirs(INSTALL_DIR, exist_ok=True)
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout, os.path.join(INSTALL_DIR, "run_log.txt"))
    if not isinstance(sys.stderr, _Tee):
        sys.stderr = _Tee(sys.stderr, os.path.join(INSTALL_DIR, "run_log.txt"))
    try:
        return _main()
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        _zip_outputs()


def _main():
    os.environ["PPML_PROFILE"] = PROFILE
    # Optional independent epsilon cross-check (Google dp-accounting). Needs Internet; a
    # failure here changes nothing else - the in-package accountant is used either way.
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "dp-accounting"],
                       timeout=300, check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except Exception:
        pass
    t_start = time.time()
    print(__doc__ if VERBOSE else "")
    ppml = _ppml_install()
    from ppml import config, datasets, models
    from ppml.harness import Harness
    _preflight(ppml)
    _import_previous_logs()
    config.set_deadline(seconds_from_now=SESSION_BUDGET_H * 3600.0)
    print("\nsession budget: %.1f h (deadline at %s)"
          % (SESSION_BUDGET_H, time.strftime("%H:%M", time.localtime(
              config.SESSION_DEADLINE))))
    from ppml import experiments as _exp
    stages = [s for s in RUN_STAGES if s in _exp.STAGES]

    # ── load every dataset of this notebook once ──────────────────────────────
    harnesses = {}
    for name in RUN_DATASETS:
        print("\n" + "#" * 78 + "\n# loading %s\n" % name + "#" * 78)
        try:
            cap = MAX_WINDOWS_BY_DATASET.get(name)
            config.MAX_WINDOWS = cap
            bundle = datasets.load(name, max_subjects=MAX_SUBJECTS, max_windows=cap or 0)
        except Exception as exc:
            print("  LOAD FAILED for %s: %s" % (name, exc))
            traceback.print_exc()
            print("  attach the dataset under Add Input (or turn Internet ON) and rerun")
            continue
        h = Harness(bundle, device=models.device_for(True))
        h.window_cap = cap
        m = bundle.meta
        print("  windows=%d subjects=%d shape=%s classes=%s names=%s"
              % (m.get("n_windows"), m.get("n_subjects"), m.get("window_shape"),
                 m.get("class_counts"), m.get("class_names")))
        print("  feature extraction %.1f s (%d features); window cap=%s"
              % (h.feature_time_s, h.n_features, cap))
        if name == "WISDM v2" and datasets.WISDM_ALIGNMENT:
            fr = [v["matched_fraction"] for v in datasets.WISDM_ALIGNMENT.values()]
            print("  WISDM accel/gyro matched fraction: min %.2f median %.2f"
                  % (min(fr), sorted(fr)[len(fr) // 2]))
        try:
            _feasibility(h)
        except Exception as exc:
            print("  feasibility check failed: %s" % exc)
        harnesses[name] = h
    # each dataset's public source for dplearn = the other dataset of this notebook
    for name, h in harnesses.items():
        others = [o for o in harnesses if o != name]
        if others:
            o = harnesses[others[0]]
            h.public_source = dict(name=o.bundle.name, Xf=o.Xf, y=o.y,
                                   n_class=datasets.N_CLASS_BY_DATASET[o.bundle.name])

    # ── stage-major loop: each stage on every dataset, then the next stage ─────
    stopped = False
    for stage in stages:
        for name, h in harnesses.items():
            if stage not in STAGES_BY_DATASET.get(name, RUN_STAGES):
                continue
            if config.budget_left_s() <= 0:
                stopped = True
                break
            datasets.set_class_space(name)       # 3 classes for WISDM, 6 otherwise
            config.MAX_WINDOWS = getattr(h, "window_cap", None)
            print("\n" + "=" * 78 + "\n%s | %s  (elapsed %.1f min)"
                  % (name, stage, (time.time() - t_start) / 60.0))
            if not _run_stage(ppml, h, stage, None, None, t_start):
                stopped = True
                break
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if stopped:
            print("\nsession budget reached after %.1f min"
                  % ((time.time() - t_start) / 60.0))
            break
    names = list(harnesses)
    _report(ppml, names)
    _cross_dataset(ppml)
    _finalise(ppml, names)
    # The raw datasets are not results: remove the downloaded copies so the saved version's
    # output is only results/ and figures/ (small, quick to save, quick to attach later).
    try:
        shutil.rmtree(config.DATA_DIR, ignore_errors=True)
        print("removed downloaded raw data from %s (results and figures are kept)"
              % config.DATA_DIR)
    except Exception:
        pass
    print("total session time used: %.1f min" % ((time.time() - t_start) / 60.0))
    if stopped:
        print("NOT COMPLETE - Save Version, attach its output as an input, run again.")
    else:
        print("ALL STAGES COMPLETE for %s" % ", ".join(names))
    return ppml


if __name__ == "__main__":
    PPML = main()
