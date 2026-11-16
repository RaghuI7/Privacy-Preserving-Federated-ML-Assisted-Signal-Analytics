"""Run report and contradiction flags (Work Order section 13).

Section 13 asks for four things the notebooks never produced:

  1. a README that says how each table and figure is reproduced - `README.md` at the top
     of the deliverable, and `reproduce_matrix()` / `write_reproduce_matrix()` here emit
     the same mapping as `results/reproduce_matrix.csv`;
  2. raw per-run logs and summary CSVs - written by the experiment modules;
  3. figures regenerated from the logs - `plotting.all_figures()`;
  4. a run report with hardware, software versions, total runtime and anything that
     could not be completed - `run_report()` below.

`contradictions()` is the part the Work Order asks for explicitly: "flag immediately any
result that contradicts a claim in the current paper ... The authors need to rewrite
those claims, not tune the experiment until it agrees." Every check compares a claim
quoted from the submission against a number read from the logs, and returns a row even
when the evidence is missing (verdict "NOT YET MEASURED") - it never silently drops a
claim that failed.
"""

import csv
import json
import os
import time

import numpy as np

from . import config, multiseed, plotting, runlog
from .runlog import read_logs

# Claims carried by the rejected submission, quoted so the response letter can point at
# them. Each entry is checked against the logs below.
PAPER_CLAIMS = {
    "eps.0.85": "FL+DP on UCI-HAR carries eps = 0.85 (Renyi accounting)",
    "eps.8.30": "CNN-BiLSTM DP carries eps = 8.30",
    "eps.crosscheck": "the reported epsilons agree with an independent accountant",
    "table5.wisdm": "Table 5 reports WISDM v2 FedAvg accuracy 84.9%",
    "params.30pct": "LightTCN needs ~30% fewer parameters than CNN-BiLSTM",
    "median.wins": "coordinate-wise median aggregation beats FedAvg under attack",
    "adaptdp.wins": "adaptive clipping beats fixed-C DP at equal budget",
    "adaptdp.cost": "adaptive clipping is privacy-cheap at this participation level",
    "fed.beats.central": "federated training matches or beats the centralised control",
    "resample.harmless": "linear upsampling to 50 Hz is harmless for the analytics",
    "zeropad.neutral": "zero-padding the missing WISDM gyroscope is neutral",
    "noniid.29pp": "UCI-HAR loses 29.4 pp at alpha = 0.5 (one partition)",
    "noniid.dp.scope": "the DP numbers of the non-IID sweep are user-level guarantees",
    "mia.adv": "DP reduces the membership-inference advantage to about chance",
    "deploy.measured": "the deployment-cost claims are backed by measurements",
    "pipeline.one": "all four datasets are run with the same fixed pipeline "
                    "(Work Order line 12)",
}

# Worked around, not run: the sheet marks each of these optional ("if feasible", "if
# time allows", "energy is optional"). They are declared here so the run report lists
# them as not measured instead of a reader assuming they were.
NOT_ATTEMPTED = (
    ("energy", "Task 9: energy per inference is not measured. The deployment claims stay "
               "prospective wording, as the Work Order permits."),
    ("dpsgd-example-level", "Task 1 optional extra: example-level DP-SGD is implemented "
                            "(experiments/exp_dpsgd.py, accountant "
                            "dp_accountant.account_dpsgd) but is opt-in in the per-dataset "
                            "files, so unless the stage was added to RUN_STAGES no "
                            "example-level number exists. Set RUN_STAGES = ['dpsgd'] "
                            "for it. It is a DIFFERENT PRIVACY "
                            "UNIT - one window, not one person - and its row is labelled "
                            "adjacency=example_level, user_level_certificate=NO. No "
                            "user-level claim in the paper may cite it."),
    ("pooled-multi-dataset", "Task 7 optional: the pooled cross-sensor channel arm "
                             "(exp_resampling.pooled_run) runs in the ALL-datasets session "
                             "only, because it needs a gyroscope dataset and a "
                             "gyroscope-less dataset in one stack; a single-dataset script "
                             "prints that it skipped. Use RUN_DATASETS = ['UCI HAR', "
                             "'WISDM v2'] with RUN_STAGES = ['resampling'] - that is "
                             "the whole session. Until it has run, the mask arm is a single-dataset "
                             "comparison, which is the case where a mask and a constant "
                             "cannot differ."),
)

# Arms that ran, but on a narrower scope than the full grid - the reader has to see this
# next to the numbers, not in a footnote.
SCOPED_ARMS = (
    ("byzantine non-IID", "the Dirichlet arm of Task 6 runs the MLP only "
                          "(config.BYZ_NONIID_ARCHES) for session cost"),
    ("Task 5 DP arm", "one z (%.1f) with both aggregators; its epsilon is not a user-level "
                      "certificate under a Dirichlet partition" % config.NONIID_DP_Z),
    ("MIA shadows", "shadow models are trained without DP (they only set the threshold); "
                    "the DP level is applied to the target, as the protocol requires"),
    ("MIA DP levels", "the membership-inference sweep attacks the non-DP model and five DP "
                      "levels (three z, the sparse release and the head-only release), so "
                      "every arm that Task 1 and Task 4 price is also MIA-tested. The "
                      "adaptive-C arm is NOT - its own certificate is the open question "
                      "there, and adding a sixth level would multiply the trial count for "
                      "the one arm whose epsilon already exceeds the others."),
)


def profile_scope_notes():
    """What replication and grid the ACTIVE profile actually runs, computed not asserted.

    Section 12 of the Work Order puts >= 6 seeds on Tasks 1, 2, 3, 7 and 8 and names six
    clip radii and four corruption fractions. Those numbers used to be hardcoded here,
    which meant this function could only ever be as honest as the last edit. Now the run
    report reads the grid out of the profile that produced the logs, so a thinner session
    says so itself and a compliant one says that too.
    """
    cfg = config.cfg()
    need = {"seeds": len(config.SEEDS), "seeds_supp": len(config.SEEDS),
            "dp_seeds": len(config.SEEDS), "byz_seeds": len(config.SEEDS_BYZANTINE),
            "mia_trials": len(config.SEEDS_MIA)}
    out = []
    thin = ["%s=%d (sheet asks %d)" % (k, len(cfg[k]), v)
            for k, v in need.items() if len(cfg[k]) < v]
    if cfg.get("clip_sweep") and len(cfg["clip_sweep"]) < len(config.CLIP_SWEEP):
        thin.append("clip radii=%d of %d" % (len(cfg["clip_sweep"]),
                                             len(config.CLIP_SWEEP)))
    for key, label, want in (("partitions", "partitions per alpha",
                             config.N_PARTITIONS_PER_ALPHA),
                             ("partitions_dp", "DP partitions per alpha",
                              config.N_PARTITIONS_DP),
                             ("byz_fractions", "corruption fractions",
                              len(config.BYZ_FRACTIONS))):
        v = cfg.get(key)
        if v is None:
            continue
        n = len(v) if isinstance(v, (tuple, list)) else int(v)
        if n < want:
            thin.append("%s=%d of %d" % (label, n, want))
    methods_note = None
    if config.METHODS_FILTER:
        keep = set(config.METHODS_FILTER)
        skipped = [l for l, _ in config.MAIN_METHODS if l not in keep]
        methods_note = (
            "DP-only session (config.METHODS_FILTER is set)",
            "Task 4 ran %d of %d arms; not run at all: %s. Section 12 asks for 'all FL "
            "methods + centralised', so these logs are a mechanism check and not the Task 4 "
            "table. Three of the sheet's paired tests lose their other arm and come back as "
            "'insufficient paired seeds' rather than as a result: median_vs_fedavg, "
            "fedprox_vs_fedavg and fedavg_vs_centralised - which means the claims 'median "
            "beats FedAvg' and 'federated beats centralised' cannot be evaluated from this "
            "session. The pairs that stay valid are the DP ones (adaptdp_vs_dp, "
            "sparse_dp_vs_plain_dp, head_dp_vs_plain_dp), since both arms of each are DP. "
            "Rerun with METHODS_FILTER=None for the reportable table."
            % (len(keep & set(l for l, _ in config.MAIN_METHODS)),
               len(config.MAIN_METHODS), ", ".join(skipped) or "none"))
    # The DP operating point, the head-sweep switch and the local-epoch budget, as
    # executed. Each is a place a session can differ from the Work Order sheet without
    # moving a single epsilon while still changing what an accuracy column means, so the
    # report states them instead of leaving a reader to infer them from the code.
    from . import dp_accountant as _acc
    _T = int(cfg["rounds_dp"])
    out.append(("DP operating point",
                "every DP arm outside the Task 1 sweep runs at z=%g, which certifies "
                "eps=%.4f at T=%d, delta=%g (config.DP_ARM_Z; the example-level DP-SGD arm "
                "keeps its own z=%g because its privacy unit is a window, not a person). "
                "The Task 1 sweep is the measured curve a budget is stated on; the accuracy "
                "a z produces is reported for that z and never the other way round."
                % (config.DP_ARM_Z,
                   _acc.account_central_gaussian(config.DP_ARM_Z, _T,
                                                 config.DELTA)["eps"],
                   _T, config.DELTA, config.DP_SGD_Z)))
    out.append(("head-only z sweep",
                "RUN: the head release is measured at every z level for %s, so its "
                "signal-to-noise curve is in privacy_table_<ds>.csv beside the plain and "
                "sparse arms." % ", ".join(config.PRIVACY_HEAD_ARCHES)
                if config.PRIVACY_HEAD_SWEEP else
                "NOT run (config.PRIVACY_HEAD_SWEEP=False): the head release is measured "
                "only at the single operating point z=%g, so the release SNR at the levels "
                "where the arithmetic says it can cross 1 is untested here. "
                "release_snr_mean on each privacy row is the column that settles it."
                % config.DP_ARM_Z))
    out.append(("local epochs",
                "E=%d (MLP) / %d (CNN) passes per client per round; the sheet names 10/5, "
                "which measures out at roughly 60-75 GPU-h for UCI HAR alone. E appears in "
                "no accountant formula - the noise is added once per round at the server, "
                "after the update is clipped to C - so this budget changes accuracy and "
                "wall-clock and no privacy number."
                % (cfg["local_epochs_mlp"], cfg["local_epochs_cnn"])))
    if thin:
        out.append(("%s profile replication" % config.PROFILE,
                    "thinner than Section 12 on: " + "; ".join(thin) +
                    ". Every table row carries its own n, and a profile other than "
                    "'full' is never a reportable result."))
    else:
        out.append(("full-profile replication",
                    "the run matrix's minimums are met as executed: %d seeds on Tasks 3/7/8, "
                    "%d seeds on every T=%d DP arm, %d clip radii, %d corruption fractions, "
                    "%d partitions per alpha (DP arm %d). No replication count or grid point "
                    "was thinned for cost."
                    % (len(cfg["seeds_supp"]), len(cfg["dp_seeds"]), config.ROUNDS_DP,
                       len(cfg.get("clip_sweep") or config.CLIP_SWEEP),
                       len(cfg.get("byz_fractions") or config.BYZ_FRACTIONS),
                       cfg.get("partitions", config.N_PARTITIONS_PER_ALPHA),
                       cfg.get("partitions_dp", config.N_PARTITIONS_DP))))
    if methods_note:
        out.append(methods_note)
    return out

# Where this pipeline's mechanism differs from the rejected submission's, and why the
# difference cannot move epsilon. Each of these is asserted in the preflight
# (fl.projection_selftest / fl.clipper_selftest / dp_accountant.selftest) before a single
# run is logged, so a deviation that would silently buy accuracy with privacy stops the
# session instead of reporting a number.
MECHANISM_DEVIATIONS = (
    ("clip radius", "C is derived from the model's own initialisation norm "
                    "(config.CLIP_REL x ||w0||) instead of one constant for every "
                    "architecture. sigma = z*C/K scales with it, so z and the certificate "
                    "are unchanged; the absolute-C arm stays in the tables as published."),
    ("sparse release (FL+DP-Sparse)", "the release updates only the largest fraction of "
                                      "the ALREADY-NOISY average, selected after the draw. "
                                      "That is post-processing, so it is the same epsilon. "
                                      "Selecting on the clean average, as DP-Forward's "
                                      "output projection does, is not covered here and is "
                                      "not what this pipeline runs."),
    ("head-only release (FL+DP-Head)", "a fixed mask decided by the architecture: only the "
                                       "classifier head's tensors are released, the feature "
                                       "extractor stays at its public initialisation. A "
                                       "constant 0/1 diagonal mask cannot raise the "
                                       "sensitivity, so epsilon is identical while the "
                                       "released dimension falls from d to d_head."),
    ("server momentum and iterate averaging", "both act on aggregates that were already "
                                              "released with noise, so they are adaptive "
                                              "post-processing: free in epsilon, and they "
                                              "do not change what left the clients."),
    ("frozen clip estimator (FL+AdaptDP)", "the Gaussian quantile estimator releases for "
                                           "the first %d of %d DP rounds and C is held "
                                           "after that. The run is accounted for the "
                                           "releases it made (%d rounds of estimator cost "
                                           "instead of %d); the unfrozen certificate is "
                                           "reported beside it in adaptive_eps_split.csv."
                                           % (config.ADAPT_FREEZE_ROUNDS, config.ROUNDS_DP,
                                              min(config.ADAPT_FREEZE_ROUNDS,
                                                  config.ROUNDS_DP), config.ROUNDS_DP)),
)


# ── coverage ──────────────────────────────────────────────────────────────────
def coverage(datasets=None, stages=None):
    """How many runs each (dataset, stage) cell has, and what is still missing."""
    from .experiments import STAGES as REG
    datasets = list(datasets or plotting.ALL_DATASETS_DEFAULT)
    stages = list(stages or REG)
    profile_seeds = len(config.cfg()["seeds"])
    rows = []
    for ds in datasets:
        for st in stages:
            runs = read_logs(dataset=ds, stage=st)
            cells = {str(r.get("cell")) for r in runs if r.get("cell")}
            seeds = {int(r["seed"]) for r in runs if r.get("seed") is not None}
            parts = {int(r.get("partition_id", 0)) for r in runs}
            rows.append(dict(
                dataset=ds, stage=st, description=REG[st][1],
                n_runs=len(runs), n_cells=len(cells),
                n_seeds=len(seeds), seeds_needed=profile_seeds,
                n_partitions=len(parts),
                complete=bool(runs and len(seeds) >= profile_seeds),
                reportable=bool(runs and all(r.get("reportable") for r in runs))))
    return rows


def completeness_note(coverage_rows):
    """Human sentence about what could not be completed, for the run report."""
    missing = [r for r in coverage_rows if not r["complete"]]
    if not missing:
        return "every dataset x stage has at least one logged run"
    by_ds = {}
    for r in missing:
        by_ds.setdefault(r["dataset"], []).append(r["stage"])
    return "; ".join("%s: %s" % (ds, ", ".join(st)) for ds, st in sorted(by_ds.items()))


# ── runtime ───────────────────────────────────────────────────────────────────
def runtime(datasets=None):
    """Total wall-clock spend reconstructed from the per-run wall_time_s fields.

    Runtime is a record of what the session DID, so smoke runs count here even though
    no table or figure may use them.
    """
    datasets = list(datasets or plotting.ALL_DATASETS_DEFAULT)
    tot, per_stage, per_ds = 0.0, {}, {}
    n = 0
    for ds in datasets:
        for r in read_logs(dataset=ds, reportable_only=False):
            w = r.get("wall_time_s")
            if not isinstance(w, (int, float)):
                continue
            tot += float(w)
            n += 1
            per_stage[r.get("stage", "?")] = per_stage.get(r.get("stage", "?"), 0.0) + float(w)
            per_ds[ds] = per_ds.get(ds, 0.0) + float(w)
    return dict(total_wall_s=tot, total_wall_h=tot / 3600.0, n_timed_runs=n,
                by_stage={k: round(v, 1) for k, v in sorted(per_stage.items())},
                by_dataset={k: round(v, 1) for k, v in sorted(per_ds.items())})


# ── contradiction checks ──────────────────────────────────────────────────────
def _row(claim_id, claim, evidence, verdict, value, action):
    return dict(claim_id=claim_id, paper_claim=claim, evidence=evidence,
                verdict=verdict, measured=value, action_needed=action)


def contradictions(datasets=None):
    """Every submission claim that the logs can confirm or refute. No tuning."""
    datasets = list(datasets or plotting.ALL_DATASETS_DEFAULT)
    out = []
    out += _claim_eps(datasets)
    out += _claim_crosscheck(datasets)
    out += _claim_table5(datasets)
    out += _claim_params(datasets)
    out += _claim_median_attack(datasets)
    out += _claim_adapt(datasets)
    out += _claim_adaptive_cost(datasets)
    out += _claim_paired(datasets, "FL-FedAvg", "Centralised", "fed.beats.central")
    out += _claim_resampling(datasets)
    out += _claim_channels(datasets)
    out += _claim_noniid(datasets)
    out += _claim_noniid_scope(datasets)
    out += _claim_mia(datasets)
    out += _claim_monotone_privacy(datasets)
    out += _claim_deployment(datasets)
    out += _claim_single_pipeline(datasets)
    return out


def _claim_single_pipeline(datasets):
    """Work Order line 12, checked from the logs rather than from the settings.

    The sheet names the defect outright: the rejected WISDM sweeps (~23%) came from an older
    broken pipeline while Table 5 (84.9%) came from the fixed script. Reading the config each
    dataset actually ran with is the only way to show that cannot still be true, so a setting
    used by one dataset and not the others is CONTRADICTED here even though the source code is
    shared - a shared file that was run with different arguments is still two pipelines.
    """
    from . import crossdataset
    mine = [d for d in datasets if d in crossdataset.DATASETS]
    if len(mine) < 2:
        only = ", ".join(mine) if mine else "no dataset"
        return [_row("pipeline.one", PAPER_CLAIMS["pipeline.one"],
                     "only %s has reportable logged runs - one dataset cannot show that "
                     "the four agree" % only, "NOT YET MEASURED", "n/a",
                     "run the other per-dataset bundles and merge their exported logs with "
                     "ppml.crossdataset.consolidate([...]) before quoting a four-dataset "
                     "table")]
    rows = crossdataset.read_all(datasets=mine)
    if not rows:
        return [_row("pipeline.one", PAPER_CLAIMS["pipeline.one"],
                     "no reportable runs logged for %s" % ", ".join(mine),
                     "NOT YET MEASURED", "n/a",
                     "run the per-dataset bundles at full profile")]
    ident = crossdataset.pipeline_identity(rows=rows, datasets=mine)
    checked = len([r for r in ident if r["setting"] != "__coverage__"])
    bad = crossdataset.identity_violations(ident)
    setting_bad = [b for b in bad if b[1] != "__coverage__"]
    missing = [b for b in bad if b[1] == "__coverage__"]
    if not bad:
        return [_row("pipeline.one", PAPER_CLAIMS["pipeline.one"],
                     "%d runs over %s: %d stage/setting comparisons all agree across the "
                     "datasets that ran them, and every dataset appears in every stage "
                     "another dataset ran" % (len(rows), ", ".join(mine), checked),
                     "supported", "%d checks, 0 violations" % checked,
                     "claim stands - cross_dataset_runs.csv and pipeline_identity.csv carry "
                     "the evidence")]
    parts = []
    for stage, setting, ds in setting_bad:
        row = next(r for r in ident
                   if (r["stage"], r["setting"], r["dataset"]) == (stage, setting, ds))
        parts.append("%s/%s for %s used %s while the others used %s"
                     % (stage, setting, ds, row["values"],
                        row["agrees_with_other_datasets"].split("used ")[-1]))
    for stage, setting, ds in missing:
        parts.append("%s has no logged run at stage %s" % (ds, stage))
    verdict = "CONTRADICTED" if setting_bad else "NOT SUPPORTED"
    return [_row("pipeline.one", PAPER_CLAIMS["pipeline.one"],
                 "%d of %d stage/setting comparisons across %s disagree: %s"
                 % (len(bad), checked, ", ".join(mine), "; ".join(parts[:6])),
                 verdict, "%d violations" % len(bad),
                 "the four datasets are not one pipeline yet - re-run the datasets named "
                 "here with the current bundle so every stage shares the same settings, "
                 "and do not present the two answers side by side until they do"
                 if setting_bad else
                 "a dataset is absent from a stage another dataset ran (the sheet's Table 7 "
                 "complaint) - run that stage for the dataset named, or scope the claim to "
                 "the datasets that actually have the number")]


def _claim_crosscheck(_datasets=None):
    """Task 1 deliverable 4: our closed form against an independent accountant.

    The submission's epsilon was never checked against anything, which is how it drifted.
    A row is emitted even when dp-accounting is absent - "NOT YET MEASURED" here means the
    cross-check itself is missing, and the reader must not take the closed form on faith.
    """
    from .experiments import exp_privacy
    try:
        rows = exp_privacy.crosschecks()
    except Exception as exc:                       # pragma: no cover
        return [_row("eps.crosscheck", PAPER_CLAIMS["eps.crosscheck"],
                     "cross_check() raised %s" % exc, "UNEXPECTED", "n/a",
                     "fix the accountant before quoting any epsilon")]
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        why = rows[0].get("error", "dp-accounting unavailable") if rows else "no levels"
        return [_row("eps.crosscheck", PAPER_CLAIMS["eps.crosscheck"],
                     "independent accountant not run (%s)" % why, "NOT YET MEASURED",
                     "n/a",
                     "install dp-accounting in its own cell and re-run "
                     "ppml.experiments.exp_privacy.write_crosschecks(), or drop the "
                     "closed-form epsilon from the paper")]
    worst = max(ok, key=lambda r: abs(float(r.get("rel_diff") or 0.0)))
    wd = abs(float(worst.get("rel_diff") or 0.0))
    return [_row("eps.crosscheck", PAPER_CLAIMS["eps.crosscheck"],
                 "%d/%d z levels agree with dp-accounting; worst relative difference "
                 "%.2e at z=%.1f (ours %.4f vs theirs %.4f)"
                 % (len(ok), len(rows), wd, float(worst["z"]), worst["ours"],
                    worst["theirs"]),
                 "supported" if wd < 1e-3 else "CONTRADICTED",
                 "worst rel diff %.2e" % wd,
                 "the closed form disagrees with the reference implementation; every "
                 "epsilon in the paper has to be recomputed" if wd >= 1e-3 else
                 "claim stands")]


def _claim_adaptive_cost(datasets=None):
    """Adaptive clipping costs privacy as well as accuracy - the split is computed.

    Like the legacy-epsilon check this needs no runs: the cost of the estimator is a
    property of the settings, and it is the same for all four datasets, so one row.
    """
    from . import dp_accountant
    from .experiments import exp_clip
    try:
        sp = exp_clip.eps_split("config")
    except Exception as exc:                       # pragma: no cover
        return [_row("adaptdp.cost", PAPER_CLAIMS["adaptdp.cost"],
                     "eps_split() raised %s" % exc, "UNEXPECTED", "n/a",
                     "fix the adaptive accounting before quoting its epsilon")]
    tot, base = float(sp["eps_total"]), float(sp["eps_updates_fixed"])
    share = float(sp["clip_fraction_share"])
    all_t = float(sp["eps_total_all_rounds"])
    W, T = int(sp["T_clip_estimator"]), int(sp["T"])
    costly = tot > 1.5 * base
    rows = [_row("adaptdp.cost", PAPER_CLAIMS["adaptdp.cost"],
                 "at z=%.1f, T=%d the update mechanism alone costs eps=%.3f. The "
                 "quantile estimator (sigma_b=%.3g, i.e. %.3g after dividing by K=%d) is "
                 "charged for the %d rounds it releases and then frozen, which raises the "
                 "total to eps=%.3f (%.0f%% of it is the clip estimate); leaving it "
                 "running for all %d rounds, as Andrew et al. publish it, would cost "
                 "eps=%.3f"
                 % (sp["z"], T, base, sp["sigma_b"],
                    float(sp["sigma_b"]) / config.N_CLIENTS, config.N_CLIENTS,
                    W, tot, 100.0 * share, T, all_t),
             "CONTRADICTED" if costly else "supported",
             "adaptive total eps %.3f vs fixed %.3f (%.1fx); %.3f un-frozen"
             % (tot, base, tot / max(base, 1e-12), all_t),
             "even after freezing the estimator after %d of %d rounds the adaptive arm "
             "cannot be sold at the paper's epsilon: quote eps=%.1f for it, or make "
             "sigma_b larger and say the clip estimate is coarse" % (W, T, tot)
             if costly else "claim stands")]
    return rows + _claim_adaptive_releases(datasets, sp, W)


def _claim_adaptive_releases(datasets, sp, W):
    """The adaptive arm's certificate has to match the releases it actually logged.

    eps_split() prices the estimator for W rounds; that is only a certificate if the run
    really stopped releasing after W. The clipper records each release in its own
    clip_history and the accountant is called with the count it read back, so this check
    closes the loop: logged count == charged count == logged epsilon.
    """
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="clip")
                if r.get("eps_total") is not None
                and r.get("T_clip_estimator") is not None]
        if not runs:
            continue
        charged = [int(r["T_clip_estimator"]) for r in runs]
        logged = []
        for r in runs:
            hist = r.get("clip_history") or []
            rel = [h for h in hist if h.get("estimator_released")]
            if hist:
                logged.append(len(rel))
        eps_bad = [float(r["eps_total"]) for r in runs
                   if abs(float(r["eps_total"]) - float(sp["eps_total"]))
                   > 1e-6 * max(1.0, float(sp["eps_total"]))]
        cnt_bad = [n for n in charged if n != W]
        if logged:
            cnt_bad += [n for n in logged if n != W]
        consistent = not eps_bad and not cnt_bad
        rows.append(_row(
            "adaptdp.release_count",
            "self-check: the charged estimator rounds equal the released ones in the log",
            "%s: %d adaptive run(s), charged %s estimator rounds, freeze set at %d, "
            "logged eps %s vs analytic eps %s%s"
            % (ds, len(runs), sorted(set(charged)), W,
               "%.3f" % float(runs[0]["eps_total"]), "%.3f" % float(sp["eps_total"]),
               "" if consistent else "; mismatches: eps %s, counts %s"
               % (["%.3f" % e for e in eps_bad], cnt_bad)),
            "supported" if consistent else "UNEXPECTED",
            "charged %s, freeze %d" % (sorted(set(charged)), W),
            "the adaptive arm's epsilon is not the epsilon of what it released; do not "
            "report it until the count and the certificate agree" if not consistent
            else "certificate and release log agree"))
    return rows


def _claim_noniid_scope(datasets):
    """A Dirichlet client mixes subjects, so its certificate is not a user-level one."""
    rows = []
    for ds in datasets:
        dp_runs = [r for r in read_logs(dataset=ds, stage="noniid")
                   if r.get("certificate_scope") and float(r.get("z") or 0.0) > 0]
        if not dp_runs:
            continue
        scoped = [r for r in dp_runs
                  if "NOT a user-level" in str(r.get("certificate_scope"))]
        alphas = sorted({str(r.get("alpha")) for r in scoped})
        if not scoped:
            rows.append(_row("noniid.dp.scope", PAPER_CLAIMS["noniid.dp.scope"],
                             "%s: every DP non-IID run is labelled user-level" % ds,
                             "supported", "n/a", "claim stands"))
            continue
        rows.append(_row("noniid.dp.scope", PAPER_CLAIMS["noniid.dp.scope"],
                         "%s: %d DP runs at alpha=%s are logged as partition-level; the "
                         "clients there draw from more than one subject, so one client's "
                         "data is spread across the silos and the add/remove-a-silo "
                         "adjacency no longer holds"
                         % (ds, len(scoped), ", ".join(alphas)),
                         "CONTRADICTED",
                         "%d of %d DP non-IID runs are not user-level"
                         % (len(scoped), len(dp_runs)),
                         "present the non-IID DP numbers as round-level privacy on a "
                         "fixed partition, never as a user-level guarantee; only the "
                         "subject-silo (IID) arm supports that wording"))
    return rows


def _claim_deployment(_datasets=None):
    """Nothing in the Work Order's measured-cost list includes energy."""
    return [_row("deploy.measured", PAPER_CLAIMS["deploy.measured"],
                 "ppml.experiments.exp_cost measures params, MACs, inference latency and "
                 "peak memory from the logs; it has no energy sensor and the Work Order "
                 "marks energy optional (\"if feasible\"), so none was collected",
                 "NOT YET MEASURED", "energy: n/a",
                 "keep the deployment section prospective, or add a measured "
                 "millijoule-per-inference number before claiming one")]


def _claim_eps(_datasets=None):
    """The reported epsilons against what the accountant says the settings cost.

    This one needs no runs: the submission's own settings are enough to show what its
    epsilon should have been.
    """
    from . import dp_accountant
    zl = dp_accountant.legacy_equivalent_z(config.LEGACY_NOMINAL_EPS)
    whole = dp_accountant.account_central_gaussian(zl, 50)["eps"]
    cnn = dp_accountant.account_per_tensor_clipping(
        zl, 50, config.LEGACY_N_TENSORS["CNN-BiLSTM"])["eps"]
    rows = []
    for cid, got, kind in (("eps.0.85", whole, "whole-model clip"),
                           ("eps.8.30", cnn, "per-tensor clip")):
        want = 0.85 if cid == "eps.0.85" else 8.30
        rows.append(_row(cid, PAPER_CLAIMS[cid],
                         "legacy settings (nominal eps=%.1f, K=%d, T=50) -> z=%.3f; "
                         "the correct client-level certificate is eps=%.1f (%s)"
                         % (config.LEGACY_NOMINAL_EPS, config.N_CLIENTS, zl, got, kind),
                         "CONTRADICTED" if got > 20 else "supported",
                         "reported %.2f vs certified %.1f" % (want, got),
                         "the paper's epsilon must be replaced by the certificate of the "
                         "settings actually run" if got > 20 else "no change"))
    return rows


def _claim_table5(datasets):
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="main")
                if str(r.get("cell")) == "FL-FedAvg" and _val(r) is not None]
        if not runs:
            if ds == "WISDM v2":
                rows.append(_row("table5.wisdm", PAPER_CLAIMS["table5.wisdm"],
                                 "no main-stage FL-FedAvg run logged for %s" % ds,
                                 "NOT YET MEASURED", "n/a",
                                 "run the main stage for %s before restating Table 5" % ds))
            continue
        s = multiseed.summarise([_val(r) for r in runs])
        if ds != "WISDM v2":
            continue
        gap = abs(s["mean"] - 0.849)
        rows.append(_row("table5.wisdm", PAPER_CLAIMS["table5.wisdm"],
                         "%s FL-FedAvg over %d seeds: %.4f +/- %.4f (CI %.4f-%.4f)"
                         % (ds, s["n"], s["mean"], s["std"], s["ci95_low"], s["ci95_high"]),
                         "supported" if gap < 0.05 else "CONTRADICTED",
                         "%.1f%% measured vs 84.9%% in Table 5" % (100 * s["mean"]),
                         "Table 5 and the figures must both come from this pipeline; the "
                         "older ~23%% sweep has to be deleted" if gap >= 0.05 else
                         "Table 5 stands, redraw the figure from the same log"))
    return rows


def _val(r, metric="accuracy"):
    v = r.get(metric)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _claim_median_attack(datasets):
    """The median claim is about attacks, so it is tested on the byzantine logs.

    One row per (architecture, partition kind) that actually ran - the Dirichlet arm is
    where a robust aggregator is supposed to earn its keep, so it is judged separately
    instead of averaged into the IID numbers.
    """
    from .experiments import exp_byzantine
    rows = []
    for ds in datasets:
        for arch in sorted({str(r.get("architecture") or r.get("arch") or "?")
                           for r in read_logs(dataset=ds, stage="byzantine")}):
            for part in sorted({str(r.get("partition") or "iid")
                                for r in read_logs(dataset=ds, stage="byzantine")
                                if str(r.get("architecture") or r.get("arch") or "?")
                                == arch}):
                tab = [r for r in exp_byzantine.table(ds)
                       if r["architecture"] == arch and r["adversary_fraction"] > 0
                       and str(r.get("partition")) == part]
                if not tab:
                    continue
                by_agg = {}
                for r in tab:
                    by_agg.setdefault(r["aggregator"], []).append(r["acc_mean"])
                ranked = sorted(((float(np.mean(v)), k, len(v))
                                 for k, v in by_agg.items()), reverse=True)
                med = [x for x in ranked if x[1] == "median"]
                avg = [x for x in ranked if x[1] == "fedavg"]
                if not med or not avg:
                    continue
                best = ranked[0]
                rows.append(_row("median.wins", PAPER_CLAIMS["median.wins"],
                                 "%s/%s/%s: mean accuracy under attack - %s"
                                 % (ds, arch, part,
                                    ", ".join("%s %.4f" % (k, m) for m, k, _n2
                                              in ranked)),
                                 "supported" if med[0][0] > avg[0][0] else "CONTRADICTED",
                                 "median %.4f vs fedavg %.4f (best is %s)"
                                 % (med[0][0], avg[0][0], best[1]),
                                 "name the aggregator that actually held up (%s) instead "
                                 "of asserting median" % best[1]
                                 if med[0][1] != best[1] or med[0][0] <= avg[0][0]
                                 else "claim stands"))
    return rows


def _claim_params(datasets):
    from .experiments import exp_cost
    rows = []
    for ds in datasets:
        t = exp_cost.ratio_table(ds)
        if not t:
            continue
        red = float(t["params_reduction_pct"])
        rows.append(_row("params.30pct", PAPER_CLAIMS["params.30pct"],
                         "measured params CNN-BiLSTM=%d, LightTCN=%d -> %.1f%% fewer"
                         % (t["params_cnn"], t["params_tcn"], red),
                         "CONTRADICTED" if red > 40 else "supported",
                         "%.1f%%" % red,
                         "state the measured reduction (%.0f%%), not ~30%%" % red))
    return rows


def _claim_paired(datasets, hi, lo, cid, claim=None):
    """Is `hi` actually better than `lo`? Verdicts are threshold-based, never tuned.

    A claim is CONTRADICTED as soon as one architecture is materially worse
    (more than 2 pp), NOT SUPPORTED when it never wins significantly, and supported
    only when it wins on every architecture with at least one Holm-significant pair.
    """
    from .experiments import exp_main
    claim = claim or PAPER_CLAIMS.get(cid, "%s beats %s" % (hi, lo))
    rows = []
    for ds in datasets:
        good = []
        for t in exp_main.paired_tests(ds):
            forward = t.get("high") == hi and t.get("low") == lo
            reverse = t.get("high") == lo and t.get("low") == hi
            if not (forward or reverse) or t.get("mean_diff") is None:
                continue
            if forward:
                good.append(t)
            else:
                # the logged pair is stored as high - low; read it from the claim's side
                flip = dict(t)
                flip["high"], flip["low"] = lo, hi
                flip["mean_diff"] = -float(t["mean_diff"])
                good.append(flip)
        if not good:
            rows.append(_row(cid, claim, "%s: no paired seeds for %s vs %s"
                             % (ds, hi, lo), "NOT YET MEASURED", "n/a",
                             "run the main stage for %s on >= 6 seeds" % ds))
            continue
        worst = min(good, key=lambda t: t["mean_diff"])
        sig = [((t.get("holm_p") if t.get("holm_p") is not None else t.get("p_ttest"))
                or 1.0) < 0.05 for t in good]
        wins_everywhere = all(t["mean_diff"] > 0 for t in good)
        materially_worse = worst["mean_diff"] < -0.02
        evidence = ("%s: %s (paired over %d architectures, Holm-significant in %d/%d)"
                    % (ds, "  ".join("%s-%s=%+.4f" % (t["high"], t["low"], t["mean_diff"])
                                     for t in good), len(good), int(sum(sig)), len(good)))
        measured = "worst paired difference %+.4f on %s" % (worst["mean_diff"],
                                                            worst["architecture"])
        if materially_worse:
            verdict = "CONTRADICTED"
            action = ("%s is materially worse than %s on %s, so the claim as written "
                      "cannot stay in the paper" % (hi, lo, worst["architecture"]))
        elif wins_everywhere and any(sig):
            verdict = "supported"
            action = "claim stands"
        else:
            verdict = "NOT SUPPORTED"
            action = ("the difference is within noise: rewrite it as not significant, do "
                      "not add seeds until it turns positive")
        rows.append(_row(cid, claim, evidence, verdict, measured, action))
    return rows


def _claim_adapt(datasets):
    from .experiments import exp_clip
    rows = []
    for ds in datasets:
        t = exp_clip.paired(ds, arch="MLP")
        if t.get("mean_diff") is None:
            rows.append(_row("adaptdp.wins", PAPER_CLAIMS["adaptdp.wins"],
                             "%s: adaptive vs fixed not paired yet" % ds,
                             "NOT YET MEASURED", "n/a", "run the clip stage for %s" % ds))
            continue
        better = t["mean_diff"] > 0
        sig = (t.get("holm_p") or t.get("p_wilcoxon") or 1.0) < 0.05
        rows.append(_row("adaptdp.wins", PAPER_CLAIMS["adaptdp.wins"],
                         "%s: AdaptDP - DP = %+.4f, p=%.4f (n=%d seeds)"
                         % (ds, t["mean_diff"], t["p_wilcoxon"], t["n_seeds"]),
                         "supported" if (better and sig) else
                         ("NOT SUPPORTED" if better else "CONTRADICTED"),
                         "accuracy delta %+.4f over %d paired seeds (p=%.4f)"
                         % (t["mean_diff"], t["n_seeds"], t["p_wilcoxon"]),
                         "report adaptive clipping as a privacy-estimator improvement, "
                         "not an accuracy win" if not (better and sig) else "claim stands"))
    return rows


def _claim_resampling(datasets):
    rows = []
    for ds in datasets:
        sp = [r for r in read_logs(dataset=ds, stage="resampling")
              if r.get("arch_role") == "spectral_distortion"]
        if not sp:
            continue
        r = sp[-1]
        lin = _val(r, "spurious_energy_fraction_linear")
        poly = _val(r, "spurious_energy_fraction_polyphase")
        if lin is None:
            rows.append(_row("resample.harmless", PAPER_CLAIMS["resample.harmless"],
                             "%s: already at 50 Hz, no upsampling artefact possible" % ds,
                             "not applicable", "n/a",
                             "say the ablation is vacuous for this dataset"))
            continue
        rows.append(_row("resample.harmless", PAPER_CLAIMS["resample.harmless"],
                         "%s: linear 20->50 Hz puts %.3f of the energy above the source "
                         "Nyquist; polyphase puts %.3f" % (ds, lin, poly if poly is not None
                                                           else float("nan")),
                         "CONTRADICTED" if lin > 0.02 else "supported",
                         "spurious fraction %.4f (reduction %.2fx)"
                         % (lin, (lin / max(poly, 1e-12)) if poly else float("nan")),
                         "resampling must be described as a measured distortion and the "
                         "polyphase variant rerun" if lin > 0.02 else "claim stands"))
    return rows


def _claim_channels(datasets):
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="resampling")
                if r.get("channel_mode") and _val(r) is not None]
        if not runs:
            continue
        by = {}
        for r in runs:
            by.setdefault(str(r["channel_mode"]), []).append(_val(r))
        base = by.get("zero_pad")
        if not base:
            continue
        worst = max(((m, float(np.mean(v)) - float(np.mean(base)))
                     for m, v in by.items() if m != "zero_pad"),
                    key=lambda kv: abs(kv[1]), default=None)
        if worst is None:
            continue
        rows.append(_row("zeropad.neutral", PAPER_CLAIMS["zeropad.neutral"],
                         "%s: %s vs zero-pad = %+.4f on raw windows (n=%d/%d runs)"
                         % (ds, worst[0], worst[1], len(by[worst[0]]), len(base)),
                         "CONTRADICTED" if abs(worst[1]) > 0.01 else "supported",
                         "%+.4f accuracy" % worst[1],
                         "the channel treatment is a confound and must be reported as a "
                         "result, not a footnote" if abs(worst[1]) > 0.01 else
                         "zero-padding is empirically neutral here"))
    return rows


def _claim_noniid(datasets):
    """The 29.4 pp figure is a non-DP number, so the DP arm of the same stage is excluded."""
    def plain(r):
        return (str(r.get("method")) == "fedavg" and not float(r.get("z") or 0.0))
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="noniid")
                if str(r.get("alpha")) == "0.5" and plain(r)]
        iid = [r for r in read_logs(dataset=ds, stage="noniid")
               if str(r.get("alpha")) == "IID" and plain(r)]
        if not runs or not iid:
            continue
        a = np.array([_val(r) for r in runs if _val(r) is not None])
        b = np.array([_val(r) for r in iid if _val(r) is not None])
        if len(a) < 2 or len(b) < 2:
            continue
        drop = float(b.mean() - a.mean())
        spread = float(a.std(ddof=1))
        rows.append(_row("noniid.29pp", PAPER_CLAIMS["noniid.29pp"],
                         "%s: alpha=0.5 over %d partitions, mean %.4f +/- %.4f "
                         "(range %.4f-%.4f); IID %.4f" % (ds, len(a), a.mean(), spread,
                                                         a.min(), a.max(), b.mean()),
                         "CONTRADICTED" if abs(drop - 0.294) > 0.05 else "supported",
                         "drop %.4f +/- %.4f pp across partitions"
                         % (100 * drop, 100 * spread),
                         "restate the degradation as mean +/- std over partitions; the "
                         "single-partition 29.4 pp figure is not reproducible"
                         if abs(drop - 0.294) > 0.05 else "claim stands"))
    return rows


def _claim_mia(datasets):
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="mia")
                if _val(r, "advantage") is not None]
        dp = [r for r in runs if r.get("dp_level") and r["dp_level"] != "non-DP"]
        if not dp:
            continue
        adv = np.array([_val(r, "advantage") for r in dp])
        auc = np.array([_val(r, "auc") for r in dp if _val(r, "auc") is not None])
        rows.append(_row("mia.adv", PAPER_CLAIMS["mia.adv"],
                         "%s: DP targets, advantage %.4f +/- %.4f, AUC %.4f (%d trials, "
                         "threshold on shadow data)" % (ds, adv.mean(),
                                                        adv.std(ddof=1) if len(adv) > 1 else 0.0,
                                                        auc.mean() if len(auc) else float("nan"),
                                                        len(dp)),
                         "supported" if (adv.mean() < 0.05 and (not len(auc) or
                                                                auc.mean() < 0.6))
                         else "CONTRADICTED",
                         "advantage %.3f, AUC %.3f" % (adv.mean(),
                                                       auc.mean() if len(auc) else float("nan")),
                         "the DP guarantee does not translate into membership privacy at "
                         "this epsilon; report the attack numbers, not the assurance"
                         if adv.mean() >= 0.05 else "claim stands"))
    return rows


def _claim_monotone_privacy(datasets):
    """Utility must fall as the certificate tightens; a rise is a finding, not noise.

    Grouped by mechanism as well as by architecture: the privacy stage runs the published
    plain release and the sparse release at the SAME epsilon, so pooling them would
    average two different utility curves into one meaningless point.
    """
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="privacy") if _val(r) is not None]
        by = {}
        for r in runs:
            if r.get("eps") is None:
                continue
            by.setdefault((_arch_of(r), _mech_of(r), float(r["eps"])), []).append(_val(r))
        if len(by) < 3:
            continue
        per_arch = {}
        for (arch, mech, e), v in by.items():
            per_arch.setdefault((arch, mech), []).append((e, float(np.mean(v))))
        for (arch, mech), pts in sorted(per_arch.items()):
            pts.sort(reverse=True)          # weak privacy (large eps) first
            accs = [a for _, a in pts]
            rises = [(pts[i][1], accs[i + 1] - accs[i]) for i in range(len(accs) - 1)
                     if accs[i + 1] - accs[i] > 0.02]
            if rises:
                rows.append(_row("privacy.monotone", "(internal check) utility decreases "
                                 "as epsilon tightens",
                                 "%s/%s %s: accuracy rises by %s as eps falls"
                                 % (ds, arch, mech, ", ".join("%+.3f at eps=%.2f"
                                                              % (d, e) for e, d in rises)),
                                 "UNEXPECTED", "non-monotone %s privacy-utility curve" % mech,
                                 "report it: it means the clip norm or the round count is "
                                 "doing more than the noise at those levels"))
    return rows


def _mech_of(r):
    sp = r.get("dp_sparsity")
    try:
        if sp is not None and float(sp) < 1.0:
            return "sparse release"
    except (TypeError, ValueError):
        pass
    if r.get("dp_head_only") or str(r.get("dp_support")) == "head":
        return "head-only release"
    # logs written before the release shape was recorded
    cell = str(r.get("cell") or "")
    if "sparse" in cell:
        return "sparse release"
    if "head" in cell:
        return "head-only release"
    return "plain release"


def _arch_of(r):
    return str(r.get("architecture") or r.get("arch") or "?")


# ── the deliverables ──────────────────────────────────────────────────────────
def reproduce_matrix():
    """Which module produces which table/figure - the README's mapping as a CSV."""
    from .experiments import STAGES as REG
    rows = []
    for st, (mod, what) in REG.items():
        rows.append(dict(stage=st, module=mod.__name__, work_order=what,
                         run_function="ppml.experiments.%s.run(harness)" % mod.__name__,
                         csv_writers=",".join(sorted(_writer_names(mod))),
                         figures=_figures_for(st)))
    rows.append(dict(stage="report", module="ppml.reporting",
                     work_order="section 13 run report + contradiction flags",
                     run_function="ppml.reporting.run_report()",
                     csv_writers="run_report.md,run_report.json,contradictions.csv,"
                                 "reproduce_matrix.csv,reviewer_points.csv,"
                                 "paired_tests_cross_dataset.csv,cross_dataset_runs.csv,"
                                 "cross_dataset_comparison.csv,pipeline_identity.csv",
                     figures="-"))
    # The one-file comparison is not a stage - it reads the other stages' logs - so it has
    # to be listed here or the README's mapping would omit a deliverable the sheet asks for.
    rows.append(dict(stage="cross-dataset", module="ppml.crossdataset",
                     work_order="line 12: one fixed pipeline, four datasets in one file",
                     run_function="ppml.crossdataset.write_outputs() / "
                                  "consolidate([...folders...])",
                     csv_writers="cross_dataset_runs.csv,"
                                 "cross_dataset_comparison.csv,pipeline_identity.csv",
                     figures="fig_cross_dataset, fig_cross_dataset_f1, "
                             "fig_cross_dataset_eps, fig_cross_dataset_eps_f1, "
                             "fig_privacy_utility, fig_privacy_utility_f1"))
    rows.append(dict(stage="figures", module="ppml.plotting",
                     work_order="all figures regenerated from results/logs",
                     run_function="ppml.plotting.all_figures()",
                     csv_writers="figure_manifest.csv/json",
                     figures="fig_main_*, fig_all_readings_*, fig_curves_*, "
                             "fig_forest_*, fig_metrics_*, fig_per_class_*, "
                             "fig_confusion_*, fig_eps_curve_*, fig_privacy_utility, "
                             "fig_privacy_utility_f1, fig_cross_dataset, "
                             "fig_cross_dataset_f1, fig_cross_dataset_eps, "
                             "fig_cross_dataset_eps_f1, fig_clip_*, "
                             "fig_eps_split_*, fig_leakage, fig_loso_*, fig_matched_*, "
                             "fig_noniid_*, fig_byzantine_*, fig_resampling, "
                             "fig_channels_*, fig_features_*, fig_cost_*, fig_mia_*"))
    return rows


def write_reproduce_matrix():
    """Emit the README's table/figure mapping as results/reproduce_matrix.csv."""
    rows = reproduce_matrix()
    if not rows:
        return None
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "reproduce_matrix.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


# The Work Order's appendix is an index of the 24 reviewer comments, each already carrying
# the Task number that answers it. That index is reproduced here as DATA - topics verbatim,
# tasks as the sheet writes them - so the response letter and the pipeline cannot drift into
# two different lists of what still has to be answered. `stages` is which stage has to have
# RUN to answer the point; None means the sheet assigns the point to the authors' writing,
# which no measurement can answer and which is therefore reported as such rather than as a
# gap in the code.
REVIEWER_POINTS = (
    ("R1-1", "Where noise is injected; Eq. (10) / Alg. 1 / Fig. 3 consistency", "1",
     ("privacy", "main")),
    ("R1-2", "Adjacency, same epsilon notion across architectures, reproducible epsilon "
             "table", "1", ("privacy",)),
    ("R1-3", "50 Hz resampling and Nyquist", "7", ("resampling",)),
    ("R1-4", "WISDM zero-padding confound", "7", ("resampling",)),
    ("R1-5", "Feature-family ablation; jerk definition", "8", ("features",)),
    ("R1-6", "FLOPs, memory, latency, payload", "9", ("cost",)),
    ("R1-7", "Federated vs centralised needs statistics", "4", ("main",)),
    ("R1-8", "Single seed; CIs for Tables 5-6", "4", ("main",)),
    ("R1-9", "Multiple Dirichlet partitions", "5", ("noniid",)),
    ("R1-10", "Actual Byzantine attacks", "6", ("byzantine",)),
    ("R1-11", "Adaptive clipping privacy cost", "2", ("clip",)),
    ("R2-1", "Which mechanism, adjacency, full derivation", "1", ("privacy",)),
    ("R2-2", "Accountant inputs, Renyi orders, worked example", "1", ("privacy",)),
    ("R2-3", "Adaptive clipping accounting", "2", ("clip",)),
    ("R2-4", "Window vs subject split; client construction", "3", ("splits",)),
    ("R2-5", "Architecture- and budget-matched comparisons", "1, 3, 9",
     ("main", "splits", "cost")),
    ("R2-6", "Multiple seeds, significance, code release", "4 + Section 13",
     ("main", "report")),
    ("R2-7", "MIA protocol, WISDM, stronger attacks", "10", ("mia",)),
    ("R2-8", "Harmonisation, label mapping, channel ablation", "7", ("resampling",)),
    ("R2-9", "Novelty / related work", "Authors (writing)", None),
    ("R2-10", "Deployment cost; prospective framing", "9", ("cost",)),
    ("R2-11", "Nominal vs cumulative epsilon; consistent counts", "1 + authors",
     ("privacy", "report")),
    ("R2-12", "Uncertainty in tables/figures; traceability", "4 + Section 13",
     ("main", "figures", "report")),
    ("R2-13", "Notation, proofreading", "Authors (writing)", None),
)

# Which files a stage has to leave behind for the point to be answerable in the letter.
# Kept to names the code actually writes (see reproduce_matrix), because a manifest that
# cites a file nothing produces is worse than no manifest.
_POINT_ARTIFACTS = {
    "privacy": "privacy_table_<ds>.csv, accountant_crosscheck.csv",
    "clip": "clip_sweep_<ds>.csv, adaptive_eps_split_<ds>.csv, dp_mechanism_<ds>.csv",
    "splits": "leakage_<ds>.csv, client_construction_<ds>.csv, loso_<ds>.csv, "
              "matched_<ds>.csv",
    "main": "<ds>__exp_main__summary_table.csv, paired_tests.csv, "
            "paired_tests_cross_dataset.csv",
    "noniid": "noniid_<ds>.csv",
    "byzantine": "byzantine_<ds>.csv",
    "resampling": "resampling_<ds>.csv, spectral_distortion_<ds>.csv, "
                  "label_map_table.csv, resampling_pooled.csv",
    "features": "feature_ablation_<ds>.csv, jerk_definition.csv",
    "cost": "compute_cost_<ds>.csv",
    "mia": "mia_<ds>.csv",
    "report": "run_report.md, contradictions.csv, reproduce_matrix.csv, "
              "cross_dataset_runs.csv, cross_dataset_comparison.csv, "
              "pipeline_identity.csv",
    "figures": "figures/ + figure_manifest.csv",
}


def reviewer_points(cov=None):
    """Each reviewer comment, and whether the logs contain the evidence that answers it.

    The verdict is counted from the run log, never asserted: a point whose stage has no
    logged run reads NOT YET RUN, which is the row the response letter has to be honest
    about. Two points are the authors' writing and are labelled as not answerable by
    measurement rather than as a pipeline gap.
    """
    cov = list(cov if cov is not None else coverage())
    by_stage = {}
    for c in cov:
        by_stage.setdefault(c["stage"], []).append(c)
    rows = []
    for cid, topic, task, stages in REVIEWER_POINTS:
        if stages is None:
            rows.append(dict(comment=cid, topic=topic, work_order_task=task, stages="-",
                             evidence_files="-", verdict="NOT A MEASUREMENT",
                             detail="the Work Order assigns this to the authors' writing"))
            continue
        runs, held, missing, files = 0, [], [], []
        produced_now = False
        for st in stages:
            # "report" and "figures" are not experiment stages: their artifact is this very
            # pass, so counting them as unrun would make every §13 point read PARTIAL forever
            if st in ("report", "figures"):
                produced_now = True
                files.append(_POINT_ARTIFACTS.get(st, "-"))
                continue
            cells = by_stage.get(st, [])
            n = sum(c["n_runs"] for c in cells)
            runs += n
            names = sorted({c["dataset"] for c in cells if c["n_runs"]})
            if n:
                held.extend(names)
            else:
                missing.append(st)
            files.append(_POINT_ARTIFACTS.get(st, "-"))
        verdict = ("EVIDENCE LOGGED" if not missing else
                   ("NOT YET RUN" if runs == 0 else "PARTIAL"))
        detail = "%d logged runs" % runs
        if held:
            detail += " on " + ",".join(sorted(set(held)))
        elif not produced_now:
            detail += "; no dataset has logged this stage"
        if produced_now:
            detail += "; plus this session's report and figures"
        if missing:
            detail += ". Still missing: " + ",".join(missing)
        rows.append(dict(comment=cid, topic=topic, work_order_task=task,
                         stages=",".join(stages),
                         evidence_files=" | ".join(files), verdict=verdict,
                         detail=detail))
    return rows


def write_reviewer_points(cov=None):
    """results/reviewer_points.csv - the answer sheet, with measured evidence status."""
    rows = reviewer_points(cov)
    if not rows:
        return None
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "reviewer_points.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def reviewer_points_note(rows=None):
    """The three-line summary printed in the report and on the console."""
    rows = list(rows if rows is not None else reviewer_points())
    have = [r["comment"] for r in rows if r["verdict"] == "EVIDENCE LOGGED"]
    part = [r["comment"] for r in rows if r["verdict"] == "PARTIAL"]
    none = [r["comment"] for r in rows if r["verdict"] == "NOT YET RUN"]
    write = [r["comment"] for r in rows if r["verdict"] == "NOT A MEASUREMENT"]
    return dict(n_total=len(rows), n_logged=len(have), n_partial=len(part),
                n_missing=len(none), n_writing=len(write),
                logged=",".join(have) or "-", partial=",".join(part) or "-",
                missing=",".join(none) or "-", writing=",".join(write) or "-")



def _writer_names(mod):
    return [n for n in dir(mod) if n.startswith("write_") or n == "table"
            or n.endswith("_table") or n in ("paired", "eps_split", "worked_example",
                                             "ratio_table", "crosschecks",
                                             "leakage_table", "loso_table",
                                             "matched_table", "client_note")]


def _figures_for(stage):
    m = {"main": "fig_main_<ds>, fig_all_readings_<ds>, fig_curves_<ds>, fig_forest_<ds>",
         "privacy": "fig_privacy_utility", "clip": "fig_clip_<ds>, fig_clip_history_<ds>, "
                                                  "fig_eps_split_<ds>",
         "splits": "fig_leakage, fig_loso_<ds>, fig_matched_<ds>",
         "noniid": "fig_noniid_<ds>", "byzantine": "fig_byzantine_<ds>_<arch>_<partition>",
         "resampling": "fig_resampling, fig_channels_<ds>",
         "features": "fig_features_<ds>", "cost": "fig_cost_<ds>", "mia": "fig_mia_<ds>"}
    return m.get(stage, "-")


def write_cross_dataset_paired(datasets=None):
    """Pool every dataset's Task 4 paired tests and Holm-correct across the pooled family.

    Each per-dataset table already carries `holm_p` for its own dataset x architecture
    family, which is the family the Work Order names. A response letter that says "Holm
    corrected" is read more strictly than that, so this file adds the stricter column too:
    for one comparison tag (`median_vs_fedavg`, `fedprox_vs_fedavg`, ...) every
    dataset x architecture instance of it forms one family and gets `holm_p_grouped`.
    Both columns are written, and neither replaces a raw p.
    """
    from .experiments import exp_main
    datasets = sorted(datasets or {c["dataset"] for c in coverage()})
    rows = []
    for ds in datasets:
        try:
            got = exp_main.paired_tests(ds)
        except Exception:
            got = []
        for r in got or []:
            row = dict(r)
            row.setdefault("dataset", ds)
            row["comparison_key"] = str(row.get("key") or row.get("comparison") or "")
            rows.append(row)
    if not rows:
        return None
    multiseed.holm_grouped(rows, "comparison_key")
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "paired_tests_cross_dataset.csv")
    fields = ["dataset", "comparison_key", "comparison", "architecture", "high", "low",
              "n_seeds", "mean_diff", "cohen_dz", "p_ttest", "p_wilcoxon", "holm_p",
              "holm_p_grouped", "holm_group_size"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def _metrics_preview(datasets, limit=24):
    """The full metric set for the main stage, straight from the logs.

    A compact copy of exp_main.metrics_table, embedded in the report so a pasted session
    shows accuracy, precision, recall, F1, balanced accuracy, specificity, kappa, AUC and,
    for the DP arms, the certificate and the measured release SNR in one place. The CSV with
    every column is what the paper's tables are transcribed from; this is the reading copy.
    """
    rows = []
    try:
        from .experiments import exp_main
    except Exception as exc:
        return [dict(dataset=",".join(datasets),
                     accuracy="(exp_main could not be imported: %s)"
                             % type(exc).__name__)]
    for ds in datasets:
        try:
            table = exp_main.metrics_table(ds)
        except Exception as exc:
            rows.append(dict(dataset=ds,
                             accuracy="(metrics_table failed: %s)" % type(exc).__name__))
            continue
        rows.extend(r for r in table
                    if str(r.get("stage", exp_main.STAGE)) == exp_main.STAGE)
    return rows[:limit]


def _cross_dataset_payload(seen):
    """The four datasets in one file, read only from the logs (Work Order line 12).

    `run_report` owns this so the section-13 deliverable and the three cross-dataset CSVs
    come out of the same session. Datasets with no logged run are named as absent rather
    than drawn as zero, and a stage one dataset never ran is a violation, not a blank.
    """
    from . import crossdataset
    mine = [d for d in crossdataset.DATASETS if d in set(seen)]
    absent = [d for d in crossdataset.DATASETS if d not in set(seen)]
    if not mine:
        return dict(datasets=[], absent=absent, n_runs=0, paths={},
                    comparison=[], identity=[], violations=[])
    rows = crossdataset.read_all(datasets=mine)
    ident = crossdataset.pipeline_identity(rows=rows, datasets=mine)
    # The comparison keeps all four columns even for the datasets with no rows here, which
    # is what the CSV holds and is the coverage statement the sheet asks for: a reader sees
    # the empty column, not a table quietly narrowed to what ran.
    return dict(datasets=mine, absent=absent, n_runs=len(rows),
                paths=crossdataset.write_outputs(datasets=mine),
                comparison=crossdataset.comparison_table(rows=rows),
                identity=ident,
                violations=[list(v) for v in crossdataset.identity_violations(ident)])


def _cross_dataset_md(cd):
    """The cross-dataset section of run_report.md - read from the logs, blanks stay blank."""
    from . import crossdataset

    def esc(x):
        return str(x).replace("|", "/")
    out = ["", "## The four datasets in one file (Work Order line 12)", ""]
    if not cd or not cd.get("datasets"):
        return out + ["No dataset has reportable logged runs in this session, so there is"
                      " nothing to put side by side yet. The four per-dataset bundles are"
                      " the run path; each writes the same fixed-schema"
                      " `cross_dataset_runs.csv`, and the merge step below stacks them.",
                      ""]
    ds = list(crossdataset.DATASETS)
    slugs = [crossdataset.slug(d) for d in ds]
    out += ["%d reportable runs over %s. Three files hold them:"
            % (cd["n_runs"], ", ".join(cd["datasets"])), "",
            "- `results/cross_dataset_runs.csv` - %d columns, one row per logged run, the"
            " same header whatever bundle wrote it, so the four sessions stack;" %
            len(crossdataset.RUN_COLUMNS),
            "- `results/cross_dataset_comparison.csv` - %d cells (stage x architecture x"
            " method) with every metric mean, std, 95%% interval and n per dataset;"
            % len(cd["comparison"]),
            "- `results/pipeline_identity.csv` - %d stage/setting comparisons that answer"
            " line 12 from the logs rather than from the source." %
            len([r for r in cd["identity"] if r["setting"] != "__coverage__"]), ""]
    if cd["absent"]:
        out += ["**Datasets absent from this session's logs:** %s. Kaggle wipes"
                " `/kaggle/working` between sessions, so their rows arrive through the"
                " merge step at the end of this section, not from this run."
                % ", ".join(cd["absent"]), ""]
    for metric, title in (("accuracy", "test accuracy"), ("f1_macro", "macro F1")):
        rows = [r for r in cd["comparison"]
                if any(("%s_mean__%s" % (metric, s)) in r for s in slugs)]
        if not rows:
            out += ["_No logged run carries a %s reading, so that table is empty rather"
                    " than zero._" % title, ""]
            continue
        out += ["### %s, one column per dataset" % title, "",
                "| stage | architecture | cell | " +
                " | ".join(esc(d) for d in ds) + " | all four |",
                "| --- | --- | --- | " + " | ".join("---" for _ in ds) + " | --- |"]
        for r in rows[:40]:
            cells = []
            for s in slugs:
                mean = r.get("%s_mean__%s" % (metric, s))
                if mean is None:
                    cells.append("| - ")
                    continue
                std = r.get("%s_std__%s" % (metric, s)) or 0.0
                ci = r.get("%s_ci__%s" % (metric, s)) or ""
                cells.append("| %.4f +-%.4f %s (n=%s) "
                             % (mean, std, esc(ci), r.get("%s_n__%s" % (metric, s), 1)))
            out.append("| %s | %s | %s %s| %s |"
                       % (esc(r["stage"]), esc(r["architecture"]), esc(r["cell"]),
                          "".join(cells), "yes" if r.get("complete") else "no"))
        if len(rows) > 40:
            out += ["", "_%d further cells are in `cross_dataset_comparison.csv`._"
                    % (len(rows) - 40)]
        out += [""]
    n_stages = len({r["stage"] for r in cd["identity"]})
    bad = set(tuple(v) for v in cd["violations"])
    out += ["### Do the four still share one pipeline?", "",
            "%d comparisons across %d stage(s): every setting a dataset was actually run"
            " with, read back from its log rows. %s"
            % (len([r for r in cd["identity"] if r["setting"] != "__coverage__"]),
               n_stages,
               "None disagrees, so the numbers below are comparable." if not bad else
               "**%d disagree - the rows those datasets produced are not comparable, and"
               " the fix is to re-run them, not to reconcile the tables afterward.**"
               % len(bad)), ""]
    if bad:
        out += ["| stage | setting | dataset | values this dataset used | verdict |",
                "| --- | --- | --- | --- | --- |"]
        for r in cd["identity"]:
            if (r["stage"], r["setting"], r["dataset"]) in bad:
                out.append("| %s | %s | %s | %s | **%s** |"
                           % (esc(r["stage"]), esc(r["setting"]), esc(r["dataset"]),
                              esc(r["values"]), esc(r["agrees_with_other_datasets"])))
        out += [""]
    out += ["### Comparable figures", "",
            "Drawn from these logs only: `fig_cross_dataset` / `fig_cross_dataset_f1`"
            " (one grouped bar per dataset, same axes, chance line marked), "
            "`fig_cross_dataset_eps` / `_f1` and `fig_privacy_utility` / `_f1`, which are"
            " the Work Order's line 65 request - accuracy and macro F1 against cumulative"
            " epsilon for every dataset and architecture.", "",
            "### Merging the four sessions", "",
            "`/kaggle/working` does not survive between sessions, so the four files are"
            " stacked afterwards, from anywhere that can import the package:", "",
            "```",
            "from ppml import crossdataset",
            "crossdataset.consolidate([<session 1 results folder>, <session 2 results"
            " folder>, ...])",
            "```", "",
            "It accepts exported `results/` folders, `results/logs` folders, individual"
            " `.jsonl` logs or `cross_dataset_runs.csv` files, refuses a CSV whose header"
            " is not the standard one, dedupes runs on `config_hash`, and rewrites all"
            " three CSVs plus the figures above from the combined set.", ""]
    return out


def run_report(datasets=None, stages=None, write_figures=True, also_contradictions=True):
    """Write results/run_report.md + .json and print the headline lines."""
    config.ensure_dirs()
    cov = coverage(datasets, stages)
    seen = sorted({c["dataset"] for c in cov})
    rt = runtime(seen)
    con = contradictions(seen) if also_contradictions else []
    figures = plotting.all_figures(seen) if write_figures else []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    points = reviewer_points(cov)
    payload = dict(generated_at=stamp, profile=config.PROFILE,
                   reportable=config.is_reportable(), hardware=runlog.hardware(),
                   versions=runlog.versions(), runtime=rt, coverage=cov,
                   incomplete=completeness_note(cov),
                   reviewer_points=points,
                   reviewer_summary=reviewer_points_note(points),
                   not_attempted=list(NOT_ATTEMPTED),
                   scoped_arms=list(SCOPED_ARMS) + profile_scope_notes(),
                   mechanism_deviations=list(MECHANISM_DEVIATIONS),
                   contradictions=con,
                   cross_dataset=_cross_dataset_payload(seen),
                   metrics_preview=_metrics_preview(sorted(
                       {c["dataset"] for c in cov})),
                   figures=[{k: v for k, v in f.items() if k != "_runs"}
                            for f in figures],
                   claims_quoted=dict(PAPER_CLAIMS))
    with open(os.path.join(config.RESULTS_DIR, "run_report.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    _write_contradictions_csv(con)
    write_reproduce_matrix()
    payload["reviewer_points_csv"] = write_reviewer_points(cov)
    payload["paired_tests_cross_dataset"] = write_cross_dataset_paired(seen)
    md = _markdown(payload)
    path = os.path.join(config.RESULTS_DIR, "run_report.md")
    with open(path, "w") as fh:
        fh.write(md)
    flags = [c for c in con if c["verdict"] in ("CONTRADICTED", "UNEXPECTED",
                                                "NOT SUPPORTED")]
    print("\n== run report -> %s" % path)
    print("   profile=%s reportable=%s  total wall time=%.2f h over %d timed runs"
          % (config.PROFILE, config.is_reportable(), rt["total_wall_h"],
             rt["n_timed_runs"]))
    print("   incomplete: %s" % payload["incomplete"])
    print("   claims flagged: %d of %d checked" % (len(flags), len(con)))
    cd = payload.get("cross_dataset") or {}
    if cd.get("datasets"):
        print("   four-dataset file: %d runs over %s -> cross_dataset_runs.csv,"
              " cross_dataset_comparison.csv, pipeline_identity.csv (%d identity"
              " violations)"
              % (cd["n_runs"], ", ".join(cd["datasets"]), len(cd["violations"])))
    if cd.get("absent"):
        print("      absent this session: %s (merge with crossdataset.consolidate([...]))"
              % ", ".join(cd["absent"]))
    rs = payload["reviewer_summary"]
    print("   reviewer points: %d of %d answered from the logs, %d partial, %d with no "
          "run yet, %d are the authors' writing (see reviewer_points.csv)"
          % (rs["n_logged"], rs["n_total"], rs["n_partial"], rs["n_missing"],
             rs["n_writing"]))
    if rs["missing"] != "-":
        print("      no evidence yet for: %s" % rs["missing"])
    if rs["partial"] != "-":
        print("      partly answered:       %s" % rs["partial"])
    for c in flags:
        print("   !! [%s] %s" % (c["verdict"], c["evidence"]))
    return payload


def _write_contradictions_csv(con):
    if not con:
        return
    path = os.path.join(config.RESULTS_DIR, "contradictions.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(con[0]))
        w.writeheader()
        w.writerows(con)


def _md_num(row, metric):
    """`mean+-std [CI]` for one metric of one cell, or an honest blank."""
    mean = row.get("%s_mean" % metric)
    if mean is None:
        return ""
    std = row.get("%s_std" % metric)
    lo, hi = row.get("%s_ci95_low" % metric), row.get("%s_ci95_high" % metric)
    txt = "%.4f" % mean
    if std is not None:
        txt += " +- %.4f" % std
    if lo is not None and hi is not None:
        txt += " [%.3f, %.3f]" % (lo, hi)
    return txt


def _md_cert(row, key):
    """A single-valued certificate or diagnostic column, formatted or blank."""
    v = row.get(key)
    if v is None or isinstance(v, (list, dict)):
        return ""
    try:
        return "%.4f" % float(v)
    except (TypeError, ValueError):
        return str(v)


def _markdown(p):
    def esc(x):
        return str(x).replace("|", "/")
    lines = ["# Run report - IEEE Access-2026-40021 revision",
             "",
             "Generated %s by `ppml.reporting.run_report()`. Profile `%s` "
             "(reportable: **%s**)." % (p["generated_at"], p["profile"], p["reportable"]),
             "",
             "## Hardware and software",
             "",
             "| item | value |",
             "| --- | --- |"]
    for k, v in sorted((p["hardware"] or {}).items()):
        lines.append("| hardware %s | %s |" % (k, esc(v)))
    for k, v in sorted((p["versions"] or {}).items()):
        lines.append("| %s version | %s |" % (k, esc(v)))
    lines += ["", "## Runtime", "",
              "Total wall-clock in the logged runs: **%.2f h** across %d timed runs."
              % (p["runtime"]["total_wall_h"], p["runtime"]["n_timed_runs"]),
              "", "| stage | hours |", "| --- | --- |"]
    for k, v in sorted(p["runtime"]["by_stage"].items()):
        lines.append("| %s | %.2f |" % (esc(k), v / 3600.0))
    lines += ["", "## Coverage (what actually ran)", "",
              "| dataset | stage | runs | cells | seeds | partitions | complete | reportable |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in p["coverage"]:
        lines.append("| %s | %s | %d | %d | %d | %d | %s | %s |"
                     % (esc(c["dataset"]), esc(c["stage"]), c["n_runs"], c["n_cells"],
                        c["n_seeds"], c["n_partitions"],
                        "yes" if c["complete"] else "NO",
                        "yes" if c["reportable"] else "NO"))
    lines += ["", "**Could not be completed:** %s" % p["incomplete"], ""]
    lines += ["", "## Results - full metric set (main stage, from the logs)", "",
              "`mean+-std` over the seeds that logged the cell, 95% t-interval in brackets. "
              "Blank means that reading was never logged, not that it was zero. The complete "
              "column set, including per-class and confusion tables, is in "
              "`results/tables/<dataset>__exp_main__metrics_table.csv`.", "",
              "| dataset | arch | cell | n | accuracy | precision | recall | macro F1 | "
              "balanced | specificity | kappa | AUC | eps | release SNR |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
              " --- | --- |"]
    for r in p.get("metrics_preview", []):
        if "cell" not in r:
            lines.append("| %s | - | %s | - | - | - | - | - | - | - | - | - | - | - |"
                         % (esc(r.get("dataset", "")), esc(r.get("accuracy", "-"))))
            continue
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     " %s | %s |"
                     % (esc(r.get("dataset", "")), esc(r.get("architecture", "?")),
                        esc(r.get("cell", "?")), r.get("n_seeds", 0),
                        _md_num(r, "accuracy"), _md_num(r, "precision"),
                        _md_num(r, "recall"), _md_num(r, "f1_macro"),
                        _md_num(r, "balanced_accuracy"), _md_num(r, "macro_specificity"),
                        _md_num(r, "kappa"), _md_num(r, "roc_auc_ovr"),
                        _md_cert(r, "eps"), _md_cert(r, "release_snr_mean")))
    lines += [""]
    lines += _cross_dataset_md(p.get("cross_dataset") or {})
    rs = p.get("reviewer_summary") or {}
    lines += ["", "## Reviewer comments, and what answers each one", "",
              "The index is the Work Order's own appendix. The status is counted from "
              "`results/logs`, not asserted, so a comment whose evidence has not been run "
              "yet reads NOT YET RUN here instead of disappearing.", ""]
    if rs:
        lines += ["%d of %d comments have logged evidence, %d are partial, %d have none "
                  "yet, %d are writing tasks no measurement can answer." % (
                      rs["n_logged"], rs["n_total"], rs["n_partial"], rs["n_missing"],
                      rs["n_writing"]), ""]
    lines += ["| comment | topic | Work Order task | stage(s) | status | detail |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in p.get("reviewer_points", []):
        lines.append("| %s | %s | %s | %s | **%s** | %s |"
                     % (esc(r["comment"]), esc(r["topic"]), esc(r["work_order_task"]),
                        esc(r["stages"]), esc(r["verdict"]), esc(r["detail"])))
    lines += ["", "**Deliberately not attempted** (the Work Order marks each of these "
              "optional, so they are listed rather than silently absent):", ""]
    for name, why in p.get("not_attempted", []):
        lines.append("- *%s* - %s" % (esc(name), esc(why)))
    lines += ["", "**Arms that ran on a narrower grid than the full matrix:**", ""]
    for name, why in p.get("scoped_arms", []):
        lines.append("- *%s* - %s" % (esc(name), esc(why)))
    lines += ["", "**Mechanism deviations from the rejected submission, and why each "
              "leaves epsilon where it was** (every one is asserted by the preflight "
              "before a run is logged):", ""]
    for name, why in p.get("mechanism_deviations", []):
        lines.append("- *%s* - %s" % (esc(name), esc(why)))
    lines += ["", "## Claims that the measurements contradict", "",
              "Nothing here was tuned to agree with the paper; each verdict is a "
              "threshold applied to a logged number.", "",
              "| claim id | paper claim | verdict | measured | evidence | what to do |",
              "| --- | --- | --- | --- | --- | --- |"]
    for c in p["contradictions"]:
        lines.append("| %s | %s | **%s** | %s | %s | %s |"
                     % (esc(c["claim_id"]), esc(c["paper_claim"])[:80], c["verdict"],
                        esc(c["measured"])[:60], esc(c["evidence"])[:160],
                        esc(c["action_needed"])))
    drawn = [f for f in p["figures"] if not f.get("skipped")]
    lines += ["", "## Figures regenerated from the logs", "",
              "%d drawn, %d skipped (a skip means the runs are not logged yet - nothing "
              "is invented to fill a panel)." % (len(drawn),
                                                 len(p["figures"]) - len(drawn)), "",
              "| figure | runs | filter |", "| --- | --- | --- |"]
    for f in drawn:
        lines.append("| %s | %s | %s |" % (esc(f.get("figure")), f.get("n_runs", ""),
                                           esc(f.get("log_filter", ""))[:90]))
    lines += ["", "## How to reproduce", "",
              "| stage | module | run | CSV writers | figures |",
              "| --- | --- | --- | --- | --- |"]
    for r in reproduce_matrix():
        lines.append("| %s | %s | `%s` | %s | %s |"
                     % (esc(r["stage"]), esc(r["module"]), esc(r["run_function"]),
                        esc(r["csv_writers"]), esc(r["figures"])[:120]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":      # pragma: no cover
    print("run report")
    run_report()
