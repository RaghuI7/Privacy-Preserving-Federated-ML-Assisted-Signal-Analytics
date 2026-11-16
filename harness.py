"""Shared harness: every experiment goes through the same data path, split rule,
client construction, training loop and logger. This is what makes the four dataset
scripts methodologically identical - they differ only in `bundle`.
"""

import os
import time

import numpy as np

from . import config, datasets, features, fl, models, runners, splitting
from .runlog import RunLog


class Split:
    def __init__(self, idx, subjects, meta, key=""):
        self.train_idx = idx["train_idx"]
        self.val_idx = idx["val_idx"]
        self.test_idx = idx["test_idx"]
        self.subjects_in_split = meta["subjects_in_split"]
        self.leakage_checks = meta["leakage_checks"]
        self.key = key


class Harness:
    """Holds one dataset plus the cached feature matrix and the GPU handle."""

    def __init__(self, bundle, device=None, compute_features=True):
        self.bundle = bundle
        self.device = device or models.device_for(True)
        self.Xw = bundle.X                       # (N, WL, C) windows
        self.y = bundle.y
        self.subjects = bundle.subject
        # native_rate windows are on the device clock; everything else is 50 Hz
        self.fs = float(bundle.meta.get("fs", config.TARGET_FS))
        self.Xf = None
        if compute_features:
            t0 = time.time()
            self.Xf = features.extract_features(self.Xw, fs=self.fs)
            self.feature_time_s = time.time() - t0
        else:
            # the pooled Task 7 arm trains raw-window models only, so it never needs the
            # 117-feature matrix - but the driver prints this attribute for every harness
            self.Xf = None
            self.feature_time_s = 0.0
        self.n_features = int(self.Xf.shape[1]) if self.Xf is not None else 0
        self.logs = {}
        self._scale_cache = {}

    # ── logging ────────────────────────────────────────────────────────────
    def log(self, stage):
        if stage not in self.logs:
            self.logs[stage] = RunLog(self.bundle.name, stage)
        return self.logs[stage]

    # ── splitting ──────────────────────────────────────────────────────────
    def split(self, seed, kind="subject"):
        if kind == "subject":
            got = splitting.subject_disjoint_split(self.subjects, self.y, seed=seed)
            return Split(got, self.subjects, got, key="subject:%d" % int(seed))
        if kind == "window":
            got = splitting.window_level_split(self.y, seed=seed)
            full = dict(train_idx=got["train_idx"], val_idx=got["val_idx"],
                        test_idx=got["test_idx"])
            leak = splitting.leakage_report(self.subjects, full, kind="window_level")
            per_split = {n: sorted(set(int(s) for s in self.subjects[v]))
                         for n, v in full.items()}
            return Split(full, self.subjects,
                         dict(subjects_in_split=per_split, leakage_checks=leak),
                         key="window:%d" % int(seed))
        raise ValueError(kind)

    def clients(self, split, seed, n_clients=None, mode=None):
        """The K client silos for one FL run.

        `mode=None` takes config.CLIENT_MODE, which is one subject per client: several
        people inside one silo would make the certificate client-level but NOT user-level,
        and would pay for it in utility (sigma = z*C/K). Task 3's client-construction rows
        still name their mode explicitly, so the comparison survives the default changing.
        """
        return splitting.federated_clients_by_subject(
            split.train_idx, self.subjects, self.y, n_clients=n_clients, seed=seed,
            mode=(config.CLIENT_MODE if mode is None else mode))

    def clients_dirichlet(self, split, seed, n_clients=None, alpha=0.5, kind="dirichlet"):
        if kind == "iid":
            return splitting.iid_partition_windows(split.train_idx, n_clients, seed=seed)
        return splitting.dirichlet_partition_windows(split.train_idx, self.y, n_clients,
                                                     alpha=alpha, seed=seed)

    # ── input selection ────────────────────────────────────────────────────
    def _scaled(self, src, split, tag):
        """Zero-mean/unit-variance using TRAIN subjects only.

        The submission fitted a scaler per client and, in the pooled path, over the
        whole dataset including the test windows. Fitting on the training split alone
        keeps the transform leak-free while giving every client identical statistics.
        """
        # The column count belongs in the key: the Task 3/7/8 arms hand `input()`
        # different-width overrides (a 54-column family, an accelerometer-only window
        # block) for the same split, and reusing the 117-column statistics either
        # broadcasts wrongly or crashes.
        key = (split.key, tag, tuple(src.shape[1:]),
               int(split.train_idx[0]) if len(split.train_idx) else -1)
        key = key + (config.WINDOW_SCALE,)
        if key not in self._scale_cache:
            tr = src[split.train_idx]
            # Raw windows (N, WL, C) are standardised PER CHANNEL (one mean/std per sensor
            # axis over all training windows and time steps); per-time-step statistics
            # would scale each sample position differently. Feature rows (N, F) are
            # standardised per feature. Statistics come from the training subjects only.
            axes = (0, 1) if (tr.ndim == 3 and config.WINDOW_SCALE == "channel") else 0
            mu = tr.mean(axis=axes, keepdims=True)
            sd = tr.std(axis=axes, keepdims=True) + 1e-8
            self._scale_cache[key] = (mu.astype(np.float32), sd.astype(np.float32))
        mu, sd = self._scale_cache[key]
        return ((src - mu) / sd).astype(np.float32)

    def input(self, arch, split, X_override=None, standardize=True):
        """Return (X_train, y_train, X_test, y_test, X_val, y_val) for `arch`.

        MLP        -> 117 hand-crafted features (paper's representation)
        MLP-Raw    -> flattened windows (representation-matched control, Task 3)
        CNN/TCN    -> (N, WL, C) windows
        """
        a = arch.upper()
        tag = "features" if a in ("MLP", "MLP-SMALL") else "windows"
        src = self.Xf if a in ("MLP", "MLP-SMALL") else self.Xw
        if X_override is not None:
            src = X_override
        if src is None:
            raise RuntimeError("%s needs features but the harness was built without them" % arch)
        if standardize:
            src = self._scaled(np.asarray(src), split, tag)
        pick = lambda idx: (src[idx], self.y[idx])
        Xtr, ytr = pick(split.train_idx)
        Xte, yte = pick(split.test_idx)
        Xv, yv = pick(split.val_idx)
        # MLP-Raw keeps the (N, WL, C) shape; RawMLP flattens inside forward().
        return Xtr, ytr, Xte, yte, Xv, yv

    def scaled_all(self, arch, split):
        """The whole standardized matrix, so any GLOBAL window index can be addressed.

        `input()` returns only the three split blocks; the MIA protocol has to score
        arbitrary subject pools drawn from the training subjects.
        """
        a = arch.upper()
        src = self.Xf if a in ("MLP", "MLP-SMALL") else self.Xw
        if src is None:
            raise RuntimeError("%s needs features but the harness was built without them" % arch)
        return self._scaled(np.asarray(src), split,
                            "features" if a in ("MLP", "MLP-SMALL") else "windows")

    # ── one run, logged ────────────────────────────────────────────────────
    def _split_audit(self, split):
        """Scalar split provenance, so every logged run carries its own leakage evidence.

        The Work Order requires the log to be the only source for the tables; the audit
        numbers from `splitting` have to travel with the run, not only with the note.
        """
        if split is None:
            return {}
        out = dict(n_train_subjects=len(split.subjects_in_split.get("train", [])),
                   n_val_subjects=len(split.subjects_in_split.get("val", [])),
                   n_test_subjects=len(split.subjects_in_split.get("test", [])))
        lc = split.leakage_checks or {}
        for key in ("subjects_overlap_train_test", "subjects_overlap_train_val",
                    "subjects_overlap_val_test", "split_kind", "n_overlapping_pairs",
                    "windows_from_overlapping_subjects"):
            if key in lc:
                out["split_" + key] = lc[key]
        out["n_windows_train"] = int(len(split.train_idx))
        out["n_windows_test"] = int(len(split.test_idx))
        return out

    def run_fl(self, cfg, stage, arch="MLP", client_indices=None, split=None, seed=42,
               dp=None, method="fedavg", mu=0.0, use_fedper=False, server_momentum=False,
               attacks=None, rounds=None, local_epochs=None, X_override=None,
               heterogeneity=None, verbose=False, partition_id=0, label=None, cell=None,
               init_weights=None):
        """Execute + log one FL run. Returns the result dict (or the cached one)."""
        log = self.log(stage)
        C_ = config.cfg()
        base_cfg = dict(dataset=self.bundle.name, stage=stage, arch=arch, method=method,
                        # `cell` is the experiment-table name, e.g. "FL-Median". The raw
                        # `method` alone cannot tell FedAvg from FedProx (both "fedavg"),
                        # so paired tests key on (arch, cell) instead of guessing.
                        cell=cell or label or method,
                        seed=int(seed), partition_id=int(partition_id),
                        rounds=int(rounds or C_["rounds"]),
                        local_epochs=int(local_epochs or
                                         (C_["local_epochs_mlp"] if arch.upper().startswith("MLP")
                                          else C_["local_epochs_cnn"])),
                        batch=int(config.BATCH), lr=float(config.LR),
                        n_clients=len(client_indices) if client_indices else None,
                        dp=dp, mu=float(mu), fedper=bool(use_fedper),
                        server_momentum=bool(server_momentum),
                        attacks=attacks, client_mode=cfg.get("client_mode", "grouped"),
                        split_kind=cfg.get("split_kind", "subject"),
                        feature_kind=cfg.get("feature_kind", "handcrafted_117"),
                        jerk_with_fs=config.JERK_WITH_SAMPLING_RATE,
                        variant=cfg.get("variant", "common_50hz"),
                        harness=cfg)
        base_cfg.update(self._split_audit(split))
        chash = self.run_id(base_cfg)
        if log.has(chash):
            return None                                     # already computed
        Xtr, ytr, Xte, yte, Xv, yv = self.input(arch, split, X_override)
        cl_block = splitting.to_block_positions(client_indices, split.train_idx) \
            if client_indices else client_indices
        res = runners.run_fl((Xtr, ytr), cl_block, arch, seed, rounds=rounds,
                             local_epochs=local_epochs, method=method, mu=mu, dp=dp,
                             use_fedper=use_fedper, server_momentum=server_momentum,
                             attacks=attacks, Xval=Xv, yval=yv, Xte=Xte, yte=yte,
                             label=label or method, partition_id=partition_id,
                             device=self.device, verbose=verbose,
                             heterogeneity=heterogeneity, extra_config=cfg,
                             init_weights=init_weights)
        rec = self._record(base_cfg, res, chash)
        return res if rec is not None else None

    def run_central(self, stage, arch="MLP", split=None, seed=42, epochs=None,
                    X_override=None, cfg=None, label="Centralised", cell="Centralised"):
        log = self.log(stage)
        n_ep = runners.central_epochs(arch, epochs)
        base_cfg = dict(dataset=self.bundle.name, stage=stage, arch=arch,
                        method="centralised", cell=cell, seed=int(seed), partition_id=0,
                        epochs=int(n_ep), local_epochs=int(n_ep),
                        batch=int(config.BATCH),
                        lr=float(config.LR), split_kind=(cfg or {}).get("split_kind", "subject"),
                        variant=(cfg or {}).get("variant", "common_50hz"),
                        jerk_with_fs=config.JERK_WITH_SAMPLING_RATE,
                        harness=cfg or {})
        base_cfg.update(self._split_audit(split))
        chash = self.run_id(base_cfg)
        if log.has(chash):
            return None
        Xtr, ytr, Xte, yte, _, _ = self.input(arch, split, X_override)
        res = runners.run_central(Xtr, ytr, Xte, yte, arch, seed, epochs=epochs,
                                  device=self.device, label=label)
        rec = self._record(base_cfg, res, chash)
        return res if rec is not None else res

    # ── record shaping ─────────────────────────────────────────────────────
    def _cfg_block(self, base_cfg):
        """The config dictionary that IDENTIFIES a run, from the settings alone.

        Two things make this its own method instead of a few lines inside `_record`:

        * the resume check and the stored `config_hash` must be hashes of the SAME
          dictionary. Before, the check hashed `base_cfg` (nested `harness` block, no
          C/z/K/T/delta) and the writer hashed the flattened block, so the two never
          matched and every session recomputed work that was already in the log.
        * the profile is part of a run's identity. Smoke and full share seeds and, for
          the pinned-T DP arms, share rounds and epochs too; without the profile a
          non-reportable smoke row would suppress its reportable twin.
        """
        cfg_block = {k: v for k, v in base_cfg.items() if k != "harness"}
        cfg_block.update({k: v for k, v in (base_cfg.get("harness") or {}).items()
                          if k not in cfg_block})
        # `arch` is the short key the harness writes; the experiment tables and the
        # figures read `architecture`, so both names must be on every record.
        cfg_block["architecture"] = cfg_block.get("architecture") or base_cfg.get("arch")
        cfg_block["profile"] = config.PROFILE
        # The window cap changes how much data every run trained on, so it belongs to a
        # run's identity: a capped table must never silently satisfy an uncapped request,
        # and it gives every table a column that discloses the subsample.
        cfg_block["max_windows"] = config.MAX_WINDOWS
        return cfg_block

    def run_id(self, base_cfg):
        """The hash a run is recognised by on a later session."""
        from .runlog import stable_hash
        return stable_hash(self._cfg_block(base_cfg))

    def _record(self, base_cfg, res, config_hash=None):
        """Flatten a result into the Work Order's required log fields and write it."""
        priv = res.get("privacy") or {}
        cfg_block = self._cfg_block(base_cfg)
        # These five describe what the run actually used; they come from the result, so
        # they are recorded but deliberately not hashed (see `run_id`).
        cfg_block.setdefault("C", res.get("clip_C"))
        cfg_block.setdefault("z", res.get("dp_z"))
        cfg_block.setdefault("K", res.get("K"))
        cfg_block.setdefault("T", res.get("rounds"))
        cfg_block.setdefault("delta", res.get("dp_delta"))
        # The release's SHAPE, as the run left it. A table that only carries z cannot tell
        # the plain arm from the sparse or head-only one, and those three sit at the SAME
        # epsilon by construction - so the mechanism has to be identifiable from the log
        # alone, not from the cell name a caller happened to pass. Same reason these are
        # recorded but not hashed: they describe the run, they do not identify its settings.
        cfg_block.setdefault("dp_sparsity", res.get("dp_sparsity"))
        cfg_block.setdefault("dp_support", res.get("dp_support"))
        cfg_block.setdefault("dp_head_only", res.get("dp_head_only"))
        cfg_block.setdefault("dp_head_tensors", res.get("dp_head_tensors"))
        cfg_block.setdefault("dp_iter_avg_window", res.get("dp_iter_avg_window"))
        cfg_block.setdefault("clip_basis", res.get("clip_basis"))
        cfg_block.setdefault("init_weight_norm", res.get("init_weight_norm"))
        cfg_block.setdefault("noise_coords", res.get("noise_coords"))
        cfg_block.setdefault("total_coords", res.get("total_coords"))
        cfg_block.setdefault("signal_energy_fraction", res.get("signal_energy_fraction"))
        per_round = res.get("per_round") or []

        def _curve(key):
            # `.get`, not `[...]`: the centralised control records a shorter per-round
            # dict than the FL loop, and a missing metric must log as a missing point
            # rather than lose the whole run's record.
            return [p.get(key) for p in per_round]

        result_block = dict(
            accuracy=res.get("accuracy"), f1=res.get("f1"), f1_macro=res.get("f1_macro"),
            precision=res.get("precision"), recall=res.get("recall"),
            f1_per_class=res.get("f1_per_class"),
            precision_per_class=res.get("precision_per_class"),
            recall_per_class=res.get("recall_per_class"),
            specificity_per_class=res.get("specificity_per_class"),
            class_support=res.get("class_support"), n_test=res.get("n_test"),
            balanced_accuracy=res.get("balanced_accuracy"),
            macro_specificity=res.get("macro_specificity"),
            kappa=res.get("kappa"), roc_auc_ovr=res.get("roc_auc_ovr"),
            nll=res.get("nll"), confusion=res.get("confusion"),
            per_round_test_acc=_curve("test_acc"),
            per_round_test_loss=_curve("test_loss"),
            per_round_val_acc=_curve("val_acc"),
            per_round_val_macro_f1=_curve("val_macro_f1"),
            per_round_macro_f1=_curve("macro_f1"),
            per_round_f1_weighted=_curve("f1_weighted"),
            per_round_precision=_curve("precision"),
            per_round_recall=_curve("recall"),
            per_round_balanced_accuracy=_curve("balanced_accuracy"),
            per_round_macro_specificity=_curve("macro_specificity"),
            per_round_kappa=_curve("kappa"),
            per_round_roc_auc=_curve("roc_auc_ovr"),
            per_round_n_clients=_curve("n_clients"),
            per_round_eps_cum=_curve("eps_cum"),
            per_round_C=_curve("C"), per_round_sigma=_curve("sigma"),
            per_round_clip_fraction=_curve("clip_fraction"),
            per_round_release_snr=_curve("release_snr"),
            clip_history=res.get("clip_history"),
            wall_time_s=res.get("wall_time_s"),
            communication_payload_bytes=res.get("communication_payload_bytes"),
            client_sizes=res.get("client_sizes"),
            partition_heterogeneity=res.get("partition_heterogeneity"),
            local_epoch_time_s=res.get("local_epoch_time_s"),
        )
        for _k in ("accuracy_iter_avg", "f1_iter_avg", "f1_macro_iter_avg",
                   "iter_avg_window", "test_acc_final_round", "selected_round",
                   "best_val_acc", "model_selection", "lr_schedule", "lr_final",
                   "class_weights", "augment"):
            if res.get(_k) is not None:
                result_block[_k] = res[_k]
        if priv:
            result_block.update(
                eps=priv.get("eps"), alpha_star=priv.get("alpha_star"),
                eps_updates=priv.get("eps_updates"),
                eps_clip_fraction=priv.get("eps_clip_fraction"),
                eps_total=priv.get("eps_total", priv.get("eps")),
                # how many rounds the clip estimator actually released, so the
                # certificate in the table can be read against the mechanism it is for
                T_clip_estimator=priv.get("T_clip_estimator"),
                clip_freeze_rounds=priv.get("freeze_rounds"),
                eps_per_round=[p["eps"] for p in priv.get("per_round_eps", [])],
                mechanism=priv.get("mechanism"), adjacency=priv.get("adjacency"),
                sigma=priv.get("sigma"), accountant_note=priv.get("note"),
                # measured, not predicted: how big the clipped update was against the noise
                # that was drawn over the released support. This is the column that answers
                # "why is this arm at chance" without anyone having to re-run anything, and
                # it is a RESULT, so it sits here rather than in the hashed config block.
                release_snr_mean=res.get("release_snr_mean"),
                release_snr_first=res.get("release_snr_first"),
                release_snr_last=res.get("release_snr_last"),
                release_signal_norm=res.get("release_signal_norm"),
                release_noise_norm=res.get("release_noise_norm"))
        return self.log(base_cfg["stage"]).record(cfg_block, result_block,
                                                  config_hash=config_hash)
