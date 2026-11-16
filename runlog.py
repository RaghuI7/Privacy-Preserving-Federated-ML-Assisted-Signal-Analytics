"""Machine-readable run logging and resume support.

Work Order rule 2: "Log every run to a machine-readable file (JSON or CSV): dataset,
method, seed, partition id, all hyper-parameters, per-round test accuracy, final
accuracy/F1, and the accountant output. Figures are drawn only from these logs."

One JSONL line per run under results/logs/<dataset>/<stage>.jsonl, plus a flattened
summary CSV per stage. `config_hash` identifies a run uniquely, so a Kaggle session
that dies at hour 11 can be resumed without recomputing anything.
"""

import csv
import hashlib
import json
import os
import time

import numpy as np

from . import config

JSON_SUFFIX = ".jsonl"
_UNSET = object()


def _np(obj):
    if isinstance(obj, dict):
        return {str(k): _np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def stable_hash(d):
    """Deterministic hash of a config dict (order-independent, float-rounded)."""
    def norm(o):
        if isinstance(o, dict):
            return {k: norm(o[k]) for k in sorted(o)}
        if isinstance(o, (list, tuple)):
            return [norm(v) for v in o]
        if isinstance(o, float):
            return round(o, 9)
        if isinstance(o, (np.floating,)):
            return round(float(o), 9)
        return o
    payload = json.dumps(norm(_np(d)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class RunLog:
    def __init__(self, dataset, stage):
        self.dataset = dataset
        self.stage = stage
        config.ensure_dirs()
        self.dir = config.LOG_DIR
        self.path_jsonl = os.path.join(self.dir, "%s__%s%s" % (dataset, stage, JSON_SUFFIX))
        self.path_csv = os.path.join(self.dir, "%s__%s.csv" % (dataset, stage))
        self._seen = self._load_hashes()

    # ── resume ────────────────────────────────────────────────────────────────
    def _load_hashes(self):
        seen = set()
        if not os.path.exists(self.path_jsonl):
            return seen
        with open(self.path_jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("config_hash"):
                    seen.add(rec["config_hash"])
        return seen

    def has(self, chash):
        return chash in self._seen

    # ── write ─────────────────────────────────────────────────────────────────
    def record(self, config_, result_, skip_existing=True, config_hash=None):
        """Append one run. `config_` = hyper-parameters, `result_` = metrics.

        `config_hash` is the run identity the caller already tested with `has()`. It has
        to be passed in: the caller hashes the settings-only block while `config_` here
        has grown the result-derived C/z/K/T/delta fields, and hashing the two different
        dictionaries is what made every session recompute finished work.

        Returns the stored record, or None when the run was already logged.
        """
        chash = config_hash if config_hash is not None else stable_hash(config_)
        if skip_existing and chash in self._seen:
            return None
        rec = {"dataset": self.dataset, "stage": self.stage, "profile": config.PROFILE,
               "reportable": config.is_reportable(), "config_hash": chash,
               "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rec.update(_np(config_))
        rec.update(_np(result_))
        with open(self.path_jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._seen.add(chash)
        self._append_csv(rec)
        return rec

    _csv_header = None

    def _append_csv(self, rec):
        # CSV carries the scalar columns; per-round curves stay in the JSONL.
        flat = {k: v for k, v in rec.items()
                if not isinstance(v, (list, dict, tuple))}
        # The header is fixed by the first row; later rows are written under THAT header
        # (missing keys blank, extra keys left to the JSONL, which is the complete record).
        # Writing each row with its own key set, as before, shifted values into the wrong
        # columns whenever two runs of a stage carried different fields.
        exists = os.path.exists(self.path_csv)
        if not exists:
            os.makedirs(os.path.dirname(self.path_csv), exist_ok=True)
            header = sorted(flat)
        else:
            with open(self.path_csv, newline="") as fh:
                header = next(csv.reader(fh), None) or sorted(flat)
        with open(self.path_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore", restval="")
            if not exists:
                w.writeheader()
            w.writerow(flat)


def read_logs(dataset=None, stage=None, reportable_only=True):
    """Load every logged run as a list of dicts (this is what plotters read)."""
    config.ensure_dirs()
    out = []
    for fname in sorted(os.listdir(config.LOG_DIR)) if os.path.isdir(config.LOG_DIR) else []:
        if not fname.endswith(JSON_SUFFIX):
            continue
        base = fname[: -len(JSON_SUFFIX)]
        ds, _, st = base.partition("__")
        if dataset and ds != dataset:
            continue
        if stage and st != stage:
            continue
        with open(os.path.join(config.LOG_DIR, fname)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if reportable_only and not rec.get("reportable", False):
                    continue
                out.append(rec)
    return out


def summarise(runs, key_fields, metric="accuracy"):
    """Mean/std/n over seeds for each distinct combination of `key_fields`."""
    groups = {}
    for r in runs:
        k = tuple(str(r.get(f, "")) for f in key_fields)
        groups.setdefault(k, []).append(r)
    rows = []
    for k, g in groups.items():
        vals = np.array([float(x[metric]) for x in g if x.get(metric) is not None], float)
        row = dict(zip(key_fields, k))
        row.update(n=int(len(vals)), mean=float(vals.mean()) if len(vals) else None,
                   std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        rows.append(row)
    return sorted(rows, key=lambda d: [str(d[f]) for f in key_fields])


def versions():
    """Software provenance for the run report (Work Order section 13)."""
    import sys
    v = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception:
            v[mod] = None
    for mod in ("dp_accounting", "opacus", "thop"):
        try:
            __import__(mod)
            v[mod] = "installed"
        except Exception:
            v[mod] = None
    return v


def hardware():
    """CPU model / RAM / GPU, recorded once per run report."""
    info = {}
    try:
        import multiprocessing
        info["cpu_count"] = multiprocessing.cpu_count()
    except Exception:
        pass
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal"):
                        info["ram_kb"] = int(line.split()[1])
                        break
    except Exception:
        pass
    try:
        import torch
        info["torch_cuda"] = bool(torch.cuda.is_available())
        if info["torch_cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info
