"""Dataset loading for UCI-HAR, PAMAP2, WISDM v2 and HARTH.

Three things the submission's loaders could not do and the Work Order requires:

1. they return the SUBJECT id of every window, without which neither subject-disjoint
   splitting nor user-level DP is possible (the old loaders concatenated the per-subject
   files and threw the identity away);
2. every channel block is mapped explicitly and reported, instead of the old
   ``[22,23,24,28,29,30,4,5,6]`` selection which put a PAMAP2 *temperature* column and
   an out-of-range index into the accelerometer block;
3. all resampling goes through ``resample_poly`` (see resampling.py).

Sensor mapping actually used
----------------------------
dataset   acc block            gyro block           total_acc block       absent
UCI-HAR   body_acc            body_gyro            total_acc             -
PAMAP2    hand acc +-16 g      hand gyroscope       chest acc +-16 g      -
          (cols 4-6)           (cols 10-12)         (cols 21-23)   [0-based, PAMAP2 readme]
WISDM     phone accel          phone gyroscope      phone accel           -
          (raw/phone/accel)    (raw/phone/gyro, aligned to accel by timestamp)
HARTH     back acc             -                    thigh acc             gyro

Class spaces (config.N_CLASS is set per dataset by the driver):
  UCI-HAR, PAMAP2, HARTH : 6 classes  0 Walking, 1 WalkUp, 2 WalkDn, 3 Sitting,
                                      4 Standing, 5 Laying
  WISDM (2019, UCI #507) : 3 classes  0 Walking (A), 1 Sitting (D), 2 Standing (E).
      The archive has 18 activities (A-S, no N). It has no lying class and does not
      separate stairs up from stairs down, so only these three map onto the shared set.
HARTH keeps 1 walking, 4 stairs up, 5 stairs down, 7 sitting, 6 standing, 8 lying and
drops running (2), shuffling (3) and cycling (13, 14, 130, 140), which ARE in the
archive but have no equivalent in the shared class set.
"""

import os
import re
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from . import config
from .resampling import (CHANNEL_LAYOUT, ablation_variants, harmonized_band_features,
                         harmonise_channels, window_native)

CLASS_NAMES = ["Walking", "Walk Up", "Walk Down", "Sitting", "Standing", "Laying"]
CLASS_NAMES_BY_DATASET = {
    "UCI HAR": CLASS_NAMES, "PAMAP2": CLASS_NAMES, "HARTH": CLASS_NAMES,
    "WISDM v2": ["Walking", "Sitting", "Standing"],
}
# Number of classes each dataset is scored on. The driver writes this into
# PPML_N_CLASS before the package is imported, so config.N_CLASS matches the bundle.
N_CLASS_BY_DATASET = {k: len(v) for k, v in CLASS_NAMES_BY_DATASET.items()}


def class_names(name):
    return CLASS_NAMES_BY_DATASET.get(name, CLASS_NAMES)

LABEL_MAPS = {
    "UCI HAR": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
    "PAMAP2": {4: 0, 12: 1, 13: 2, 2: 3, 3: 4, 1: 5},
    # WISDM 2019 activity_key.txt: A walking, B jogging, C stairs, D sitting,
    # E standing, F typing, G brushing teeth, ... Only A/D/E have a shared-class
    # equivalent; they become a 3-class problem (0 Walking, 1 Sitting, 2 Standing).
    "WISDM v2": {"A": 0, "D": 1, "E": 2},
    # HARTH (UCI #779): 1 walking, 2 running, 3 shuffling, 4 stairs up, 5 stairs down,
    # 6 standing, 7 sitting, 8 lying, 13/14/130/140 cycling. Running, shuffling and
    # cycling have no shared-class equivalent and are dropped (window discarded).
    "HARTH": {1: 0, 4: 1, 5: 2, 7: 3, 6: 4, 8: 5},
}

# 0-based column indices into each raw per-sample record
CHANNEL_MAP = {
    "UCI HAR": dict(acc=(0, 1, 2), gyro=(3, 4, 5), total=(6, 7, 8), n_native=9),
    # PAMAP2 readme, 0-based: 0 time, 1 activity, 2 heart rate, then three 17-column IMU
    # blocks (hand 3-19, chest 20-36, ankle 37-53), each: temp, acc16 x3, acc6 x3,
    # gyro x3, mag x3, orientation x4. So hand acc16 = 4-6, hand gyro = 10-12,
    # chest acc16 = 21-23.
    "PAMAP2": dict(acc=(4, 5, 6), gyro=(10, 11, 12), total=(21, 22, 23), n_native=54),
    "WISDM v2": dict(acc=(0, 1, 2), gyro=(3, 4, 5), total=(0, 1, 2), n_native=6),
    "HARTH": dict(acc=(0, 1, 2), gyro=None, total=(3, 4, 5), n_native=6),
}

NATIVE_FS = {"UCI HAR": 50.0, "PAMAP2": 100.0, "WISDM v2": 20.0, "HARTH": 50.0}

_URLS = {
    # current UCI addresses first; the legacy /ml/machine-learning-databases/ paths are
    # kept only as a fallback
    "UCI HAR": ["https://archive.ics.uci.edu/static/public/240/"
                "human+activity+recognition+using+smartphones.zip",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
                "UCI%20HAR%20Dataset.zip"],
    "PAMAP2": ["https://archive.ics.uci.edu/static/public/231/"
               "pamap2+physical+activity+monitoring.zip",
               "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/"
               "PAMAP2_Dataset.zip"],
    "WISDM v2": ["https://archive.ics.uci.edu/static/public/507/"
                 "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
                 "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/"
                 "wisdm-dataset.zip"],
    "HARTH": ["https://archive.ics.uci.edu/static/public/779/harth.zip",
              "https://zenodo.org/record/5178138/files/harth.zip"],
}


class DatasetBundle:
    """Everything one experiment needs for one dataset."""

    def __init__(self, name, X, y, subject, meta):
        self.name = name
        self.X = X                # (N, WL, 9) harmonised @ 50 Hz
        self.y = y                # (N,) shared class ids
        self.subject = subject    # (N,) original subject id
        self.meta = meta
        self.variants = {}        # filled by load(): Task 7 data variants

    def __len__(self):
        return len(self.y)

    def summary(self):
        cnt = dict(zip(*np.unique(self.y, return_counts=True)))
        return dict(dataset=self.name, n_windows=int(len(self.y)),
                    n_subjects=int(len(np.unique(self.subject))),
                    window_shape=list(self.X.shape),
                    native_fs=NATIVE_FS[self.name],
                    class_counts={int(k): int(v) for k, v in cnt.items()},
                    missing_classes=[c for c in range(config.N_CLASS) if c not in cnt],
                    class_names=list(class_names(self.name)),
                    channel_map={k: v for k, v in CHANNEL_MAP[self.name].items()},
                    label_map={str(k): v for k, v in LABEL_MAPS[self.name].items()})

    def variant(self, name):
        """Task 7 data variants: common_50hz | native_rate | band_limited_features."""
        return self.variants[name]

    def as_variant(self, name):
        """A bundle rebuilt on one Task-7 variant, carrying the sampling rate its
        windows are actually at (native_rate keeps the device clock)."""
        if name not in self.variants:
            raise KeyError("variant %r not loaded (have %s)"
                           % (name, sorted(self.variants)))
        X, y, s = self.variants[name]
        fs = getattr(self, "variants_fs", {}).get(name, config.TARGET_FS)
        b = DatasetBundle(self.name, X, y, s, dict(self.meta))
        b.variants = self.variants
        b.variants_fs = getattr(self, "variants_fs", {})
        b.meta.update(variant=name, fs=float(fs))
        b.meta.update(b.summary())
        b.meta["variant"] = name
        b.meta["fs"] = float(fs)
        return b


def _fetch(name, dest_dir):
    """Find the dataset, in this order, and return the directory its loader reads:

    1. any folder under /kaggle/input whose CONTENTS match the dataset (folder names are
       not trusted: "har" is inside "harth", and Kaggle copies are named freely);
    2. a zip under /kaggle/input, unpacked into dest_dir;
    3. an earlier download in dest_dir;
    4. a fresh download from UCI (Internet ON).
    """
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.isdir("/kaggle/input"):
        root = _signature_root(name, "/kaggle/input")
        if root:
            print("  %s: using attached Kaggle input %s" % (name, root))
            return root
        for dp, dn, fn in os.walk("/kaggle/input"):
            for f in fn:
                if f.lower().endswith(".zip") and _zip_is(name, os.path.join(dp, f)):
                    print("  %s: unpacking attached archive %s" % (name, f))
                    with zipfile.ZipFile(os.path.join(dp, f), "r") as z:
                        z.extractall(dest_dir)
                    _extract_nested_zips(dest_dir)
                    root = _signature_root(name, dest_dir)
                    if root:
                        return root
    root = _signature_root(name, dest_dir)
    if root:
        return root
    for url in _URLS[name]:
        zip_path = os.path.join(dest_dir, name.replace(" ", "_") + ".zip")
        print("  %s: downloading %s" % (name, url))
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest_dir)
            os.remove(zip_path)
            _extract_nested_zips(dest_dir)
            root = _signature_root(name, dest_dir)
            if root:
                return root
            print("  %s: archive unpacked but the expected files were not found" % name)
        except Exception as exc:
            print("  %s: failed (%s)" % (name, exc))
    return None


_HARTH_FILE = re.compile(r"^S\d+\.csv$", re.I)
_PAMAP_FILE = re.compile(r"^subject1\d\d\.dat$", re.I)
_WISDM_ACC = re.compile(r"^data_\d+_accel_phone\.txt$", re.I)


def _signature_root(name, base):
    """The directory under `base` whose files identify `name`, or None.

    UCI HAR : the folder holding train/Inertial Signals (raw signals, not the 561-feature
              CSV copies that circulate on Kaggle - those cannot be used)
    PAMAP2  : the Protocol folder with subject101.dat ... (never Optional/)
    WISDM   : raw/phone/accel with data_16xx_accel_phone.txt
    HARTH   : the folder with S0xx.csv files
    """
    if not os.path.isdir(base):
        return None
    for dp, dn, fn in os.walk(base):
        parts = dp.split(os.sep)
        # macOS archive metadata (__MACOSX/) mirrors the real folders with "._" files;
        # the UCI-HAR zip ships one, and it must never be taken for the data
        dn[:] = [d for d in dn if d != "__MACOSX"]
        if "__MACOSX" in parts:
            continue
        if name == "UCI HAR":
            if os.path.isdir(os.path.join(dp, "train", "Inertial Signals")):
                return dp
        elif name == "PAMAP2":
            if "Optional" in parts:
                continue
            if any(_PAMAP_FILE.match(f) for f in fn):
                return dp
        elif name == "WISDM v2":
            if dp.endswith(os.path.join("phone", "accel")) and "raw" in parts \
                    and any(_WISDM_ACC.match(f) for f in fn):
                return dp
        elif name == "HARTH":
            if sum(1 for f in fn if _HARTH_FILE.match(f)) >= 2:
                return dp
    return None


def _zip_is(name, path):
    """Peek inside an attached zip (without unpacking) to see whether it is `name`."""
    keys = {"UCI HAR": ("inertial signals", "uci har", "human+activity"),
            "PAMAP2": ("pamap2", "subject101"),
            "WISDM v2": ("wisdm",),
            "HARTH": ("harth", "s006.csv")}[name]
    low = os.path.basename(path).lower()
    if any(k in low for k in keys):
        return True
    try:
        with zipfile.ZipFile(path) as z:
            names = " ".join(z.namelist()[:200]).lower()
        return any(k in names for k in keys)
    except zipfile.BadZipFile:
        return False


def _extract_nested_zips(dest_dir, max_depth=3):
    """UCI's /static/public/ archives wrap the original archive in another zip
    (e.g. PAMAP2_Dataset.zip, UCI HAR Dataset.zip, wisdm-dataset.zip). Unpack every
    inner zip in place so the loaders see the real files."""
    for _ in range(max_depth):
        inner = [os.path.join(dp, f) for dp, dn, fn in os.walk(dest_dir) for f in fn
                 if f.lower().endswith(".zip")]
        if not inner:
            return
        for zp in inner:
            try:
                with zipfile.ZipFile(zp, "r") as z:
                    z.extractall(os.path.dirname(zp))
                os.remove(zp)
            except zipfile.BadZipFile:
                print("    skipping unreadable inner archive %s" % zp)
                os.remove(zp)


def _mount_candidates(name):
    out = []
    if not os.path.isdir("/kaggle/input"):
        return out
    keys = {"UCI HAR": ("uci", "har"), "PAMAP2": ("pamap",),
            "WISDM v2": ("wisdm",), "HARTH": ("harth",)}
    for d in sorted(os.listdir("/kaggle/input")):
        low = d.lower()
        if any(k in low for k in keys[name]):
            out.append(os.path.join("/kaggle/input", d))
    return out


def _find_root(name, dest):
    if name == "WISDM v2":
        # The v2 archive carries the same streams twice: parseable .txt under
        # raw/phone/accel and .arff under arff_files/phone/accel. Walking returns
        # whichever the filesystem lists first, so the raw copy is chosen explicitly.
        hits = [r for r, d, f in os.walk(dest)
                if r.endswith(os.path.join("phone", "accel"))]
        for r in hits:
            if "raw" in r.split(os.sep):
                return r
        return hits[0] if hits else dest
    for root, dirs, files in os.walk(dest):
        if name == "UCI HAR" and os.path.isdir(os.path.join(root, "train")):
            return root
        if name == "PAMAP2" and root.endswith("Protocol"):
            return root
        if name == "HARTH" and any(f.endswith(".csv") for f in files):
            return root
    return dest


# ── UCI HAR (pre-windowed) ────────────────────────────────────────────────────
def _load_uci(root):
    base = os.path.join(root, "UCI HAR Dataset")
    if not os.path.isdir(base):
        base = root
    sig_ch = ["body_acc_x", "body_acc_y", "body_acc_z",
              "body_gyro_x", "body_gyro_y", "body_gyro_z",
              "total_acc_x", "total_acc_y", "total_acc_z"]
    Xs, ys, ss = [], [], []
    for split in ("train", "test"):
        sig = os.path.join(base, split, "Inertial Signals")
        try:
            arrs = [np.loadtxt(os.path.join(sig, "%s_%s.txt" % (c, split))) for c in sig_ch]
        except (OSError, ValueError):
            return None
        Xs.append(np.stack(arrs, axis=2).astype(np.float32))
        ys.append(np.loadtxt(os.path.join(base, split, "y_%s.txt" % split), dtype=int) - 1)
        ss.append(np.loadtxt(os.path.join(base, split, "subject_%s.txt" % split), dtype=int))
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(int)
    s = np.concatenate(ss).astype(int)
    return X, y, s


# ── PAMAP2 ────────────────────────────────────────────────────────────────────
def _load_pamap2(root):
    # PAMAP2 has 9 subjects (subject101-109). Only Protocol/ is read: Optional/ holds the
    # same subjects' optional activities, none of which map to the shared classes.
    prot = _signature_root("PAMAP2", root) or root
    files = sorted(os.path.join(dp, f) for dp, d, fn in os.walk(prot) for f in fn
                   if _PAMAP_FILE.match(f) and "Optional" not in dp.split(os.sep))
    if not files:
        print("    PAMAP2: no .dat streams under %s" % prot)
    recs = []
    cm = CHANNEL_MAP["PAMAP2"]
    for fp in files:
        fn = os.path.basename(fp)
        subj = _subject_id(fn)
        try:
            df = pd.read_csv(fp, sep=r"\s+", header=None, dtype=float)
        except Exception as exc:
            print("    %s: skip (%s)" % (fn, exc))
            continue
        if df.shape[1] < cm["n_native"]:
            print("    %s: only %d cols, expected %d - skip"
                  % (fn, df.shape[1], cm["n_native"]))
            continue
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(0.0, inplace=True)
        act = df.iloc[:, 1].values.astype(int)
        blocks = [df.iloc[:, list(cm[k])].values.astype(np.float32)
                  for k in ("acc", "gyro", "total")]
        sig = np.concatenate(blocks, axis=1)
        sig = np.clip(sig, -config.CLIP_RANGE, config.CLIP_RANGE)
        recs.append((subj, sig, act))
    # Expected: 9 subjects. Which ids were actually read is printed so a partial
    # archive cannot pass unnoticed.
    print("    PAMAP2: %d .dat under %s -> %d subjects %s"
          % (len(files), prot, len(set(r[0] for r in recs)),
             sorted(set(r[0] for r in recs))))
    return recs


# ── WISDM v2 ──────────────────────────────────────────────────────────────────
def _read_wisdm_stream(path):
    """One raw WISDM file -> DataFrame(act, t, x, y, z) for the mapped activities only.
    Line format: subject,activity,timestamp,x,y,z;"""
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            parts = line.strip().replace(";", "").split(",")
            if len(parts) < 6:
                continue
            act = parts[1].strip()
            if act not in LABEL_MAPS["WISDM v2"]:
                continue
            try:
                rows.append((act, int(float(parts[2])), float(parts[3]),
                             float(parts[4]), float(parts[5])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["act", "t", "x", "y", "z"])


# filled by _load_wisdm so the driver can print how well accel and gyro lined up
WISDM_ALIGNMENT = {}


def _load_wisdm(root):
    """Phone accelerometer + phone gyroscope, aligned by timestamp.

    The two sensors are separate files with their own clocks, so rows are NOT matched by
    position: each accelerometer sample takes the nearest gyroscope sample of the same
    activity within half a sampling period (pd.merge_asof). Accelerometer samples with no
    gyroscope partner are dropped, and the matched fraction per subject is recorded in
    WISDM_ALIGNMENT and printed by the driver.
    """
    accel = _signature_root("WISDM v2", root) or root
    gyro = os.path.join(os.path.dirname(accel), "gyro")
    if not os.path.isdir(accel):
        return None
    has_gyro = os.path.isdir(gyro)
    files = sorted(f for f in os.listdir(accel) if _WISDM_ACC.match(f))
    recs = []
    WISDM_ALIGNMENT.clear()
    for fn in files:
        subj = _subject_id(fn)
        acc = _read_wisdm_stream(os.path.join(accel, fn))
        if len(acc) < config.WL:
            continue
        gfile = os.path.join(gyro, fn.replace("_accel_", "_gyro_")) if has_gyro else None
        if gfile and os.path.exists(gfile):
            gyr = _read_wisdm_stream(gfile).rename(columns={"x": "gx", "y": "gy", "z": "gz"})
        else:
            gyr = pd.DataFrame(columns=["act", "t", "gx", "gy", "gz"])
        parts = []
        n_acc, n_match = 0, 0
        for act, a_df in acc.groupby("act", sort=False):
            a_df = a_df.sort_values("t")
            g_df = gyr[gyr["act"] == act].sort_values("t")
            n_acc += len(a_df)
            if len(g_df) == 0:
                continue
            dt = np.median(np.diff(a_df["t"].values)) if len(a_df) > 1 else 0
            tol = max(1, int(abs(dt) * 0.5))
            m = pd.merge_asof(a_df, g_df[["t", "gx", "gy", "gz"]], on="t",
                              direction="nearest", tolerance=tol)
            m = m.dropna(subset=["gx", "gy", "gz"])
            n_match += len(m)
            parts.append(m)
        WISDM_ALIGNMENT[subj] = dict(accel_samples=n_acc, matched=n_match,
                                     matched_fraction=(n_match / n_acc) if n_acc else 0.0)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        if len(df) < config.WL:
            continue
        sig = df[["x", "y", "z", "gx", "gy", "gz"]].values.astype(np.float32)
        sig = np.clip(sig, -config.CLIP_RANGE, config.CLIP_RANGE)
        codes = np.array([LABEL_MAPS["WISDM v2"][a] for a in df["act"].values], dtype=int)
        recs.append((subj, sig, codes))
    if WISDM_ALIGNMENT:
        fr = [v["matched_fraction"] for v in WISDM_ALIGNMENT.values()]
        print("    WISDM accel/gyro alignment: %d subjects, matched fraction "
              "min %.2f / median %.2f" % (len(fr), min(fr), float(np.median(fr))))
    return recs


# ── HARTH ─────────────────────────────────────────────────────────────────────
def _subject_id(name):
    """The FIRST digit run of a file name is the subject; later runs are activity,
    trial or file numbers.

    Concatenating every digit in the name (the old behaviour) turned HARTH's
    `S01_A02.csv` / `S01_A03.csv` into two different "subjects", so one person's windows
    were split across two silos. That is not a cosmetic bug: the subject-disjoint split,
    the K client silos and the claim that a client-level certificate is also a user-level
    one all rest on "one silo = one person".
    """
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else 0


_HARTH_COLS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]


def _load_harth(root):
    """HARTH S0xx.csv files, sensor columns read BY NAME (some files carry extra
    index columns, which shifts every positional index)."""
    root = _signature_root("HARTH", root) or root
    csvs = sorted(os.path.join(dp, f) for dp, dn, fn in os.walk(root) for f in fn
                  if _HARTH_FILE.match(f))
    recs = []
    for fp in csvs:
        subj = _subject_id(os.path.basename(fp))
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            print("    %s: skip (%s)" % (os.path.basename(fp), exc))
            continue
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not all(c in df.columns for c in _HARTH_COLS + ["label"]):
            print("    %s: columns %s do not include %s - skip"
                  % (os.path.basename(fp), list(df.columns), _HARTH_COLS + ["label"]))
            continue
        sig = df[_HARTH_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        lbl = pd.to_numeric(df["label"], errors="coerce").values
        sig = np.clip(np.nan_to_num(sig, nan=0.0), -config.CLIP_RANGE, config.CLIP_RANGE)
        codes = np.array([LABEL_MAPS["HARTH"].get(int(v), -1)
                          for v in np.nan_to_num(lbl, nan=-1).astype(int)], dtype=int)
        if len(sig) < config.WL:
            continue
        recs.append((subj, sig, codes))
    print("    HARTH: %d files -> %d subjects" % (len(csvs), len(set(r[0] for r in recs))))
    return recs


# ── build bundle ──────────────────────────────────────────────────────────────
def _cap_windows(X, y, s, cap, seed):
    """Deterministic stratified subsample: every subject keeps a share of `cap`.

    A flat `X[:cap]` would silently delete the later subjects, and the subject pool is
    what the K silos, the LOSO folds and the three MIA pools are built from. Budget per
    subject is cap/n_subjects, so a dataset with more subjects loses proportionally more
    windows but never loses a person.
    """
    if not cap or len(X) <= cap:
        return X, y, s
    subjects = np.unique(s)
    per = max(1, int(cap) // max(1, len(subjects)))
    rng = np.random.RandomState(int(seed))
    keep = []
    for subj in subjects:
        idx = np.flatnonzero(s == subj)
        if len(idx) > per:
            idx = np.sort(rng.choice(idx, size=per, replace=False))
        keep.append(idx)
    k = np.concatenate(keep)
    return X[k], y[k], s[k]


def load(name, max_subjects=None, channel_mode="zero_pad", seed=config.SEEDS[0],
         max_windows=None):
    """Return a DatasetBundle for `name` in one of
    {'UCI HAR', 'PAMAP2', 'WISDM v2', 'HARTH'}."""
    if name not in _URLS:
        raise ValueError("unknown dataset %r" % name)
    set_class_space(name)
    max_windows = config.MAX_WINDOWS if max_windows is None else max_windows
    root = _fetch(name, os.path.join(config.DATA_DIR, name.replace(" ", "_")))
    if root is None:
        raise RuntimeError("could not obtain %s" % name)
    variants = {}
    if name == "UCI HAR":
        got = _load_uci(root)
        if got is None:
            raise RuntimeError("UCI HAR layout not recognised in %s" % root)
        X, y, s = got
        if max_subjects:
            keep = np.unique(s)[:int(max_subjects)]
            m = np.isin(s, keep)
            X, y, s = X[m], y[m], s[m]
        # UCI ships 128-sample windows at 50 Hz with the 9 harmonised blocks already
        variants["common_50hz"] = (X, y, s)
        variants["native_rate"] = (X, y, s)
        variants["band_limited_features"] = (harmonized_band_features(X, config.TARGET_FS), y, s)
    else:
        recs = {"PAMAP2": _load_pamap2, "WISDM v2": _load_wisdm,
                "HARTH": _load_harth}[name](root)
        if not recs:
            raise RuntimeError("no usable %s files under %s" % (name, root))
        if max_subjects:
            recs = sorted(recs, key=lambda r: r[0])[:int(max_subjects)]
        fs = NATIVE_FS[name]
        lmap = LABEL_MAPS[name]
        identity_map = {k: k for k in range(config.N_CLASS)}
        per_variant = {k: [] for k in ("common_50hz", "native_rate",
                                       "band_limited_features")}
        for subj, sig, codes in recs:
            lab = (np.array([lmap.get(int(c), -1) for c in codes], dtype=int)
                   if name == "PAMAP2" else np.asarray(codes, dtype=int))
            vs = ablation_variants(sig, lab, identity_map, fs, sig.shape[1])
            if "common_50hz" not in vs:
                continue
            for k in per_variant:
                if k in vs:
                    Xv, yv = vs[k]
                    per_variant[k].append((subj, Xv, yv))
        for k, lst in per_variant.items():
            if not lst:
                continue
            X9s, y9s, s9s = [], [], []
            for subj, Xv, yv in lst:
                X9, msk = harmonise_channels(Xv, config.N_CH_HARMONISED, mode=channel_mode,
                                             layout=CHANNEL_LAYOUT.get(name))
                if msk is not None:
                    # keep the mask as an extra input channel: the network is TOLD which
                    # sensor block is absent instead of inferring it from constant zeros
                    X9 = np.concatenate([X9, msk], axis=2)
                X9s.append(X9)
                y9s.append(yv)
                s9s.append(np.full(len(yv), subj, dtype=int))
            variants[k] = (np.concatenate(X9s), np.concatenate(y9s),
                           np.concatenate(s9s))
        if "common_50hz" not in variants:
            raise RuntimeError("%s produced no windows" % name)
        X, y, s = variants["common_50hz"]
    # Cap every variant with the same seed: Task 7/8 swap one variant in for another and
    # an uncapped `band_limited_features` would make those arms cost different amounts of
    # data from the arm they are compared against.
    if max_windows:
        variants = {k: _cap_windows(vx, vy, vs, max_windows, seed)
                    for k, (vx, vy, vs) in variants.items()}
        X, y, s = variants["common_50hz"]
    b = DatasetBundle(name, X, y, s, {})
    b.variants_fs = {k: (NATIVE_FS[name] if k == "native_rate" else config.TARGET_FS)
                     for k in variants}
    b.meta.update({"dataset": name, "native_fs": NATIVE_FS[name],
                   "channel_mode": channel_mode, "variant": "common_50hz",
                   "fs": float(config.TARGET_FS)})
    b.variants = variants
    b.meta.update(b.summary())
    b.meta["variant"] = "common_50hz"
    b.meta["fs"] = float(config.TARGET_FS)
    return b


ALL_DATASETS = ("UCI HAR", "PAMAP2", "WISDM v2", "HARTH")
SLUG = {"UCI HAR": "uci_har", "PAMAP2": "pamap2", "WISDM v2": "wisdm_v2", "HARTH": "harth"}

_LOADERS = {"PAMAP2": _load_pamap2, "WISDM v2": _load_wisdm, "HARTH": _load_harth}


def raw_recs(name, max_subjects=None):
    """Per-subject NATIVE-rate streams: [(subject_id, (T, C) signal, per-sample codes)].

    Task 7 needs the un-resampled signal to measure what a resampler actually does;
    `load()` only returns finished windows.
    """
    if name not in _LOADERS:
        raise ValueError("raw_recs is for the resampled datasets (%s), not %r"
                         % (", ".join(sorted(_LOADERS)), name))
    root = _fetch(name, os.path.join(config.DATA_DIR, name.replace(" ", "_")))
    if root is None:
        raise RuntimeError("could not obtain %s" % name)
    recs = _LOADERS[name](root)
    if max_subjects:
        recs = sorted(recs, key=lambda r: r[0])[:int(max_subjects)]
    return recs


def set_class_space(name):
    """Point config.N_CLASS at this dataset's class count (WISDM is 3, the rest 6).

    Every module reads config.N_CLASS at call time, so a session that loads WISDM after
    UCI HAR re-scores against the right label set and the right chance line."""
    config.N_CLASS = int(N_CLASS_BY_DATASET.get(name, len(CLASS_NAMES)))
    config.CLASS_NAMES = list(class_names(name))
    return config.N_CLASS


def label_map_table():
    """Task 7 deliverable: the activity-label mapping across the four datasets.
    Rows are activity NAMES, so WISDM's 3-class indices line up with the others."""
    rows = []
    for cls_name in CLASS_NAMES:
        row = dict(shared_name=cls_name)
        for ds, m in LABEL_MAPS.items():
            names = class_names(ds)
            if cls_name not in names:
                row["%s source codes" % ds] = "- (class not in this dataset)"
                continue
            idx = names.index(cls_name)
            src = [str(k) for k, v in sorted(m.items(), key=lambda kv: str(kv[0]))
                   if v == idx]
            row["%s source codes" % ds] = "+".join(src) if src else "-"
        rows.append(row)
    # one row per dataset naming the native labels that are NOT mapped anywhere, so the
    # dropped-window decision is on the record instead of buried in the loader
    for ds in LABEL_MAPS:
        dropped = sorted(x for x in _NATIVE_LABELS[ds] if x not in LABEL_MAPS[ds])
        row = dict(shared_name="DROPPED (no shared-class equivalent)")
        for ds2 in LABEL_MAPS:
            row["%s source codes" % ds2] = (",".join(str(x) for x in dropped)
                                            if ds2 == ds else "-")
        rows.append(row)
    return rows


# Native label codes each loader can see, used only to report what was dropped.
_NATIVE_LABELS = {
    "UCI HAR": set(range(1, 7)),
    "PAMAP2": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    "WISDM v2": set("ABCDEFGHIJKLMOPQRS"),   # 18 activities, no N
    "HARTH": set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                  20, 21, 22, 23, 24, 130, 140]),
}
