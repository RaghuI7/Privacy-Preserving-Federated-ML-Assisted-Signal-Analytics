"""117-dimensional hand-crafted feature extractor and the nine feature families.

Identical mathematics to the submission's ``extract_features``, with two changes the
Work Order requires:

* the sampling rate is an explicit argument instead of a hidden global, so WISDM
  (20 Hz) and PAMAP2 (100 Hz) are not analysed as if they were 50 Hz signals;
* ``jerk`` gains the sampling-rate factor, matching the derivative it claims to be.
  Which convention was used is written into every run record (Task 8).

Feature layout for ``nc`` channels (nc = 9 gives exactly 117 columns):

  block                     columns            count
  A statistical moments     0 .. 53              54   (6 per channel)
  B cross-axis correlation  54 .. 59              6
  C signal-magnitude area   60                    1
  D jerk                    61 .. 62              2
  E FFT band power          63 .. 107            45   (5 per channel)
  F gravity magnitude       108 .. 109            2
  G tilt                    110                   1
  H step lag                111 .. 113            3
  I spectral entropy        114 .. 116            3
"""

import numpy as np

from . import config

FAMILY_ORDER = ("statistical_moments", "cross_axis_correlation", "sma", "jerk",
                "fft_band_power", "gravity_magnitude", "tilt", "step_lag",
                "spectral_entropy")


def _dominant_lag(sig, fs):
    ac = np.correlate(sig - sig.mean(), sig - sig.mean(), mode="full")
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] + 1e-10)
    for lag in range(5, min(60, len(ac) - 1)):
        if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1] and ac[lag] > 0.1:
            return float(lag) / fs
    return 0.0


def _spectral_entropy(sig):
    psd = np.abs(np.fft.rfft(sig)) ** 2
    pk = psd / (psd.sum() + 1e-10)
    return float(-np.sum(pk * np.log(pk + 1e-10)))


def n_features(nc=None):
    nc = nc if nc is not None else config.N_CH_HARMONISED
    return nc * 6 + config.N_CORR + 1 + 2 + nc * config.N_FFT_BANDS + 2 + 1 + 3 + 3


def family_slices(nc=None):
    """Column index ranges of the nine families, derived not hard-coded.

    `nc` is the NUMBER OF HARMONISED CHANNELS (config.N_CH_HARMONISED), not the number
    of feature columns - the families partition the whole 117-column vector.
    """
    nc = nc if nc is not None else config.N_CH_HARMONISED
    p = 0
    out = {}
    def take(name, k):
        nonlocal p
        out[name] = list(range(p, p + k))
        p += k
    take("statistical_moments", nc * 6)
    take("cross_axis_correlation", config.N_CORR)
    take("sma", 1)
    take("jerk", 2)
    take("fft_band_power", nc * config.N_FFT_BANDS)
    take("gravity_magnitude", 2)
    take("tilt", 1)
    take("step_lag", 3)
    take("spectral_entropy", 3)
    if p != n_features(nc):
        raise ValueError("the nine families cover %d columns but the extractor emits "
                         "%d - the two layouts drifted apart" % (p, n_features(nc)))
    return out


def feature_names(nc=None):
    nc = nc if nc is not None else config.N_CH_HARMONISED
    names = (["Ch%d_%s" % (c + 1, s) for c in range(nc)
              for s in ("mean", "std", "min", "max", "nrg", "zcr")]
             + ["Xcorr_%s" % p for p in ("01", "02", "12", "34", "35", "45")]
             + ["SMA", "Jerk_mean", "Jerk_std"]
             + ["FFT_Ch%d_B%d" % (c + 1, b + 1) for c in range(nc)
                for b in range(config.N_FFT_BANDS)]
             + ["GravMag_mean", "GravMag_std", "Tilt_mean"]
             + ["StepLag_Ch%d" % (c + 1) for c in range(3)]
             + ["SpEnt_Gyro%d" % (c + 1) for c in range(3)])
    assert len(names) == n_features(nc), (len(names), n_features(nc))
    return names


def extract_features(X, fs=None, bands_hz=None, jerk_with_fs=None):
    """X: (N, window, C) harmonised windows -> (N, 117) float array."""
    fs = float(config.TARGET_FS if fs is None else fs)
    bands_hz = bands_hz or config.FFT_BANDS_HZ
    jerk_with_fs = (config.JERK_WITH_SAMPLING_RATE if jerk_with_fs is None
                    else jerk_with_fs)
    N, wl, nc = X.shape
    F = np.zeros((N, n_features(nc)), dtype=np.float64)

    # A: statistical moments, 6 per channel
    for c in range(nc):
        sig = X[:, :, c]
        b = c * 6
        F[:, b + 0] = sig.mean(1)
        F[:, b + 1] = sig.std(1) + 1e-10
        F[:, b + 2] = sig.min(1)
        F[:, b + 3] = sig.max(1)
        F[:, b + 4] = (sig ** 2).mean(1)
        F[:, b + 5] = (np.diff(np.sign(sig), axis=1) != 0).mean(1)

    # B: cross-axis correlation over the three sensor blocks
    p = nc * 6
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    for j, (a, b_) in enumerate([(a, b_) for a, b_ in pairs if a < nc and b_ < nc]):
        ca = X[:, :, a] - X[:, :, a].mean(1, keepdims=True)
        cb = X[:, :, b_] - X[:, :, b_].mean(1, keepdims=True)
        F[:, p + j] = np.nan_to_num(
            (ca * cb).sum(1) / (np.sqrt((ca ** 2).sum(1) * (cb ** 2).sum(1)) + 1e-10))
    p += config.N_CORR

    # C: signal-magnitude area over the accelerometer block
    F[:, p] = (np.abs(X[:, :, 0]) + np.abs(X[:, :, 1]) + np.abs(X[:, :, 2])).mean(1)
    p += 1

    # D: jerk. The magnitude is taken across the 3 accelerometer axes of the first
    # difference; the sampling-rate factor turns it into a true time derivative.
    ad = np.diff(X[:, :, :3], axis=1)
    jm = np.sqrt((ad ** 2).sum(2))
    scale = fs if jerk_with_fs else 1.0
    F[:, p + 0] = jm.mean(1) * scale
    F[:, p + 1] = jm.std(1) * scale
    p += 2

    # E: FFT band power, per channel, over the given band edges
    fb = np.fft.rfftfreq(wl, d=1.0 / fs)
    for c in range(nc):
        fm = np.abs(np.fft.rfft(X[:, :, c], axis=1)) ** 2
        for bi, (fl, fh) in enumerate(bands_hz):
            mk = (fb >= fl) & (fb < fh)
            F[:, p + c * len(bands_hz) + bi] = fm[:, mk].mean(1) if mk.any() else 0.0
    p += nc * len(bands_hz)

    # F: gravity magnitude
    gm = np.sqrt((X[:, :, :3] ** 2).sum(2))
    F[:, p + 0] = gm.mean(1)
    F[:, p + 1] = gm.std(1)
    p += 2

    # G: tilt
    F[:, p] = np.arctan2(X[:, :, 1], X[:, :, 2] + 1e-10).mean(1)
    p += 1

    # H: step lag (dominant autocorrelation lag of each accelerometer axis)
    for ci in range(min(3, nc)):
        F[:, p + ci] = [_dominant_lag(X[i, :, ci], fs) for i in range(N)]
    p += 3

    # I: spectral entropy of the gyroscope block
    for ci in range(3, min(6, nc)):
        F[:, p + (ci - 3)] = [_spectral_entropy(X[i, :, ci]) for i in range(N)]
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def redundancy(F, thresh=0.9):
    """Task 8: within- and across-family mean |r| and count of |r| > thresh pairs."""
    fam_of = {}
    for fam, idx in family_slices().items():
        for i in idx:
            fam_of[i] = fam
    ok = F.std(0) > 1e-12
    C = np.corrcoef(F[:, ok].T) if ok.sum() > 1 else np.zeros((0, 0))
    cols = np.where(ok)[0]
    within, across = [], []
    n_hi = 0
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            r = abs(float(C[a, b])) if np.isfinite(C[a, b]) else 0.0
            (within if fam_of[cols[a]] == fam_of[cols[b]] else across).append(r)
            if r > thresh:
                n_hi += 1
    return dict(n_features=int(len(cols)),
                mean_abs_r_within=float(np.mean(within)) if within else 0.0,
                mean_abs_r_across=float(np.mean(across)) if across else 0.0,
                n_pairs_above=float(n_hi), pair_thresh=thresh,
                n_families=len(FAMILY_ORDER),
                family_names=list(FAMILY_ORDER))
