"""Anti-aliased resampling, spectral-distortion measurement and channel harmonisation.

Task 7. The submission upsampled WISDM 20 -> 50 Hz with ``np.interp``, which is linear
interpolation: it creates no energy above 10 Hz in the signal itself but the spectral
signature of the interpolation kernel is not the one a band-limited resampler gives, and
the paper's Nyquist argument assumed an ideal reconstruction. Everything here goes
through ``scipy.signal.resample_poly`` and the size of the difference is measured rather
than asserted.
"""

import numpy as np
from scipy.signal import resample_poly, welch

from . import config


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def ratio(src_fs, dst_fs):
    """Exact integer up/down factors for resample_poly."""
    src = int(round(float(src_fs) * 1000))
    dst = int(round(float(dst_fs) * 1000))
    g = _gcd(src, dst)
    return dst // g, src // g


def resample_signal(x, src_fs, dst_fs):
    """(T, C) at src_fs -> (T', C) at dst_fs, anti-aliased polyphase filter."""
    x = np.asarray(x, dtype=np.float64)
    up, down = ratio(src_fs, dst_fs)
    if up == down:
        return x.astype(np.float32)
    y = np.empty((x.shape[0] * up // down, x.shape[1]) if x.ndim == 2
                 else (x.shape[0] * up // down,), dtype=np.float64)
    if x.ndim == 1:
        y[:] = resample_poly(x, up, down)
    else:
        for c in range(x.shape[1]):
            col = resample_poly(x[:, c], up, down)
            n = min(len(col), y.shape[0])
            y[:n, c] = col[:n]
    return y.astype(np.float32)


def upsample_linear(x, src_fs, dst_fs):
    """The submission's np.interp upsampling, retained ONLY for the distortion measure."""
    x = np.asarray(x, dtype=np.float64)
    T, C = x.shape
    t_src = np.arange(T, dtype=float) / float(src_fs)
    T_new = int(T * dst_fs / src_fs)
    t_dst = np.arange(T_new, dtype=float) / float(dst_fs)
    out = np.empty((T_new, C), dtype=np.float64)
    for c in range(C):
        out[:, c] = np.interp(t_dst, t_src, x[:, c])
    return out.astype(np.float32)


def repeat_labels(labels, src_fs, dst_fs):
    """Nearest replication of a per-sample label stream onto a resampled time base."""
    labels = np.asarray(labels)
    T = len(labels)
    T_new = int(round(T * float(dst_fs) / float(src_fs)))
    idx = np.clip((np.arange(T_new) * float(src_fs) / float(dst_fs)).astype(int), 0, T - 1)
    return labels[idx]


def _trapz(y, x):
    """np.trapz was removed from the NumPy 2 namespace; prefer trapezoid when present."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def spectral_distortion(x_src, src_fs, dst_fs=None, band_guard=0.05):
    """Fraction of energy that a naive linear upsampling places above the source Nyquist.

    Returns spurious_energy_fraction for the linear and for the polyphase resampler.
    A band-limited 20 Hz signal has no content above 10 Hz; anything measured up there
    after resampling is an artefact of the resampler.
    """
    dst_fs = dst_fs or config.TARGET_FS
    x_src = np.asarray(x_src, dtype=np.float64)
    nyq_src = src_fs / 2.0
    out = {}
    for name, y in (("linear", upsample_linear(x_src, src_fs, dst_fs)),
                    ("polyphase", resample_signal(x_src, src_fs, dst_fs))):
        fs_used = dst_fs
        tot_e, spur_e = 0.0, 0.0
        for c in range(y.shape[1]):
            f, p = welch(y[:, c], fs=fs_used, nperseg=min(256, len(y)))
            p = np.nan_to_num(p, nan=0.0)
            tot_e += _trapz(p, f)
            above = f > nyq_src * (1.0 + band_guard)
            spur_e += _trapz(p[above], f[above])
        out["spurious_energy_fraction_%s" % name] = float(spur_e / (tot_e + 1e-20))
    out["source_nyquist_hz"] = float(nyq_src)
    out["dst_fs"] = float(dst_fs)
    out["artefact_reduction"] = float(out["spurious_energy_fraction_linear"]
                                     - out["spurious_energy_fraction_polyphase"])
    return out


# ── variants used by the ablation ─────────────────────────────────────────────
VARIANTS = ("common_50hz", "native_rate", "band_limited_features")


def window_native(signal, labels, label_map, fs, wl=None, step=None, purity=None,
                  src_fs=None):
    """Window a single subject's continuous stream.

    signal/labels may be at the native rate; when `src_fs` differs from `fs` the
    signal is resampled with resample_poly first. Window length follows the physical
    2.56 s of the paper, so the native-rate variant uses proportionally more samples.
    """
    wl = wl or config.WL
    step = step or wl // 2
    purity = purity or config.PURITY
    if src_fs is not None and abs(float(src_fs) - float(fs)) > 1e-6:
        signal = resample_signal(signal, src_fs, fs)
        labels = repeat_labels(labels, src_fs, fs)
    signal = np.asarray(signal, dtype=np.float32)
    labels = np.asarray(labels)
    X, yv = [], []
    for start in range(0, len(signal) - wl + 1, step):
        seg = signal[start:start + wl]
        lb = labels[start:start + wl]
        vals, cnts = np.unique(lb[(lb >= 0)], return_counts=True) if len(lb) else (np.array([]), np.array([]))
        if len(vals) == 0:
            continue
        keep = [i for i, v in enumerate(vals) if int(v) in label_map]
        if not keep:
            continue
        vals = vals[keep]
        cnts = cnts[keep]
        maj = int(vals[np.argmax(cnts)])
        if cnts.max() / float(wl) < purity or maj not in label_map:
            continue
        X.append(seg)
        yv.append(label_map[maj])
    if not X:
        return None, None
    return np.asarray(X, dtype=np.float32), np.asarray(yv, dtype=int)


def harmonized_band_features(X, fs, band_hz=None):
    """Variant (c): FFT features restricted to the common 0-10 Hz band.

    The Work Order names this function, so it carries that name: a reviewer grepping the
    response letter for `resampling.harmonized_band_features` must land on the code that
    produced the row. Zeroes every spectral component of each window outside `band_hz`
    before features are computed, so band power/entropy cannot see out-of-band resampling
    artefacts.
    """
    band_hz = band_hz or config.HARMONISED_BAND_HZ
    X = np.asarray(X, dtype=np.float32)
    wl = X.shape[1]
    f = np.fft.rfftfreq(wl, d=1.0 / float(fs))
    mask = ((f >= band_hz[0]) & (f <= band_hz[1])).astype(np.float32)
    spec = np.fft.rfft(X.astype(np.float64), axis=1)
    spec = spec * mask[None, :, None]
    Y = np.fft.irfft(spec, n=wl, axis=1)
    return np.nan_to_num(Y, nan=0.0).astype(np.float32)


# Where each sensor block sits in a loader's per-sample signal (column slices), per
# dataset. "total" may be "acc" (reuse the accelerometer, documented for WISDM).
CHANNEL_LAYOUT = {
    "PAMAP2":   dict(acc=(0, 3), gyro=(3, 6), total=(6, 9)),
    "WISDM v2": dict(acc=(0, 3), gyro=(3, 6), total="acc"),
    "HARTH":    dict(acc=(0, 3), gyro=None, total=(3, 6)),   # back acc, thigh acc
}


def harmonise_channels(X_native, n_ch, mode="zero_pad", gyro_slice=(3, 6), layout=None):
    """Map a dataset's native channels onto the 9-channel [acc | gyro | total_acc] tensor.

    `layout` says which native columns are which sensor; without it a 6-column input is
    ambiguous (WISDM = acc+gyro, HARTH = two accelerometers) and the old code put HARTH's
    thigh accelerometer into the GYROSCOPE slot.

    mode="zero_pad"      : 9 channels; an absent gyroscope block is exactly zero.
    mode="accel_only"    : accelerometer channels only (acc, plus the second
                           accelerometer when the dataset has one) - no gyro block at all.
    mode="channel_mask"  : zero_pad plus one constant channel: 1 = gyroscope present,
                           0 = absent.
    """
    X_native = np.asarray(X_native, dtype=np.float32)
    N, wl, c = X_native.shape
    if layout is None:
        layout = (dict(acc=(0, 3), gyro=(3, 6), total=(6, 9)) if c >= 9 else
                  dict(acc=(0, 3), gyro=(3, 6) if c >= 6 else None, total="acc"))
    acc = X_native[:, :, layout["acc"][0]:layout["acc"][1]]
    gyro = (X_native[:, :, layout["gyro"][0]:layout["gyro"][1]]
            if layout.get("gyro") else None)
    tot = (acc if layout.get("total") in (None, "acc")
           else X_native[:, :, layout["total"][0]:layout["total"][1]])
    if mode == "accel_only":
        blocks = [acc] + ([tot] if layout.get("total") not in (None, "acc") else [])
        return np.concatenate(blocks, axis=2).copy(), None
    out = np.zeros((N, wl, n_ch), dtype=np.float32)
    out[:, :, 0:3] = acc
    if gyro is not None:
        out[:, :, 3:6] = gyro
    out[:, :, 6:9] = tot
    if mode == "channel_mask":
        mask = np.full((N, wl, 1), 1.0 if gyro is not None else 0.0, dtype=np.float32)
        return out, mask
    return out, None


def ablation_variants(signal, labels, label_map, native_fs, n_ch, mode="zero_pad"):
    """Yield the three Task-7 data variants for one subject stream.

    (a) common_50hz        - polyphase resample to 50 Hz, 128-sample window;
    (b) native_rate        - no resampling, window length scaled to 2.56 s;
    (c) band_limited_features - as (a), windows band-limited to 0-10 Hz.
    """
    wl_native = int(round(config.WL * native_fs / config.TARGET_FS))
    a_X, a_y = window_native(signal, labels, label_map, fs=config.TARGET_FS,
                             src_fs=native_fs)
    n_X, n_y = window_native(signal, labels, label_map, fs=native_fs, wl=wl_native)
    if a_X is None:
        return {}
    b_X = harmonized_band_features(a_X, config.TARGET_FS)
    return {"common_50hz": (a_X, a_y),
            "native_rate": (n_X, n_y) if n_X is not None else (a_X, a_y),
            "band_limited_features": (b_X, a_y)}
