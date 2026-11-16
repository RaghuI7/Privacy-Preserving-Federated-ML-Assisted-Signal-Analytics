"""Federated aggregation, the corrected DP mechanism, adaptive clipping and Byzantine attacks.

Task 1 in full. The single rule that fixes the submission: clip each client's WHOLE
update with ONE L2 norm over ALL tensors, average with EQUAL weights 1/K, then add ONE
Gaussian draw of std sigma = z*C/K at the server. That construction has L2 sensitivity
C/K, so the noise multiplier really is z and the certificate from dp_accountant applies.
The same function is used for MLP, CNN-BiLSTM and LightTCN.

Optional second rule (`sparsity`, `head_tensors`): release only part of the averaged
update. At a fixed noise multiplier the per-coordinate signal is C/sqrt(d) while the noise
is z*C/K, so everything the model can learn from is governed by K/(z*sqrt(d)) - the client
count over the noise multiplier times the SQUARE ROOT OF THE MODEL DIMENSION. That is the
whole story of why the submission's DP arms sit near chance at d = 71,750 and it also says
where the only real levers are: more clients (K), a smaller released vector (d), or less
noise (z, which is the budget). Sparsifying the support leaves sqrt(d) alone and only trades
noise energy for update coverage, so its net effect is measured, not promised; restricting
the release to the classifier head shrinks d itself and is the change with a derivation
behind it. Both are functions of an already-released or architecturally-fixed object, so
neither can move epsilon. See `dp_aggregate` and `projection_selftest`.
"""

import numpy as np

from . import config


# ── primitives ────────────────────────────────────────────────────────────────
def l2_norm(tensors):
    return float(np.sqrt(sum(float(np.sum(np.asarray(t) ** 2)) for t in tensors)))


def clip_delta(delta, C):
    """Whole-update clipping: ONE norm over all tensors, one scale for all tensors."""
    n = l2_norm(delta)
    f = min(1.0, float(C) / (n + 1e-12))
    return [np.asarray(d) * f for d in delta], n, (1.0 if n > C else 0.0)


def weighted_mean(client_ws, client_ns):
    tot = float(sum(client_ns))
    out = [np.zeros_like(w) for w in client_ws[0]]
    for w, n_k in zip(client_ws, client_ns):
        for i, o in enumerate(out):
            out[i] = out[i] + np.asarray(w[i]) * (float(n_k) / tot)
    return out


def equal_mean(client_ws):
    K = float(len(client_ws))
    out = [np.zeros_like(w) for w in client_ws[0]]
    for w in client_ws:
        for i, o in enumerate(out):
            out[i] = out[i] + np.asarray(w[i]) / K
    return out


def top_k_projection(vec, keep_fraction, energy_from=None):
    """Boolean masks marking the `keep_fraction` largest-|value| coordinates of `vec`.

    `vec` must be an object the mechanism has ALREADY released - here, the noisy average of
    this round - never the clean average. Choosing the support from the clean average is
    what DP-Forward does, and it is a data-dependent selection whose worst-case effect on
    sensitivity is not bounded by C/K on its own: two averages that differ by delta can
    have supports that differ by a whole near-threshold coordinate. This pipeline certifies
    only what it can prove, so the support comes from a released object and the ranking of
    candidates is post-processing: epsilon is exactly the value dp_accountant returns for
    (z, T, delta), with no lemma and no assumption about how concentrated the update is.

    `energy_from` is the clean average, used only to report `signal_energy_fraction` - the
    honest cost of the trick: the coordinates not updated this round keep their old value,
    so whatever the round agreed on there is not released.
    """
    frac = float(keep_fraction)
    if frac >= 1.0:
        return None, dict(sparsity=1.0, noise_coords=None, total_coords=None,
                          signal_energy_fraction=1.0)
    flat = np.concatenate([np.asarray(m, dtype=np.float64).ravel() for m in vec])
    d = int(flat.size)
    m_keep = max(1, int(round(frac * d)))
    mag = np.abs(flat)
    if m_keep >= d:
        return None, dict(sparsity=1.0, noise_coords=d, total_coords=d,
                          signal_energy_fraction=1.0)
    idx = np.argpartition(mag, d - m_keep)[d - m_keep:]
    thresh = float(mag[idx].min())
    masks = [np.abs(np.asarray(m, dtype=np.float64)) >= thresh for m in vec]
    n_kept = int(sum(int(np.count_nonzero(k)) for k in masks))
    ref = [np.asarray(m, dtype=np.float64) for m in (energy_from or vec)]
    total_e = float(sum(float(np.sum(r ** 2)) for r in ref))
    kept_e = float(sum(float(np.sum(r[k] ** 2)) for r, k in zip(ref, masks)))
    return masks, dict(sparsity=frac, noise_coords=n_kept, total_coords=d,
                       signal_energy_fraction=(kept_e / total_e if total_e > 0 else 1.0))


def dp_aggregate(global_w, client_ws, C, z, rng, return_diag=False, sparsity=None,
                 head_tensors=None):
    """The reference mechanism (Work Order Task 1), with an optional release support.

    Returns the new global weights. Note there is deliberately no client-count weighting
    here: sensitivity C/K is only true for equal weights 1/K.

    The mechanism drawn in full every time is the plain one: the clipped average plus one
    Gaussian draw of std sigma = z*C/K on all d coordinates. That vector is what the
    certificate prices, and both options below are functions of it - never of the clean
    average - which is why neither can move epsilon.

    sparsity=None or 1.0  -> every coordinate is updated (the mechanism the rejected paper
                             ran, kept intact so its claim is tested as published).
    sparsity=0.03..0.5    -> only the `sparsity` fraction of coordinates with the largest
                             RELEASED magnitude is applied; the rest keep their previous
                             value. The support is a function of noise that is already in
                             the release, so it is post-processing and epsilon is identical.
                             What it buys (less random walk in most of the model) against
                             what it costs (only part of the round's update lands) is a
                             measurement, not a derivation - Tasks 1, 2 and 4 report it.
    head_tensors=H        -> the last H tensors, the classifier head, are the only
                             coordinates released; the feature extractor stays at its
                             public initialisation. This is the one option that moves the
                             signal-to-noise law itself: the update the model can receive is
                             spread over d_head coordinates instead of d, and the ratio the
                             mechanism has to beat is K/(z*sqrt(d)). The support is fixed by
                             the architecture rather than the data, so the mask is a 0/1
                             diagonal map that cannot increase an L2 distance: sensitivity
                             stays <= C/K and the certificate is exactly the same one.
    """
    deltas = []
    norms = []
    clipped_flags = []
    for w in client_ws:
        d = [np.asarray(wi, dtype=np.float64) - np.asarray(gi, dtype=np.float64)
             for wi, gi in zip(w, global_w)]
        cd, n, was = clip_delta(d, C)
        deltas.append(cd)
        norms.append(n)
        clipped_flags.append(was)
    K = len(deltas)
    if K == 0:
        return (global_w, None) if return_diag else global_w
    mean = [np.sum([d[i] for d in deltas], axis=0) / float(K) for i in range(len(global_w))]
    sigma = float(z) * float(C) / float(K)
    # the plain mechanism first, in full: this vector is what the certificate is for
    noisy_all = [np.asarray(m, dtype=np.float64) + rng.normal(0.0, sigma, m.shape)
                 for m in mean]
    ht = 0 if head_tensors is None else int(head_tensors)
    sp = 1.0 if sparsity is None else float(sparsity)
    if 0 < ht < len(mean):
        keep = [i >= len(mean) - ht for i in range(len(mean))]
        noisy = [n if k else np.zeros_like(n) for n, k in zip(noisy_all, keep)]
        total_e = float(sum(float(np.sum(np.asarray(m, dtype=np.float64) ** 2))
                            for m in mean))
        head_e = float(sum(float(np.sum(np.asarray(m, dtype=np.float64) ** 2))
                           for m, k in zip(mean, keep) if k))
        proj_diag = dict(sparsity=None, support="head",
                         noise_coords=int(sum(int(np.size(m))
                                              for m, k in zip(mean, keep) if k)),
                         total_coords=int(sum(int(np.size(m)) for m in mean)),
                         signal_energy_fraction=(head_e / total_e if total_e > 0 else 1.0))
    elif sp >= 1.0:
        noisy = noisy_all
        proj_diag = dict(sparsity=1.0, support="all", noise_coords=None,
                         total_coords=None, signal_energy_fraction=1.0)
    else:
        # the support is a function of the released noisy vector, never of `mean`
        masks, proj_diag = top_k_projection(noisy_all, sp, energy_from=mean)
        proj_diag["support"] = "released-noise"
        noisy = noisy_all if masks is None else [n * k for n, k in zip(noisy_all, masks)]
    new_w = [np.asarray(g + n, dtype=np.float32) for g, n in zip(global_w, noisy)]
    if not return_diag:
        return new_w
    diag = dict(K=K, sigma=sigma, update_norms=norms,
                clip_fraction=float(np.mean(clipped_flags)), clip_C=float(C), z=float(z))
    # the realised signal-to-noise of this one release, measured from the vectors actually
    # drawn rather than predicted from d: ||noise|| here is the empirical L2 of the Gaussian
    # over the coordinates that were really released, so the head-only and sparse supports
    # report their own ratio and not the whole-model one. This is the number that says
    # whether a chance-level DP row is the mechanism's fault or the model's.
    noise_vecs = [np.asarray(n, dtype=np.float64) - np.asarray(m, dtype=np.float64)
                  for n, m in zip(noisy, mean)]
    signal_norm = l2_norm(mean)
    noise_norm = l2_norm(noise_vecs)
    diag["release_signal_norm"] = float(signal_norm)
    diag["release_noise_norm"] = float(noise_norm)
    diag["release_snr"] = (float(signal_norm / noise_norm)
                           if noise_norm > 0 else float("inf"))
    diag.update(proj_diag)
    return new_w, diag


def projection_selftest(verbose=True):
    """Preflight assertions for the two alternative release supports - proof or stop.

    The claim being tested is narrow and load-bearing: releasing only part of the averaged
    update is worth the SAME epsilon as releasing all of it. The construction makes that a
    post-processing statement instead of a lemma - the noise is drawn in full exactly as in
    the plain mechanism, and the coordinates kept are either selected from the noisy vector
    itself (sparsity) or fixed by the architecture (head-only) - so the assertions below
    check that the code really does that. If any of them fails, `dp_sparsity` or
    `dp_head_only` is a way of faking accuracy and the arm is worthless; the preflight
    stops the whole run rather than logging it.
    """
    rng = np.random.RandomState(0)
    hard = []
    # (i) masking a FIXED support can only shrink a difference, so it is 1-Lipschitz in L2.
    # This is the only Lipschitz fact the mechanism needs, and it is the one that holds.
    worst = 0.0
    for _ in range(400):
        d = 500
        x = rng.normal(size=d)
        y = rng.normal(size=d) + 0.7
        x = x * min(1.0, 1.0 / (np.linalg.norm(x) + 1e-12))
        y = y * min(1.0, 1.0 / (np.linalg.norm(y) + 1e-12))
        mx, _ = top_k_projection([x], 0.1)
        ratio = (np.linalg.norm(x * mx[0] - y * mx[0])
                 / (np.linalg.norm(x - y) + 1e-12))
        worst = max(worst, float(ratio))
    hard.append(("fixed mask is 1-Lipschitz", worst, 1.0, 1e-9, "<="))
    # (ii) the noise multiplier is C-free: sigma/C = z/K for any clip radius. This is the
    # whole reason CLIP_REL can rescale C per architecture without touching a certificate.
    from . import dp_accountant
    a = dp_accountant.sigma_of(10.6, 0.10, 5) / 0.10
    b = dp_accountant.sigma_of(10.6, 4.72, 5) / 4.72
    hard.append(("sigma/C = z/K for any C", float(a / b), 1.0, 1e-9, "=="))
    # (iii) end-to-end. Both arms are run on the same clients and the SAME noise draw, so
    # the sparsified release must be a coordinate-subset of the plain DP release: nothing
    # reaches the model without noise on it, and at most m coordinates are touched.
    d, K, sp = 1000, 4, 0.1
    gw = [np.full(d, 0.3, dtype=np.float32)]
    cw = [gw[0] + rng.normal(scale=0.05, size=d) for _ in range(K)]
    plain, diag_p = dp_aggregate(gw, cw, C=0.5, z=10.6, rng=np.random.RandomState(1),
                                 return_diag=True, sparsity=None)
    sparse, diag_s = dp_aggregate(gw, cw, C=0.5, z=10.6, rng=np.random.RandomState(1),
                                  return_diag=True, sparsity=sp)
    s0, p0, g0 = np.asarray(sparse[0]), np.asarray(plain[0]), np.asarray(gw[0])
    # a coordinate is legitimate only if it is either untouched (== the old weight) or the
    # plain release's value. Anything else is an un-noised - i.e. unprotected - release.
    bad = int(np.count_nonzero((s0 != g0) & (s0 != p0)))
    hard.append(("no coordinate escapes noise", float(bad), 0.0, 1.0, "<="))
    updated = int(np.count_nonzero(s0 != g0))
    hard.append(("coords updated <= m+1", float(updated), float(round(sp * d) + 1),
                 1e-9, "<="))
    hard.append(("sigma identical to plain", float(diag_s["sigma"]),
                 float(diag_p["sigma"]), 1e-12, "=="))
    hard.append(("sigma = z*C/K", float(diag_s["sigma"]), 10.6 * 0.5 / K, 1e-9, "=="))
    hard.append(("support read from release",
                 1.0 if diag_s["support"] == "released-noise" else 0.0, 1.0, 1e-9, "=="))
    # (iiib) the head-only arm: a support fixed by the ARCHITECTURE. Nothing outside the
    # head may move at all, and what does move must be the plain release's own values.
    gb = [np.full(900, 0.2, dtype=np.float32), np.full(100, -0.4, dtype=np.float32)]
    r2 = np.random.RandomState(7)
    client_ws = [[gb[0] + r2.normal(scale=0.02, size=900),
                  gb[1] + r2.normal(scale=0.05, size=100)] for _ in range(K)]
    plain_h, diag_ph = dp_aggregate(gb, client_ws, C=0.5, z=10.6,
                                    rng=np.random.RandomState(2), return_diag=True)
    head_h, diag_hh = dp_aggregate(gb, client_ws, C=0.5, z=10.6,
                                   rng=np.random.RandomState(2), return_diag=True,
                                   head_tensors=1)
    hard.append(("frozen body does not move",
                 float(np.count_nonzero(np.asarray(head_h[0]) != np.asarray(gb[0]))),
                 0.0, 1.0, "<="))
    hard.append(("head arm adds no new values",
                 float(np.count_nonzero(np.asarray(head_h[1]) != np.asarray(plain_h[1]))),
                 0.0, 1.0, "<="))
    hard.append(("head arm coords = head size", float(diag_hh["noise_coords"]), 100.0,
                 1e-9, "=="))
    hard.append(("head arm sigma unchanged", float(diag_hh["sigma"]),
                 float(diag_ph["sigma"]), 1e-12, "=="))
    hard.append(("head support labelled",
                 1.0 if diag_hh["support"] == "head" else 0.0, 1.0, 1e-9, "=="))
    # the clipped average the certificate is written for really is bounded by C
    per_client = [clip_delta([np.asarray(w) - g0], 0.5)[0][0] for w in cw]
    mean_c = [np.sum(per_client, axis=0) / float(K)]
    hard.append(("clipped mean norm <= C", l2_norm(mean_c), 0.5, 1e-9, "<="))
    # (iv) the noise the mechanism actually drew corresponds to multiplier z, so the
    # certificate dp_accountant issues for (z, T, delta) is the certificate for this
    # release - read back from sigma, not assumed. Nothing here takes C or sparsity as an
    # argument, which is the whole reason neither the clip basis nor dp_sparsity can move
    # epsilon.
    z_implied = float(diag_s["sigma"]) * K / 0.5
    hard.append(("drawn noise is multiplier z", float(z_implied), 10.6, 1e-9, "=="))
    hard.append(("eps from drawn noise = certified eps",
                 float(dp_accountant.account_central_gaussian(z_implied, 20, 1e-5)["eps"]),
                 float(dp_accountant.account_central_gaussian(10.6, 20, 1e-5)["eps"]),
                 1e-9, "=="))
    ok = True
    if verbose:
        print("  dp mechanism assertions:")
    for name, got, exp, tol, cmp_ in hard:
        if cmp_ == "<=":
            within = got <= exp + tol * max(abs(exp), 1e-9)
        elif cmp_ == ">=":
            within = got >= exp - tol * max(abs(exp), 1e-9)
        else:
            within = abs(got - exp) <= tol * max(abs(exp), 1e-9)
        ok &= within
        if verbose:
            print("    %-28s got=%12.4f  %s %-12.4f  %s"
                  % (name, got, cmp_, exp, "OK" if within else "FAIL"))
    return ok


class AdaptiveClipper:
    """Gaussian quantile estimator of Andrew et al. (2021), replacing the submission's
    unaccounted Laplace clip-fraction update.

        b~ = ( sum_k 1[||Delta_k|| <= C] + N(0, sigma_b^2) ) / K
        C  <- C * exp( -eta * (b~ - target_q) )

    `freeze_rounds` stops the release after that many rounds and holds C fixed for the
    rest of the run. The estimator is the expensive part of this arm's certificate
    (1/(2*sigma_b^2) per round against 1/(2*z^2) for the model release), so the number of
    rounds it actually runs - `rounds_released` - is what the accountant composes, and it
    is read from this object rather than assumed. Nothing else changes: sigma = z*C/K
    tracks whatever radius is in force, so a frozen or a moving C costs the update
    mechanism the same z.
    """

    def __init__(self, C0=None, sigma_b=None, eta=None, target_q=None,
                 c_min=None, c_max=None, freeze_rounds=None):
        self.C = float(config.CLIP_C if C0 is None else C0)
        self.sigma_b = float(config.ADAPT_SIGMA_B if sigma_b is None else sigma_b)
        self.eta = float(config.ADAPT_ETA if eta is None else eta)
        self.target_q = float(config.ADAPT_TARGET_Q if target_q is None else target_q)
        self.c_min = float(config.ADAPT_C_MIN if c_min is None else c_min)
        self.c_max = float(config.ADAPT_C_MAX if c_max is None else c_max)
        self.freeze_rounds = (None if freeze_rounds is None
                              else int(freeze_rounds))
        self.history = []

    @property
    def rounds_released(self):
        """How many clip-fraction statistics this run actually released."""
        return int(sum(1 for h in self.history if h.get("released")))

    def update(self, update_norms, rng):
        K = float(len(update_norms))
        frozen = (self.freeze_rounds is not None
                  and self.rounds_released >= self.freeze_rounds)
        if K == 0 or frozen:
            # no release: no noise drawn, no budget spent, C held at its last value
            self.history.append(dict(round=len(self.history) + 1, C=self.C,
                                     noisy_fraction=None, true_fraction=None,
                                     released=False))
            return self.C
        indicators = np.array([1.0 if n <= self.C else 0.0 for n in update_norms])
        noisy_sum = float(indicators.sum() + rng.normal(0.0, self.sigma_b))
        b_tilde = float(np.clip(noisy_sum / K, 0.0, 1.0))
        self.C = float(np.clip(self.C * np.exp(-self.eta * (b_tilde - self.target_q)),
                               self.c_min, self.c_max))
        self.history.append(dict(round=len(self.history) + 1, C=self.C,
                                 noisy_fraction=b_tilde,
                                 true_fraction=float(indicators.mean()),
                                 released=True))
        return self.C


def clipper_selftest(verbose=True):
    """Preflight assertions for the frozen estimator - the count is the certificate.

    The adaptive arm's epsilon is composed from the number of clip-fraction statistics it
    actually released, so the pipeline is only honest if `rounds_released` counts releases
    and nothing else. A clipper that kept emitting the statistic while reporting a smaller
    number would be understating its own privacy cost, which is the exact error the
    rejected submission made. These assertions check the bookkeeping, not the arithmetic
    (dp_accountant.selftest does that).
    """
    rng = np.random.RandomState(0)
    T, W, K = 30, 7, 5
    hard = []
    c_freeze = AdaptiveClipper(C0=1.0, freeze_rounds=W)
    for _ in range(T):
        c_freeze.update(list(rng.uniform(0.2, 3.0, size=K)), rng)
    hard.append(("releases = freeze_rounds", float(c_freeze.rounds_released), float(W),
                 1e-9, "=="))
    hard.append(("history has every round", float(len(c_freeze.history)), float(T),
                 1e-9, "=="))
    after = [h["C"] for h in c_freeze.history[W:]]
    hard.append(("C constant once frozen", float(len(set(after))), 1.0, 1e-9, "=="))
    hard.append(("frozen rounds log no statistic",
                 float(sum(1 for h in c_freeze.history[W:] if h["noisy_fraction"] is None)),
                 float(T - W), 1e-9, "=="))
    hard.append(("only released rounds flagged",
                 float(sum(1 for h in c_freeze.history if h.get("released"))), float(W),
                 1e-9, "=="))
    # the published mechanism must still be available: no freeze, T releases
    c_all = AdaptiveClipper(C0=1.0, freeze_rounds=None)
    for _ in range(T):
        c_all.update(list(rng.uniform(0.2, 3.0, size=K)), rng)
    hard.append(("unfrozen releases every round", float(c_all.rounds_released), float(T),
                 1e-9, "=="))
    hard.append(("unfrozen radius moves", float(len(set(h["C"] for h in c_all.history)) > 1),
                 1.0, 1e-9, "=="))
    ok = True
    if verbose:
        print("  adaptive-clipper assertions:")
    for name, got, exp, tol, cmp_ in hard:
        within = abs(got - exp) <= tol * max(abs(exp), 1e-9)
        ok &= within
        if verbose:
            print("    %-30s got=%12.4f  %s %-12.4f  %s"
                  % (name, got, cmp_, exp, "OK" if within else "FAIL"))
    return ok


# ── robust aggregators (Task 6) ───────────────────────────────────────────────
# A client's weights are a LIST OF TENSORS, and the tensors differ in shape (an MLP
# has 8 of them, a CNN-BiLSTM 26), so every coordinate-wise operation has to run one
# tensor at a time. Stacking the clients directly raises
# "inhomogeneous shape" on the median arm and silently mis-slices everywhere else.
def _stack_by_tensor(client_ws, i):
    return np.stack([np.asarray(w[i]) for w in client_ws], axis=0)


def coordinate_median(client_ws):
    return [np.median(_stack_by_tensor(client_ws, i), axis=0)
            for i in range(len(client_ws[0]))]


def trimmed_mean(client_ws, trim=0.2):
    K = float(len(client_ws))
    n_trim = int(np.floor(K * trim))
    out = []
    for i in range(len(client_ws[0])):
        s = _stack_by_tensor(client_ws, i)
        if n_trim == 0:
            out.append(np.mean(s, axis=0))
            continue
        srt = np.sort(s, axis=0)
        keep = srt[n_trim:int(K) - n_trim] if K - 2 * n_trim > 0 else srt
        out.append(np.mean(keep, axis=0))
    return out


def krum(client_ws, assumed_f=None, n_select=1):
    """KRUM / blurred-krum with a FIXED assumed number of colluders.

    The submission passed the true number of attackers, which no honest server knows.
    Here `assumed_f` is a declared parameter (default config.BYZ_ASSUMED_F = 2) and is
    never derived from the actual attack configuration.
    """
    assumed_f = config.BYZ_ASSUMED_F if assumed_f is None else int(assumed_f)
    n = len(client_ws)
    if n <= 2 * assumed_f + 2:
        return coordinate_median(client_ws)
    # Krum scores are defined on the WHOLE update, so all tensors of a client are
    # concatenated into one vector before the pairwise distances are taken.
    flat = [np.concatenate([np.asarray(t).reshape(-1) for t in w]) for w in client_ws]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.sum((flat[i] - flat[j]) ** 2))
    m = n - assumed_f - 2
    scores = np.array([np.sum(np.sort(D[i])[1:m + 1]) for i in range(n)])
    sel = np.argsort(scores)[:max(1, int(n_select))]
    chosen = [client_ws[k] for k in sel]
    return [np.mean(_stack_by_tensor(chosen, i), axis=0)
            for i in range(len(chosen[0]))]


AGGREGATORS = {"fedavg": weighted_mean, "equal_mean": equal_mean,
               "median": coordinate_median, "trimmed_mean": trimmed_mean,
               "krum": krum}


def aggregate_plain(client_ws, client_ns, how="fedavg", assumed_f=None):
    if how == "fedavg":
        return weighted_mean(client_ws, client_ns)
    if how == "equal_mean":
        return equal_mean(client_ws)
    if how == "median":
        return coordinate_median(client_ws)
    if how == "trimmed_mean":
        return trimmed_mean(client_ws)
    if how == "krum":
        return krum(client_ws, assumed_f=assumed_f)
    raise ValueError("unknown aggregator %r" % how)


# ── Byzantine attacks ─────────────────────────────────────────────────────────
UPDATE_ATTACKS = ("none", "signflip", "scaled", "noise", "ipm", "signflip_x1", "scaled_neg5", "ipm_eps0.5")
DATA_ATTACKS = ("labelflip",)


def is_malicious(client_ids, n_clients, fraction, rng):
    """Deterministic adversary set: the first ceil(fraction*K) client ids."""
    n_adv = int(round(float(fraction) * n_clients))
    adv = set(int(c) for c in np.array(client_ids)[:n_adv]) if n_adv > 0 else set()
    return adv


def attack_update(delta, global_w, how, rng, honest_deltas=None, factor=None,
                  noise_std=None, ipm_eps=None):
    """Transform one malicious client's delta. `honest_deltas` are the deltas of the
    non-colluding clients (needed by IPM)."""
    d = [np.asarray(x).copy() for x in delta]
    if how == "none":
        return d
    if how in ("signflip", "signflip_x1"):
        # sign-flipping attack: the client sends the NEGATIVE of its honest update
        return [-x for x in d]
    if how in ("scaled", "scaled_neg5"):
        # scaled poisoning: the reversed update, boosted by BYZ_SCALED_FACTOR, so a
        # minority can outweigh the honest majority under plain averaging
        f = config.BYZ_SCALED_FACTOR if factor is None else factor
        return [x * -float(f) for x in d]
    if how == "noise":
        s = config.BYZ_NOISE_STD if noise_std is None else noise_std
        return [x + rng.normal(0.0, float(s), x.shape) for x in d]
    if how in ("ipm", "ipm_eps0.5"):
        # Xie et al. 2019: malicious update = -eps * MEAN OF THE HONEST UPDATES.
        # The submission's "ipm" was an alias for sign-flip; this is the real thing.
        eps = config.BYZ_IPM_EPS if ipm_eps is None else ipm_eps
        if not honest_deltas:
            return [x * -float(eps) for x in d]
        mean_honest = [np.mean([h[i] for h in honest_deltas], axis=0)
                       for i in range(len(d))]
        return [np.asarray(m) * (-float(eps)) for m in mean_honest]
    raise ValueError("unknown update attack %r" % how)


def flip_labels(y, rng, n_class=None, to=None):
    n_class = n_class or config.N_CLASS
    to = config.BYZ_LABEL_FLIP_TO if to is None else to
    out = np.asarray(y).copy()
    return np.where(out == to, (out + 1) % n_class, out)
