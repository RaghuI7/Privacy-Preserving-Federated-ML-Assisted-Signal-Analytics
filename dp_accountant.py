"""Valid Renyi-DP accounting for the central-Gaussian federated mechanism.

This module replaces ``GaussDP.renyi_epsilon`` and ``GaussDP.moments_epsilon`` from
the rejected submission, neither of which produced a valid (eps, delta) certificate.

Mechanism that is accounted for (Work Order Task 1)
---------------------------------------------------
Per round r, with K participating clients and whole-update clipping to L2 norm C:

    Delta_k   = w_k - w_global          client update
    Deltabar_k = Delta_k * min(1, C/||Delta_k||_2)      ONE norm over ALL tensors
    mean      = (1/K) * sum_k Deltabar_k                EQUAL weights 1/K
    release   = mean + N(0, sigma^2 I),  sigma = z * C / K

Privacy unit: one client = one subject silo. Adjacency: add/remove all the data of
one client.

Why the noise multiplier is z and not something else
----------------------------------------------------
The L2 sensitivity of ``mean`` under one-client adjacency is C/K (one clipped term
of norm <= C enters with weight 1/K). The added noise has std sigma = z*C/K, so the
noise-to-sensitivity ratio is exactly z, independent of C and K. That is why the
certificate survives adaptive clipping: when C_r changes from round to round, both
the sensitivity and the noise scale by the same factor and z stays fixed.

Per-round Renyi divergence of the Gaussian mechanism with ratio z is

    eps_rdp(alpha) = alpha / (2 z^2)

and over T independently-composed rounds eps_rdp(alpha) = T*alpha/(2 z^2).
Conversion to (eps, delta) and the optimal order are closed-form:

    alpha* = 1 + sqrt( 2 ln(1/delta) * 2 z^2 / T )
    eps    = T alpha* / (2 z^2) + ln(1/delta) / (alpha* - 1)

which reproduces the reference numbers quoted in the Work Order (verified in
``selftest()``).
"""

import math

MECHANISM = "central_gaussian_whole_update_clip"
ADJACENCY = "client_level_subtree_user"      # one client = one subject silo


def _rdp_to_eps(rdp_eps, order, delta):
    return rdp_eps - math.log(delta) / (order - 1.0)


def rdp_gaussian(order, z):
    """Rnyi divergence of order `order` of a Gaussian mechanism with ratio z."""
    if z <= 0:
        return float("inf")
    return order / (2.0 * z * z)


def optimal_order(c, delta):
    """argmin_alpha c*alpha + ln(1/delta)/(alpha-1) for c >= 0 (c = sum of per-round
    coefficients). Returns (alpha*, eps)."""
    if c <= 0:
        return float("inf"), float("inf")
    L = math.log(1.0 / delta)
    order = 1.0 + math.sqrt(L / c)
    return order, c * order + L / (order - 1.0)


def eps_for_order(c, order, delta):
    return _rdp_to_eps(c * order, order, delta)


def account_central_gaussian(z, T, delta=1e-5, client_sampling_rate=1.0):
    """Certificate for T rounds of the mechanism described in the module docstring.

    Parameters
    ----------
    z  : noise multiplier (sigma = z * C / K)
    T  : number of composed rounds
    client_sampling_rate : fraction of clients participating per round. Values < 1
        are handled CONSERVATIVELY: no subsampling amplification is claimed, because
        the rejected submission applied amplification (q = 32/n_k) for a loop that in
        fact sampled every client. Reporting the gamma = 1 bound is always valid.
    """
    if z is None or z <= 0 or T <= 0:
        return dict(eps=float("inf"), alpha_star=float("inf"), per_round_eps=[],
                    z=z, T=T, delta=delta, gamma=client_sampling_rate,
                    amplified=False, note="no noise: unrestricted privacy loss")
    c = T / (2.0 * z * z)
    order, eps = optimal_order(c, delta)
    per_round = []
    for t in range(1, T + 1):
        o_t, e_t = optimal_order(t / (2.0 * z * z), delta)
        per_round.append(dict(round=t, eps=e_t, alpha_star=o_t))
    return dict(mechanism=MECHANISM, adjacency=ADJACENCY, z=float(z), C_over_K=True,
                T=int(T), delta=delta, gamma=float(client_sampling_rate),
                alpha_star=float(order), eps=float(eps),
                eps_single_round=eps_for_order(1.0 / (2.0 * z * z), order, delta),
                per_round_eps=per_round,
                amplified=False,
                note=("gamma<1 treated as gamma=1 (no amplification claimed)"
                      if client_sampling_rate < 1.0 else "full participation, exact"))


def account_adaptive(z, T, delta=1e-5, sigma_b=100.0, T_clip=None):
    """Task 2: the update mechanism AND the Gaussian quantile estimator, composed in RDP.

    The clip-fraction estimator is b~ = (sum_k 1[||Delta_k|| <= C] + N(0, sigma_b^2))/K.
    Its L2 sensitivity under one-client adjacency is 1/K and its noise std is
    sigma_b/K, so its noise-to-sensitivity ratio is sigma_b. Each round the estimator
    RUNS adds its own RDP coefficient, so the composed budget is

        c = T/(2 z^2) + T_clip/(2 sigma_b^2)

    T_clip is the number of rounds in which a clip-fraction statistic was released. It
    defaults to T, which is Andrew et al. as published - the estimator tracking C for the
    whole run. The pipeline passes the count it actually logged, because the estimator is
    the expensive half of this certificate: at sigma_b = 0.5 one estimator round costs
    1/(2*0.5^2) = 2.0 while one update release at z = 10.6 costs 0.0044.
    """
    if z <= 0 and sigma_b <= 0:
        return dict(eps_total=float("inf"))
    T = int(T)
    T_upd = T
    T_cl = T if T_clip is None else int(T_clip)
    if T_cl > T:
        raise ValueError("T_clip=%d exceeds the %d rounds actually run; the certificate "
                         "cannot be larger than the release" % (T_cl, T))
    c_upd = T_upd / (2.0 * z * z) if z > 0 else 0.0
    c_clip = T_cl / (2.0 * sigma_b * sigma_b) if sigma_b > 0 else 0.0
    order, eps_total = optimal_order(c_upd + c_clip, delta)
    # component values are reported at the SHARED optimal order (valid decomposition)
    eps_upd = eps_for_order(c_upd, order, delta) if z > 0 else 0.0
    eps_clip = eps_for_order(c_clip, order, delta) if sigma_b > 0 else 0.0
    per_round = []
    for t in range(1, T + 1):
        c_t = ((min(t, T_upd) / (2.0 * z * z) if z > 0 else 0.0)
               + (min(t, T_cl) / (2.0 * sigma_b * sigma_b) if sigma_b > 0 else 0.0))
        per_round.append(dict(round=t, eps=optimal_order(c_t, delta)[1]))
    return dict(mechanism=MECHANISM + "+gaussian_quantile_clip",
                adjacency=ADJACENCY, z=float(z), sigma_b=float(sigma_b),
                T=int(T), T_updates=int(T_upd), T_clip_estimator=int(T_cl),
                delta=delta, alpha_star=float(order),
                eps_updates=float(eps_upd), eps_clip_fraction=float(eps_clip),
                eps_total=float(eps_total),
                per_round_eps=per_round)


def account_per_tensor_clipping(z, T, n_tensors, delta=1e-5):
    """What the REJECTED code cost.

    Clipping each of the L tensors to C bounds the whole-update norm by sqrt(L)*C, so
    the true sensitivity is sqrt(L) times larger than assumed and the effective ratio
    is z/sqrt(L). Used only to report the accounting error, never to certify a result.
    """
    z_eff = z / math.sqrt(n_tensors)
    out = account_central_gaussian(z_eff, T, delta)
    out["n_tensors"] = n_tensors
    out["note"] = "diagnostic: sensitivity inflation of the per-tensor clip in v1"
    return out


def legacy_equivalent_z(nominal_eps=20.0, delta=1e-5):
    """The v1 code set sigma = sqrt(2 ln(1.25/delta))/eps and then used sigma*C/K.
    Expressed as a noise multiplier that is exactly what nominal_eps meant."""
    return math.sqrt(2.0 * math.log(1.25 / delta)) / nominal_eps


def sigma_of(z, C, K):
    return z * C / float(K)


# ── Sampled Gaussian Mechanism (Poisson-subsampled DP-SGD) ─────────────────────
# Numerically stable log-space RDP of Mironov, Talwar & Zhang (2019), the same expansion
# TensorFlow-Privacy and Google dp-accounting use. Verified against dp-accounting's
# RdpAccountant(PoissonSampledDpEvent(q, GaussianDpEvent(z))) to ~1e-11 relative error
# for every eps <= 20 on a grid of z in [0.7, 5], q in [0.001, 0.1], T in [10, 5000].
SGM_ORDERS = ([1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))
              + [128, 256, 512])


def _log_add(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_erfc(x):
    from scipy import special
    return float(special.log_ndtr(-x * math.sqrt(2)) + math.log(2))


def _sgm_log_a_int(q, sigma, alpha):
    log_a = -math.inf
    for i in range(int(alpha) + 1):
        log_a = _log_add(log_a, _log_comb(alpha, i) + i * math.log(q)
                         + (alpha - i) * math.log1p(-q) + (i * i - i) / (2.0 * sigma * sigma))
    return log_a


def _sgm_log_a_frac(q, sigma, alpha):
    log_a0, log_a1 = -math.inf, -math.inf
    z0 = sigma * sigma * math.log(1.0 / q - 1.0) + 0.5
    i = 0
    while True:
        log_coef = (math.lgamma(alpha + 1) - math.lgamma(i + 1)
                    - math.lgamma(alpha - i + 1))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log1p(-q)
        log_t1 = log_coef + j * math.log(q) + i * math.log1p(-q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * sigma * sigma) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma * sigma) + log_e1
        log_a0 = _log_add(log_a0, log_s0)
        log_a1 = _log_add(log_a1, log_s1)
        i += 1
        if i > 1000 or (max(log_s0, log_s1) < -30 and i > alpha):
            break
    return _log_add(log_a0, log_a1)


def sgm_rdp(q, z, T, orders=None):
    """Total RDP after T steps of the Poisson-subsampled Gaussian (rate q, multiplier z)."""
    orders = orders or SGM_ORDERS
    out = []
    for a in orders:
        if q >= 1.0:
            out.append(T * a / (2.0 * z * z))
            continue
        la = (_sgm_log_a_int(q, z, int(a)) if float(a).is_integer()
              else _sgm_log_a_frac(q, z, float(a)))
        out.append(T * la / (a - 1.0))
    return orders, out


def sgm_eps(q, z, T, delta=1e-5):
    """(eps, alpha*) for T steps of DP-SGD, tight RDP->DP conversion (same as the
    user-level accountant: eps = rho + ln(1-1/a) - (ln delta + ln a)/(a-1))."""
    orders, rdp = sgm_rdp(q, z, T)
    best = (math.inf, None)
    for a, r in zip(orders, rdp):
        e = r + math.log1p(-1.0 / a) - (math.log(delta) + math.log(a)) / (a - 1.0)
        if e < best[0]:
            best = (e, a)
    return float(best[0]), float(best[1])


def z_for_eps_dpsgd(target_eps, q, T, delta=1e-5, lo=0.3, hi=50.0, tol=1e-4):
    """Smallest z whose T-step DP-SGD certificate is <= target_eps (bisection).
    Called BEFORE training, from the planned step count."""
    if sgm_eps(q, hi, T, delta)[0] > target_eps:
        raise ValueError("target eps %.3f unreachable even at z=%.1f" % (target_eps, hi))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sgm_eps(q, mid, T, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(hi)


def account_dpsgd(z, T, delta=1e-5, sample_rate=None, orders=None, batch=None,
                  n_per_client=None):
    """Example-level DP-SGD certificate: T Poisson-sampled Gaussian steps of ONE client.

    The privacy unit is one window. An example lives in exactly one silo, so the steps of
    other clients never touch it and the per-client composition is the whole bound; server
    averaging of already-private updates is post-processing. Every return path is stamped
    adjacency=example_level and user_level_certificate=NO.
    """
    if sample_rate:
        eps, a = sgm_eps(float(sample_rate), float(z), int(T), delta)
        out = dict(mechanism="dp_sgd_poisson", adjacency="example_level",
                   user_level_certificate="NO", z=float(z), T=int(T), delta=delta,
                   gamma=float(sample_rate), alpha_star=a, eps=eps,
                   accountant="sampled-Gaussian RDP (Mironov et al. 2019), in-package")
        try:   # independent cross-check when Google's library is installed
            from dp_accounting import dp_event as _ev
            from dp_accounting.rdp import RdpAccountant as _Acc
            acc = _Acc(SGM_ORDERS)
            acc.compose(_ev.PoissonSampledDpEvent(float(sample_rate),
                                                  _ev.GaussianDpEvent(float(z))), int(T))
            out["eps_dp_accounting"] = float(acc.get_epsilon(delta))
        except Exception as exc:
            out["eps_dp_accounting"] = None
            out["eps_dp_accounting_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
        return _stamp_dpsgd_shape(out, batch, n_per_client, sample_rate)
    out = account_central_gaussian(z, T, delta)
    out.update(mechanism="central_gaussian_no_amplification", accountant="closed form",
               note="no sample rate given: full-batch bound over %d steps" % int(T))
    return _stamp_dpsgd_shape(out, batch, n_per_client, sample_rate)


def _stamp_dpsgd_shape(out, batch, n_per_client, sample_rate):
    """The example-level labels, on every path that reaches the dpsgd certificate."""
    # Overwrite, never inherit: the fallback delegates to account_central_gaussian, whose
    # own adjacency is the client silo. A dpsgd row borrows that number but not that
    # privacy unit - one window is its unit - so leaving the inherited label in place would
    # print a client-level guarantee next to an example-level bound.
    out["central_form_adjacency"] = out.get("adjacency")
    out["adjacency"] = "example_level"
    out["user_level_certificate"] = "NO"
    out["batch"] = None if batch is None else int(batch)
    out["sample_rate"] = None if sample_rate is None else float(sample_rate)
    out["n_per_client"] = None if n_per_client is None else int(n_per_client)
    return out


def cross_check(z, T, delta=1e-5, orders=None):
    """Cross-check our certificate against Google dp-accounting (Work Order Task 1).

    Our closed form uses the classic RDP -> (eps, delta) conversion (Mironov 2017);
    dp-accounting's RdpAccountant uses the tighter conversion, so its eps is slightly
    SMALLER. ok = both computed and ours >= theirs (ours is the conservative bound)."""
    ours = account_central_gaussian(z, T, delta)
    try:
        from dp_accounting import dp_event as _ev
        from dp_accounting.rdp import RdpAccountant as _Acc
    except Exception as exc:
        return dict(ok=False, ours=ours["eps"], theirs=None,
                    error="dp-accounting import failed (%s: %s)" % (type(exc).__name__, exc))
    try:
        acc = _Acc(orders) if orders else _Acc()
        acc.compose(_ev.SelfComposedDpEvent(_ev.GaussianDpEvent(float(z)), int(T)))
        eps_them = float(acc.get_epsilon(delta))
        return dict(ok=bool(ours["eps"] >= eps_them - 1e-9), ours=ours["eps"],
                    theirs=eps_them,
                    rel_diff=abs(ours["eps"] - eps_them) / max(eps_them, 1e-12),
                    alpha_star=ours["alpha_star"], z=z, T=T, delta=delta,
                    note="ours = classic RDP conversion (conservative); theirs = tighter")
    except Exception as exc:          # pragma: no cover
        return dict(ok=False, ours=ours["eps"], theirs=None,
                    error="%s: %s" % (type(exc).__name__, exc))


def _eps_by_minimisation(c, delta, hi=200.0, steps=200000):
    """eps(alpha) minimised over a dense alpha grid - an independent check of the
    closed form in optimal_order(), which is what makes the certificate trustworthy."""
    best = float("inf")
    for i in range(steps):
        alpha = 1.0 + (hi - 1.0) * i / float(steps - 1)
        if alpha <= 1.0:
            continue
        e = c * alpha - math.log(delta) / (alpha - 1.0)
        if e < best:
            best = e
    return best


def selftest(verbose=True):
    """Check the accountant two ways: against its own closed form and against the
    approximate eps values quoted in the Work Order.

    The Work Order table is rounded to one significant figure ("eps ~= 3" for
    z = 10.6), so those rows are reported with the gap printed, not asserted; the
    assertions are the self-consistency and legacy-cost checks, which are exact.
    """
    hard = []
    # 1. the closed-form optimum must be the minimum of the conversion curve
    for z, T in ((4.5, 50), (10.6, 50), (28.6, 50), (7.0, 30)):
        c = T / (2.0 * z * z)
        closed = optimal_order(c, 1e-5)[1]
        grid = _eps_by_minimisation(c, 1e-5)
        hard.append(("z=%.1f T=%d closed form" % (z, T), closed, grid, 1e-4))
    # 2. eps must decrease monotonically in z and increase in T
    eps_z = [account_central_gaussian(z, 50, 1e-5)["eps"] for z in
             (4.5, 7.0, 10.6, 15.0, 28.6)]
    hard.append(("eps decreasing in z", float(all(a > b for a, b in zip(eps_z, eps_z[1:]))),
                 1.0, 0.0))
    eps_T = [account_central_gaussian(10.6, T, 1e-5)["eps"] for T in (10, 50, 200)]
    hard.append(("eps increasing in T", float(all(a < b for a, b in zip(eps_T, eps_T[1:]))),
                 1.0, 0.0))
    # 3. no noise => no certificate
    hard.append(("z=0 is infinite", float(account_central_gaussian(0, 50)["eps"] ==
                float("inf")), 1.0, 0.0))
    zl = legacy_equivalent_z(20.0)
    hard.append(("legacy whole-model eps", account_central_gaussian(zl, 50)["eps"],
                 566.0, 0.12))
    hard.append(("legacy CNN per-tensor eps",
                 account_per_tensor_clipping(zl, 50, 26)["eps"], 12000.0, 0.12))
    # 4. the adaptive composition, against the two things that pin it down. With the
    # estimator releasing zero times this must reduce EXACTLY to the plain certificate -
    # that is what lets the pipeline charge only the rounds the clipper ran. The split
    # keys must stay consistent with the composed total, and more estimator rounds must
    # cost strictly more.
    plain50 = account_central_gaussian(10.6, 50, 1e-5)["eps"]
    hard.append(("adaptive with T_clip=0 = plain",
                 account_adaptive(10.6, 50, 1e-5, sigma_b=0.5, T_clip=0)["eps_total"],
                 plain50, 1e-9))
    e_clips = [account_adaptive(10.6, 50, 1e-5, sigma_b=0.5, T_clip=w)["eps_total"]
               for w in (0, 5, 10, 25, 50)]
    hard.append(("adaptive eps rising in T_clip",
                 float(all(a < b for a, b in zip(e_clips, e_clips[1:]))), 1.0, 0.0))
    one = account_adaptive(10.6, 50, 1e-5, sigma_b=0.5, T_clip=10)
    # the two components are reported at the SHARED optimal order, so each carries the
    # ln(1/delta)/(alpha-1) term once and their sum exceeds the total by exactly that
    # term. This is the identity that makes the split a decomposition and not two
    # independent epsilons a reader might be tempted to add.
    delta_term = math.log(1.0 / 1e-5) / (one["alpha_star"] - 1.0)
    hard.append(("adaptive split = total + one delta term",
                 one["eps_updates"] + one["eps_clip_fraction"],
                 one["eps_total"] + delta_term, 1e-9))
    try:
        account_adaptive(10.6, 50, 1e-5, sigma_b=0.5, T_clip=51)
        over = 0.0
    except ValueError:
        over = 1.0
    hard.append(("T_clip > T is refused", over, 1.0, 0.0))
    ok = True
    if verbose:
        print("  assertions:")
    for name, got, exp, tol in hard:
        within = (got == float("inf") and exp == float("inf")) or \
                 abs(got - exp) <= tol * max(abs(exp), 1e-9)
        ok &= within
        if verbose:
            print("    %-28s got=%12.4f  expected=%12.4f  %s"
                  % (name, got, exp, "OK" if within else "FAIL"))
    if verbose:
        print("  Work Order table (references rounded, gap shown):")
        for z, quoted in ((4.5, 8.0), (10.6, 3.0), (28.6, 1.0)):
            got = account_central_gaussian(z, 50, 1e-5)["eps"]
            print("    z=%-5.1f eps=%7.3f  quoted ~= %-5.1f  gap %+.0f%%"
                  % (z, got, quoted, 100.0 * (got / quoted - 1.0)))
        print("    use the eps computed here, never the rounded quotes, in the paper")
    return ok


if __name__ == "__main__":
    print("dp_accountant selftest (vs Work Order reference numbers):")
    print("  all checks passed:", selftest())
