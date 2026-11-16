"""One federated-learning training loop for every dataset, architecture and mechanism.

Every table and figure in the revision is produced by `run_fl` (or `run_central`), so
methodology cannot drift between datasets or between the MLP and the CNN - the
inconsistency the reviewers found (WISDM at 23% in the figures, 84.9% in Table 5) is a
property of this file, not of the experiment scripts.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import config, dp_accountant, fl, models
from .metrics import classification_metrics

MAX_DEVICE_BYTES = 2_000_000_000


class LocalData:
    """Client/test data, parked on the GPU when it fits and on the CPU otherwise."""

    def __init__(self, X, y, device, dtype=None):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y).astype(np.int64), dtype=torch.long)
        nbytes = (self.X.numel() + self.y.numel() * 8) * 4
        self.to_device = (device.type == "cuda") and nbytes < MAX_DEVICE_BYTES
        self.device = device
        if self.to_device:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        return len(self.y)

    def batch(self, idx):
        xb = self.X[idx]
        yb = self.y[idx]
        if self.to_device:
            return xb, yb
        return xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)

    def subset(self, idxs):
        return LocalData._view(self, idxs, self.device)

    @staticmethod
    def _view(parent, idxs, device):
        child = object.__new__(LocalData)
        child.X = parent.X[idxs]
        child.y = parent.y[idxs]
        child.to_device = parent.to_device
        child.device = device
        return child


def class_weights_of(y, n_class=None):
    """Inverse-frequency class weights from ONE silo's own labels (no other client's data
    is read). Classes absent from the silo get weight 0 - they produce no loss terms
    anyway - and the weights are normalised to mean 1 over the classes present."""
    n_class = n_class or config.N_CLASS
    y = np.asarray(y).astype(int)
    cnt = np.bincount(y, minlength=n_class).astype(np.float64)
    w = np.zeros(n_class)
    present = cnt > 0
    w[present] = cnt[present].sum() / (present.sum() * cnt[present])
    return w


def _eps_value(v):
    """per_round_eps entries are dicts ({"round", "eps", ...}); the curve needs the number."""
    if isinstance(v, dict):
        v = v.get("eps")
    return None if v is None else float(v)


def lr_at(rnd, rounds, lr=None, lr_min=None):
    """Cosine decay from LR (round 1) to LR_MIN (last round)."""
    lr = config.LR if lr is None else lr
    lr_min = config.LR_MIN if lr_min is None else lr_min
    if config.LR_SCHEDULE != "cosine" or rounds <= 1:
        return float(lr)
    t = (rnd - 1) / float(rounds - 1)
    return float(lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos(np.pi * t)))


def _small_rotations(n, device, max_deg=15.0, gen=None):
    """n random 3x3 rotations of at most max_deg degrees (axis-angle). Small on purpose:
    the direction of gravity is what separates sitting, standing and lying."""
    axis = torch.randn(n, 3, generator=gen)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
    ang = (torch.rand(n, generator=gen) * 2 - 1) * np.deg2rad(max_deg)
    K = torch.zeros(n, 3, 3)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    I = torch.eye(3).expand(n, 3, 3)
    s_, c_ = torch.sin(ang).view(-1, 1, 1), torch.cos(ang).view(-1, 1, 1)
    return (I + s_ * K + (1 - c_) * (K @ K)).to(device)


def augment_windows(xb, gen=None):
    """Scaling (x N(1, 0.1)), jitter (+ N(0, 0.05)) and one small 3-D rotation per window
    applied to every 3-channel sensor triple. Works on a copy; raw-window inputs only."""
    if xb.dim() != 3:
        return xb
    n, wl, c = xb.shape
    dev = xb.device
    out = xb * (1.0 + 0.1 * torch.randn(n, 1, c, generator=gen).to(dev))
    out = out + 0.05 * torch.randn(out.shape, generator=gen).to(dev)
    n_tri = c // 3
    if n_tri:
        R = _small_rotations(n, dev, gen=gen)
        parts = []
        for t in range(n_tri):
            parts.append(torch.einsum("nij,nwj->nwi", R, out[:, :, 3 * t:3 * t + 3]))
        if c % 3:
            parts.append(out[:, :, 3 * n_tri:])
        out = torch.cat(parts, dim=2)
    return out


def train_local(model, data, epochs, batch, lr, device, mu=0.0, global_torch=None,
                freeze_head=False, grad_clip=5.0, seed=0, eval_every=0, eval_fn=None,
                class_weights=None, augment=None, cosine_epochs=False):
    """Real local gradient steps. mu > 0 is a genuine FedProx proximal term.

    eval_every/eval_fn give a trajectory for the pooled (non-federated) baseline without
    changing the optimisation: the checkpoint pass runs under no_grad, so it consumes no
    random numbers and cannot shift the training stream.
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY,
                           foreach=True)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)),
                                                        eta_min=config.LR_MIN)
             if cosine_epochs and config.LR_SCHEDULE == "cosine" else None)
    cw = (torch.as_tensor(class_weights, dtype=torch.float32, device=device)
          if class_weights is not None else None)
    do_aug = config.AUGMENT if augment is None else bool(augment)
    if freeze_head and hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad_(False)
    model.train()
    n = len(data)
    if n == 0:
        return 0.0
    nb = max(1, int(batch))
    final_loss = 0.0
    for ep in range(1, int(epochs) + 1):
        perm = torch.randperm(n, generator=gen)
        if data.to_device:
            perm = perm.to(device)
        for start in range(0, n, nb):
            idx = perm[start:start + nb]
            xb, yb = data.batch(idx)
            if do_aug and xb.dim() == 3:
                xb = augment_windows(xb, gen=gen)
            opt.zero_grad()
            logits = model(xb)
            loss = Fn.cross_entropy(logits, yb, weight=cw)
            if mu > 0 and global_torch is not None:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_torch)
                           if p.requires_grad)
                loss = loss + 0.5 * float(mu) * prox
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip)
            opt.step()
            final_loss = float(loss.item())
        if sched is not None:
            sched.step()
        if eval_every and eval_fn is not None and (ep % int(eval_every) == 0 or
                                                   ep == int(epochs)):
            eval_fn(ep)
            model.train()          # evaluate() switches to eval mode; restore it
    if freeze_head and hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad_(True)
    return final_loss


@torch.no_grad()
def evaluate(model, data, batch=512):
    model.eval()
    preds, probs, losses = [], [], []
    n = len(data)
    for start in range(0, n, int(batch)):
        idx = torch.arange(start, min(start + int(batch), n))
        xb, yb = data.batch(idx.to(data.device) if data.to_device else idx)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        losses.append(Fn.cross_entropy(logits, yb, reduction="none").cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros(0, int)
    prob = np.concatenate(probs) if probs else np.zeros((0, config.N_CLASS))
    loss = np.concatenate(losses) if len(losses) else np.zeros(0)
    return pred, prob, loss


def _init_global(arch, feature_dim, window_shape, n_class, device, seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    return models.build(arch, feature_dim=feature_dim, window_shape=window_shape,
                        n_class=n_class, device=device)


def dp_clip_C(dp, weights, shared_slice, arch):
    """Resolve the clip radius for one DP run, BEFORE any client data is touched.

    An absolute `dp["C"]` is used as given (that is what the Task 2 sweep varies and what
    the rejected paper's FL+DP arm certifies at). `C=None` asks for CLIP_REL x ||w0||: the
    init is seeded and public, so this reads no private window, and eps cannot move because
    sigma = z*C/K and sensitivity = C/K scale together. What it fixes is reach - a constant
    0.10 bounds how far T rounds of clipped updates can travel, and the answer depends on
    d and on the architecture, not on the privacy budget.
    """
    raw = dp.get("C")
    if raw is not None:
        return float(raw), None
    init_norm = fl.l2_norm([np.asarray(w, dtype=np.float64)
                            for w in list(weights)[shared_slice]])
    return float(config.CLIP_REL) * init_norm, init_norm


def run_fl(bundle_data, client_indices, arch, seed, rounds=None, local_epochs=None,
           batch=None, lr=None, method="fedavg", mu=0.0, dp=None, use_fedper=False,
           server_momentum=False, attacks=None, Xval=None, yval=None, Xte=None, yte=None,
           label="FL", partition_id=0, extra_config=None, device=None, verbose=False,
           grad_clip=5.0, heterogeneity=None, keep_model=False, init_weights=None):
    """Federated training for one (seed, partition, mechanism) cell.

    dp = None                       non-private
    dp = dict(z=..., C=..., T=...)   fixed clipping
    dp = dict(z=..., adaptive=True)  Gaussian quantile estimator, accounted
    attacks = dict(fraction=, update=, labelflip=bool, aggreg=)
    """
    C_ = config.cfg()
    config.check_budget()
    rounds = C_["rounds"] if rounds is None else rounds
    local_epochs = (C_["local_epochs_mlp"] if arch.upper().startswith("MLP")
                    else C_["local_epochs_cnn"]) if local_epochs is None else local_epochs
    batch = batch or config.BATCH
    lr = lr or config.LR
    device = device or models.device_for(True)

    Xtr, ytr = bundle_data
    data_tr = LocalData(Xtr, ytr, device)
    data_te = LocalData(Xte, yte, device)
    data_val = LocalData(Xval, yval, device) if Xval is not None else None

    gm = _init_global(arch, Xtr.shape[1] if Xtr.ndim == 2 else None,
                      Xtr.shape[1:], None, device, seed)
    n_class = models.n_class_of(gm)
    if init_weights is not None:
        # a public starting point (stage dplearn): the body was trained on a DIFFERENT
        # dataset, so it reads no private window and costs no privacy budget
        models.set_weights(gm, init_weights)
    weights = models.get_weights(gm)
    n_head = models.n_head_params(gm) if use_fedper else 0
    shared_slice = slice(0, len(weights) - n_head) if n_head else slice(0, len(weights))

    rng = np.random.RandomState(int(seed) + 1)
    torch.manual_seed(int(seed) + 1)

    # FedProx is FedAvg aggregation plus the proximal term in LOCAL training (mu > 0),
    # so the method name is not an aggregator and must not reach aggregate_plain.
    aggreg_name = "fedavg" if method == "fedprox" else method

    adaptive = None
    iter_snap = []
    if dp is not None:
        if aggreg_name not in ("fedavg", "equal_mean"):
            raise ValueError("the C/K sensitivity argument needs equal 1/K weights; "
                             "aggregator %r is not certified" % aggreg_name)
        dp = dict(dp)
        dp.setdefault("T", rounds)
        dp.setdefault("delta", config.DELTA)
        dp.setdefault("sparsity", None)
        dp.setdefault("iter_avg", 0)
        C_eff, init_norm = dp_clip_C(dp, weights, shared_slice, arch)
        dp["C"] = C_eff
        dp["init_norm"] = init_norm
        # the head-only support is fixed by the architecture, so it is decided here, before
        # any client data is read, and never depends on a released value
        dp["head_tensors"] = None
        if dp.get("head_only"):
            if n_head:
                raise ValueError("dp head_only releases the trailing tensors of the "
                                 "aggregated vector; combining it with FedPer's slice is "
                                 "undefined and silently uncertified")
            ht = int(models.n_head_params(gm))
            if ht <= 0 or ht >= len(weights):
                raise ValueError("arch %s exposes %d head tensors of %d: head_only would "
                                 "release nothing or everything" % (arch, ht, len(weights)))
            dp["head_tensors"] = ht
        if dp.get("adaptive"):
            # The estimator releases a statistic every round it runs, and that statistic
            # is the dominant term of this arm's certificate. W is the budget it is
            # allowed to spend, set in config before any accuracy was read; the run then
            # accounts for the releases it actually made, read back from the clipper.
            adaptive = fl.AdaptiveClipper(C0=C_eff,
                                          freeze_rounds=dp.get("freeze_rounds",
                                                               config.ADAPT_FREEZE_ROUNDS))

    adv = set()
    attack_update = "none"
    labelflip = False
    aggreg = aggreg_name
    if attacks is not None:
        adv = fl.is_malicious(list(range(len(client_indices))), len(client_indices),
                              attacks.get("fraction", 0.0), rng)
        attack_update = attacks.get("update", "none")
        labelflip = bool(attacks.get("labelflip", False))
        aggreg = attacks.get("aggreg", aggreg)

    vel = None
    per_round = []
    clip_history = []
    # IPM is defined against the mean of the HONEST updates (Xie et al. 2019), so every
    # update attack is applied once the honest deltas of the round are known. For IPM the
    # attacker's own local training is then not part of the attack, so it is skipped.
    attack_deferred = attack_update in ("ipm", "ipm_eps0.5")
    t_start = time.time()
    # The privacy cost already paid at each round, for the live print and the per-round log.
    # Closed-form and cheap, so it is computed once for the whole run, not once per round.
    # The adaptive arm gets None here: its certificate depends on how many estimator
    # releases actually happened, which only the final accounting knows - inventing a
    # per-round curve for it would be the kind of number this pipeline exists to avoid.
    eps_curve = None
    if dp is not None and adaptive is None:
        try:
            eps_curve = dp_accountant.account_central_gaussian(
                dp["z"], rounds, dp["delta"]).get("per_round_eps") or []
        except Exception:
            eps_curve = None

    best_sel = None          # ((val_acc, val_macro_f1), round, weights)
    for rnd in range(1, rounds + 1):
        global_np = weights
        g_send = list(global_np[shared_slice])
        client_ws, client_ns, client_ids = [], [], []
        honest_deltas = []
        pending_adv = []
        up_times = []
        for k, idx in enumerate(client_indices):
            if len(idx) == 0:
                continue
            is_adv = k in adv
            cm = _init_global(arch, Xtr.shape[1] if Xtr.ndim == 2 else None,
                              Xtr.shape[1:], n_class, device, seed * 1000 + k)
            models.set_weights(cm, global_np)
            y_client = ytr[idx]
            if labelflip and is_adv:
                y_client = fl.flip_labels(y_client, rng, n_class)
            data_cl = LocalData(Xtr[idx], y_client, device)
            gt = ([torch.as_tensor(w, dtype=torch.float32, device=device)
                   for w in global_np] if mu > 0 else None)
            if not (is_adv and attack_deferred):
                t0 = time.time()
                train_local(cm, data_cl, local_epochs, batch,
                            lr_at(rnd, rounds, lr), device, mu=mu,
                            global_torch=gt, freeze_head=use_fedper,
                            grad_clip=grad_clip, seed=seed * 100 + rnd * 7 + k,
                            class_weights=(class_weights_of(y_client, n_class)
                                           if config.CLASS_WEIGHTS else None))
                up_times.append(time.time() - t0)
            w_k = models.get_weights(cm)
            delta_full = [np.asarray(a) - np.asarray(b) for a, b in zip(w_k, global_np)]
            if is_adv and attack_update != "none":
                pending_adv.append((k, delta_full, w_k, int(len(idx))))
            else:
                honest_deltas.append(delta_full)
                # FedPer contributes only the shared part
                w_send = list(w_k[shared_slice]) if n_head else w_k
                client_ws.append(w_send)
                client_ids.append(k)
                client_ns.append(int(len(idx)))
            del cm
        for k, delta_full, w_k, n_k in pending_adv:
            delta_full = fl.attack_update(delta_full, global_np, attack_update, rng,
                                         honest_deltas=honest_deltas or None)
            w_k = [np.asarray(g) + np.asarray(d) for g, d in zip(global_np, delta_full)]
            w_send = list(w_k[shared_slice]) if n_head else w_k
            client_ws.append(w_send)
            client_ids.append(k)
            client_ns.append(n_k)
            del w_k

        if not client_ws:
            break

        dp_diag = None
        if dp is not None:
            est_released = None
            if adaptive is not None:
                norms = [fl.l2_norm([np.asarray(w) - np.asarray(g)
                                     for w, g in zip(cw, g_send)])
                         for cw in client_ws]
                C_now = adaptive.update(norms, rng)
                est_released = bool(adaptive.history[-1].get("released"))
            else:
                C_now = float(dp["C"])
            new_shared, dp_diag = fl.dp_aggregate(g_send, client_ws, C_now, dp["z"], rng,
                                                  return_diag=True,
                                                  sparsity=dp.get("sparsity"),
                                                  head_tensors=dp.get("head_tensors"))
            clip_history.append(dict(round=rnd, C=float(C_now),
                                     clip_fraction=dp_diag["clip_fraction"],
                                     sigma=dp_diag["sigma"],
                                     sparsity=dp_diag.get("sparsity"),
                                     support=dp_diag.get("support"),
                                     estimator_released=est_released,
                                     noise_coords=dp_diag.get("noise_coords"),
                                     total_coords=dp_diag.get("total_coords"),
                                     signal_energy_fraction=dp_diag.get(
                                         "signal_energy_fraction"),
                                     release_signal=dp_diag["release_signal_norm"],
                                     release_noise=dp_diag["release_noise_norm"],
                                     release_snr=dp_diag["release_snr"],
                                     update_norms=[float(x) for x in dp_diag["update_norms"]]))
        else:
            new_shared = fl.aggregate_plain(client_ws, client_ns, how=aggreg,
                                            assumed_f=config.BYZ_ASSUMED_F)

        if server_momentum:
            delta_agg = [np.asarray(a) - np.asarray(b) for a, b in zip(new_shared, g_send)]
            if vel is None:
                vel = [np.zeros_like(d) for d in delta_agg]
            vel = [config.MOMENTUM_BETA * v + d for v, d in zip(vel, delta_agg)]
            new_shared = [np.asarray(g) + config.MOMENTUM_ETA_S * v
                          for g, v in zip(g_send, vel)]

        if n_head:
            # keep each client's private head locally; global carries the shared body only
            merged = list(new_shared) + list(weights[shared_slice.stop:])
        else:
            merged = list(new_shared)
        weights = [np.asarray(m, dtype=np.float32) for m in merged]
        models.set_weights(gm, weights)
        if dp is not None and int(dp.get("iter_avg") or 0) > 0:
            iter_snap.append([np.asarray(w, dtype=np.float64) for w in weights])
            while len(iter_snap) > int(dp["iter_avg"]):
                iter_snap.pop(0)

        pred, prob, loss = evaluate(gm, data_te)
        m = classification_metrics(yte, pred, prob, loss, n_class)
        val_acc = None
        val_f1 = None
        if data_val is not None and len(data_val):
            pv, _, lv = evaluate(gm, data_val)
            val_acc = float(np.mean(pv == np.asarray(yval)))
            if lv is not None and len(pv):
                val_f1 = classification_metrics(yval, pv, None, lv,
                                                n_class)["f1_macro"]
        if config.SELECT_BEST_VAL and val_acc is not None:
            key = (val_acc, val_f1 if val_f1 is not None else -1.0)
            if best_sel is None or key > best_sel[0]:
                best_sel = (key, rnd, [np.asarray(w).copy() for w in weights])
        ch = clip_history[-1] if clip_history else {}
        per_round.append(dict(round=rnd, test_acc=m["accuracy"], test_loss=m["loss"],
                              val_acc=val_acc, val_macro_f1=val_f1,
                              macro_f1=m["f1_macro"], f1_weighted=m["f1_weighted"],
                              precision=m["precision"], recall=m["recall"],
                              balanced_accuracy=m["balanced_accuracy"],
                              macro_specificity=m["macro_specificity"],
                              kappa=m["kappa"], roc_auc_ovr=m.get("roc_auc_ovr"),
                              # how many clients actually contributed this round: K is in
                              # the noise scale, so a round with dropouts is not the round
                              # the certificate assumes
                              n_clients=len(client_ws),
                              eps_cum=(_eps_value(eps_curve[rnd - 1]) if eps_curve
                                       and rnd - 1 < len(eps_curve) else None),
                              C=ch.get("C"), sigma=ch.get("sigma"),
                              clip_fraction=ch.get("clip_fraction"),
                              release_snr=ch.get("release_snr"),
                              wall_s=time.time() - t_start,
                              local_epoch_time_s=float(np.mean(up_times)) if up_times else 0.0))
        # config.PRINT_ROUNDS is the one global quiet switch: it gates only this line, every
        # value on it is already in the per-round record above, which reaches the logs
        # whatever the setting.
        if verbose and config.PRINT_ROUNDS and (
                rnd == 1 or rnd % max(1, rounds // 5) == 0 or rnd == rounds):
            # The live reading is the full metric set, not accuracy alone: a run that looks
            # flat on accuracy can still be losing precision or collapsing to one class,
            # and on Kaggle that is the only view of the run before the tables exist.
            line = ("    R%3d/%d K=%d  acc=%.4f prec=%.4f rec=%.4f F1w=%.4f "
                    "F1m=%.4f bal=%.4f spec=%s kappa=%.4f loss=%.4f val=%s"
                    % (rnd, rounds, len(client_ws), m["accuracy"], m["precision"],
                       m["recall"], m["f1_weighted"], m["f1_macro"],
                       m["balanced_accuracy"],
                       "n/a" if m["macro_specificity"] is None else "%.4f" % m["macro_specificity"],
                       m["kappa"], m["loss"],
                       "n/a" if val_acc is None else "%.4f" % val_acc))
            auc = m.get("roc_auc_ovr")
            if auc is not None:
                line += " auc=%.4f" % auc
            if ch:
                snr = ch.get("release_snr")
                eps_c = per_round[-1]["eps_cum"]
                line += ("  eps=%s C=%.3f sigma=%.3f clipfrac=%.3f snr=%s"
                         % ("n/a" if eps_c is None else "%.4f" % eps_c, float(ch["C"]),
                            float(ch["sigma"]), float(ch["clip_fraction"]),
                            "n/a" if snr is None else "%.4f" % snr))
            print(line)

    wall = time.time() - t_start
    if not per_round:
        # no client contributed in any round: still report the initialised global model
        pred, prob, loss = evaluate(gm, data_te)
        m = classification_metrics(yte, pred, prob, loss, n_class)
        per_round.append(dict(round=0, test_acc=m["accuracy"], test_loss=m["loss"],
                              val_acc=None, val_macro_f1=None, macro_f1=m["f1_macro"],
                              f1_weighted=m["f1_weighted"], precision=m["precision"],
                              recall=m["recall"],
                              balanced_accuracy=m["balanced_accuracy"],
                              macro_specificity=m["macro_specificity"], kappa=m["kappa"],
                              roc_auc_ovr=m.get("roc_auc_ovr"), n_clients=0, eps_cum=None,
                              C=None, sigma=None, clip_fraction=None, release_snr=None,
                              wall_s=wall, local_epoch_time_s=0.0))
    privacy = None
    iter_avg = None
    if dp is not None and len(iter_snap) >= 2:
        # POST-PROCESSING, not a second mechanism: every snapshot here is a deterministic
        # function of releases the run already made, so the privacy loss accounting is
        # unchanged by the post-processing theorem - while averaging over the window
        # cancels part of the fresh per-round noise. Both numbers are reported; the
        # headline `accuracy` stays the final model, so nothing is silently swapped.
        avg_w = [np.mean([s[i] for s in iter_snap], axis=0)
                 for i in range(len(weights))]
        models.set_weights(gm, [np.asarray(a, dtype=np.float32) for a in avg_w])
        p2, q2, l2 = evaluate(gm, data_te)
        m2 = classification_metrics(yte, p2, q2, l2, n_class)
        iter_avg = dict(window=len(iter_snap), accuracy=m2["accuracy"],
                        f1=m2["f1_weighted"], f1_macro=m2["f1_macro"], loss=m2["loss"])
        models.set_weights(gm, weights)
    if dp is not None:
        T = int(dp.get("T", rounds))
        if adaptive is not None:
            # only the releases this run made are charged for; rounds_released is read
            # back from the clipper's own history, so a frozen estimator cannot quietly
            # keep spending or keep releasing while claiming to have stopped
            privacy = dp_accountant.account_adaptive(dp["z"], T, dp["delta"],
                                                     sigma_b=adaptive.sigma_b,
                                                     T_clip=adaptive.rounds_released)
            privacy["freeze_rounds"] = adaptive.freeze_rounds
        else:
            privacy = dp_accountant.account_central_gaussian(dp["z"], T, dp["delta"])
        privacy.update(C=float(dp.get("C", config.CLIP_C)), K=len(client_ws),
                       adaptive=adaptive is not None,
                       sigma=dp_accountant.sigma_of(dp["z"], privacy.get("C", config.CLIP_C),
                                                    max(1, len(client_ws))))

    final = per_round[-1] if per_round else dict(test_acc=0.0, test_loss=float("nan"))
    test_acc_final_round = final.get("test_acc")
    selected_round = final.get("round")
    best_val_acc = None
    if best_sel is not None:
        # Model selection on the VALIDATION subjects only; the test set is read once, for
        # the selected model. The last-round test accuracy stays on the row beside it.
        best_val_acc = float(best_sel[0][0])
        selected_round = int(best_sel[1])
        models.set_weights(gm, [np.asarray(w, dtype=np.float32) for w in best_sel[2]])
        pred, prob, loss = evaluate(gm, data_te)
        m = classification_metrics(yte, pred, prob, loss, n_class)
        final = dict(final, test_acc=m["accuracy"], test_loss=m["loss"])
        models.set_weights(gm, weights)
    out = dict(label=label, seed=int(seed), partition_id=int(partition_id),
               architecture=arch.upper(), method=method,
               rounds=int(rounds), local_epochs=int(local_epochs), batch=int(batch),
               lr=float(lr), K=len(client_indices),
               accuracy=final["test_acc"], f1=m["f1_weighted"], f1_macro=m["f1_macro"],
               test_acc_final_round=test_acc_final_round,
               selected_round=selected_round, best_val_acc=best_val_acc,
               model_selection=("best validation round" if best_sel is not None
                                else "last round"),
               lr_schedule=config.LR_SCHEDULE, lr_final=lr_at(rounds, rounds, lr),
               class_weights=config.CLASS_WEIGHTS, augment=bool(config.AUGMENT),
               precision=m["precision"], recall=m["recall"],
               f1_per_class=m["f1_per_class"], precision_per_class=m["precision_per_class"],
               recall_per_class=m["recall_per_class"], confusion=m["confusion"],
               # the class-balanced set IEEE Access reviewers ask for alongside accuracy:
               # a rare activity class can be invisible in accuracy and cannot hide here.
               balanced_accuracy=m["balanced_accuracy"],
               macro_specificity=m["macro_specificity"],
               specificity_per_class=m["specificity_per_class"], kappa=m["kappa"],
               roc_auc_ovr=m.get("roc_auc_ovr"), nll=m.get("nll"),
               n_test=m["n_test"], class_support=m["class_support"],
               y_pred=pred, y_proba=prob, y_true=np.asarray(yte),
               per_round=per_round, clip_history=clip_history,
               privacy=privacy, wall_time_s=wall,
               client_sizes=[int(n) for n in client_ns],
               client_subjects=None, partition_heterogeneity=heterogeneity,
               attack=(attacks or {}).get("update") if attacks else None,
               attack_fraction=(attacks or {}).get("fraction") if attacks else 0.0,
               aggregator=aggreg, mu=float(mu), fedper=bool(use_fedper),
               server_momentum=bool(server_momentum),
               communication_payload_bytes=_payload_bytes(weights, len(client_indices), rounds),
               local_epoch_time_s=float(np.mean([p["local_epoch_time_s"] for p in per_round]))
               if per_round else 0.0)
    if dp is not None:
        out["clip_C"] = float(dp.get("C", config.CLIP_C))
        out["dp_z"] = float(dp["z"])
        out["dp_delta"] = float(dp["delta"])
        # Which of the four eps-neutral knobs this cell used, per row: a table that shows
        # an improved DP arm without showing the knob that improved it is not evidence.
        out["dp_sparsity"] = dp.get("sparsity")
        out["dp_head_only"] = bool(dp.get("head_only"))
        out["dp_head_tensors"] = dp.get("head_tensors")
        out["dp_iter_avg_window"] = int(dp.get("iter_avg") or 0)
        out["clip_basis"] = ("CLIP_REL x ||w0||" if dp.get("init_norm") is not None
                             else "absolute")
        out["init_weight_norm"] = dp.get("init_norm")
        # which released object picked the coordinates, so a reader can tell the two
        # sparse variants apart instead of trusting a method name
        out["dp_support"] = (clip_history[-1].get("support") if clip_history else None)
        # ... and what the sparse release cost in signal: the fraction of the clipped
        # mean's L2 energy that survived, averaged over the rounds. It lives in
        # clip_history, which the CSV writer flattens away, so the headline number goes
        # on the record.
        sef = [c.get("signal_energy_fraction") for c in clip_history
               if c.get("signal_energy_fraction") is not None]
        if sef:
            out["signal_energy_fraction"] = float(sum(sef) / len(sef))
        last = clip_history[-1] if clip_history else {}
        out["noise_coords"] = last.get("noise_coords")
        out["total_coords"] = last.get("total_coords")
        # The privacy-utility diagnostic, measured rather than predicted: the L2 of the
        # clipped mean update against the L2 of the Gaussian actually drawn over the
        # released support, averaged over the rounds that ran. A chance-level DP row is only
        # a result if the table can say how far below one its release sat, and the ratio is
        # read off the drawn noise so the head-only and sparse arms report their own support.
        hist = [c for c in clip_history if c.get("release_snr") is not None]
        if hist:
            n = float(len(hist))
            out["release_snr_mean"] = float(sum(c["release_snr"] for c in hist) / n)
            out["release_snr_first"] = float(hist[0]["release_snr"])
            out["release_snr_last"] = float(hist[-1]["release_snr"])
            out["release_signal_norm"] = float(sum(c["release_signal"] for c in hist) / n)
            out["release_noise_norm"] = float(sum(c["release_noise"] for c in hist) / n)
        if iter_avg is not None:
            out["accuracy_iter_avg"] = iter_avg["accuracy"]
            out["f1_iter_avg"] = iter_avg["f1"]
            out["f1_macro_iter_avg"] = iter_avg["f1_macro"]
            out["iter_avg_window"] = iter_avg["window"]
    if extra_config:
        out["extra_config"] = extra_config
    if keep_model:
        # the MIA protocol needs per-example losses on sets the run never evaluated
        out["model"] = gm
        out["global_weights"] = weights
    return out


def _payload_bytes(weights, K, rounds):
    """Uplink+downlink float32 bytes moved between server and clients."""
    numel = int(sum(np.asarray(w).size for w in weights))
    return int(numel * 4 * (K + 1) * int(rounds))


def central_epochs(arch, epochs=None):
    """The pooled baseline's epoch budget, in one place so the log cannot say 0."""
    C_ = config.cfg()
    if epochs:
        return int(epochs)
    if arch.upper().startswith("MLP"):
        return int(C_["local_epochs_mlp"] * C_["rounds"])
    return int(max(20, C_["local_epochs_cnn"] * 4))


def run_central(Xtr, ytr, Xte, yte, arch, seed, epochs=None, batch=None, lr=None,
               device=None, label="Centralised", feature_dim=None, checkpoints=10):
    """Same model, same optimiser, pooled training data - the matched control."""
    config.check_budget()
    device = device or models.device_for(True)
    epochs = central_epochs(arch, epochs)
    data_tr = LocalData(Xtr, ytr, device)
    data_te = LocalData(Xte, yte, device)
    m = _init_global(arch, Xtr.shape[1] if Xtr.ndim == 2 else None, Xtr.shape[1:], None,
                     device, seed)
    traj = []

    def _at(ep):
        p, pb, ls = evaluate(m, data_te)
        met = classification_metrics(yte, p, pb, ls, models.n_class_of(m))
        traj.append(dict(round=int(ep), test_acc=met["accuracy"],
                         test_loss=met["loss"], val_acc=None,
                         macro_f1=met["f1_macro"], wall_s=None, local_epoch_time_s=None))

    t0 = time.time()
    train_local(m, data_tr, epochs, batch or config.BATCH, lr or config.LR, device,
                seed=seed, grad_clip=5.0, cosine_epochs=True,
                class_weights=(class_weights_of(ytr, models.n_class_of(m))
                               if config.CLASS_WEIGHTS else None),
                eval_every=max(1, epochs // max(1, int(checkpoints))), eval_fn=_at)
    pred, prob, loss = evaluate(m, data_te)
    met = classification_metrics(yte, pred, prob, loss, models.n_class_of(m))
    wall = time.time() - t0
    for point in traj:
        point["wall_s"] = wall
    return dict(label=label, seed=int(seed), architecture=arch.upper(), rounds=0,
                accuracy=met["accuracy"], f1=met["f1_weighted"], f1_macro=met["f1_macro"],
                precision=met["precision"], recall=met["recall"],
                f1_per_class=met["f1_per_class"], precision_per_class=met["precision_per_class"],
                recall_per_class=met["recall_per_class"], confusion=met["confusion"],
                # same headline set as run_fl, so the centralised control sits in the same
                # columns of the metric table instead of showing blanks
                balanced_accuracy=met["balanced_accuracy"],
                macro_specificity=met["macro_specificity"],
                specificity_per_class=met["specificity_per_class"], kappa=met["kappa"],
                roc_auc_ovr=met.get("roc_auc_ovr"), nll=met.get("nll"),
                n_test=met["n_test"], class_support=met["class_support"],

                y_pred=pred, y_proba=prob, y_true=np.asarray(yte),
                per_round=traj, wall_time_s=wall,
                privacy=None, K=1, local_epochs=int(epochs), method="centralised",
                batch=int(batch or config.BATCH), lr=float(lr or config.LR))
