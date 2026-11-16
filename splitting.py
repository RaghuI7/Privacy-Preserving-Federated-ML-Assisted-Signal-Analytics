"""Subject-level splitting, subject-silo clients, LOSO and the leakage audit.

This is the module the submission did not have. Every split it produces is
disjoint in SUBJECTS, never in windows, and it records which subject ids landed in
which split so the paper can state exactly how clients were formed (Task 3).
"""

import numpy as np

from . import config


# ── helpers ───────────────────────────────────────────────────────────────────
def _subject_arrays(subjects):
    return np.unique(subjects)


def _class_coverage(y, subjects, subset_subs, n_class=None):
    n_class = n_class if n_class is not None else config.N_CLASS
    mask = np.isin(subjects, subset_subs)
    present = set(int(c) for c in np.unique(y[mask]))
    return present, [c for c in range(n_class) if c not in present]


def _greedy_split(subjects, y, fracs, rng, n_class):
    """Assign whole subjects to splits until the target fractions are met, while
    keeping every class represented in every split when that is achievable."""
    subs = _subject_arrays(subjects)
    counts = {int(s): int(np.sum(subjects == s)) for s in subs}
    order = list(subs)
    rng.shuffle(order)
    tot = float(sum(counts.values()))
    names = ("train", "val", "test")
    target = {n: fracs[i] * tot for i, n in enumerate(names)}
    got = {n: 0 for n in names}
    assigned = {n: [] for n in names}
    cls_have = {n: set() for n in names}
    cls_of = {int(s): set(int(c) for c in y[subjects == s]) for s in subs}

    def deficit(n):
        return (target[n] - got[n]) / max(target[n], 1.0)

    for s in order:
        # prefer the split still missing some class, else the one furthest below target
        needs = [n for n in names if (set(range(n_class)) - cls_have[n]) & cls_of[int(s)]]
        cand = needs or names
        pick = max(cand, key=lambda n: deficit(n) + (0.5 if n in needs else 0.0))
        assigned[pick].append(int(s))
        got[pick] += counts[int(s)]
        cls_have[pick] |= cls_of[int(s)]
    return assigned


def subject_disjoint_split(subjects, y, seed=config.SEEDS[0], fracs=(0.70, 0.10, 0.20),
                           n_class=None):
    """Train/val/test with no subject appearing in two splits.

    Returns dict(train_idx, val_idx, test_idx, subjects_in_split, leakage_checks).
    """
    n_class = n_class if n_class is not None else config.N_CLASS
    subjects = np.asarray(subjects)
    y = np.asarray(y)
    rng = np.random.RandomState(int(seed))
    best = None
    for attempt in range(24):
        r = rng if attempt == 0 else np.random.RandomState(int(seed) + 1000 + attempt)
        assigned = _greedy_split(subjects, y, fracs, r, n_class)
        miss = {n: _class_coverage(y, subjects, assigned[n], n_class)[1] for n in assigned}
        score = sum(len(v) for v in miss.values())
        cand = dict(assigned=assigned, score=score, missing=miss, attempt=attempt)
        if best is None or cand["score"] < best["score"]:
            best = cand
        if best["score"] == 0:
            break
    assigned = best["assigned"]
    idx = {n: np.where(np.isin(subjects, assigned[n]))[0] for n in assigned}
    checks = leakage_report(subjects, idx, kind="subject_disjoint",
                            missing_classes=best["missing"],
                            n_attempts=best["attempt"] + 1, seed=seed)
    return dict(train_idx=idx["train"], val_idx=idx["val"], test_idx=idx["test"],
                subjects_in_split={k: sorted(int(x) for x in v) for k, v in assigned.items()},
                leakage_checks=checks)


def window_level_split(y, seed=config.SEEDS[0], fracs=(0.70, 0.10, 0.20)):
    """The OLD (leaking) split, kept only so the leakage gap can be MEASURED.

    Stratified window-level train_test_split, exactly the construction the reviewers
    objected to. Never used for a reportable result.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    try:
        tmp, te = train_test_split(idx, test_size=fracs[2], stratify=y, random_state=seed)
    except ValueError:                                  # rare class with <2 members
        tmp, te = train_test_split(idx, test_size=fracs[2], random_state=seed)
    vfrac = fracs[1] / (fracs[0] + fracs[1])
    tr, va = train_test_split(tmp, test_size=vfrac, stratify=y[tmp], random_state=seed)
    return dict(train_idx=tr, val_idx=va, test_idx=te)


def subject_leak(subjects, split_idx):
    """How many subjects appear in more than one split (0 is the only acceptable value)."""
    sets = {k: set(int(s) for s in np.asarray(subjects)[v]) for k, v in split_idx.items()}
    names = list(sets)
    dupes = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sets[names[i]] & sets[names[j]]
            if inter:
                dupes.append((names[i], names[j], sorted(inter)))
    n_windows_exposed = 0
    if dupes:
        exposed = set(x for d in dupes for x in d[2])
        n_windows_exposed = int(sum(1 for s in subjects if int(s) in exposed))
    return dict(n_overlapping_pairs=len(dupes),
                overlapping_subjects=[(a, b, len(v)) for a, b, v in dupes],
                windows_from_overlapping_subjects=n_windows_exposed,
                n_subjects=len(sets[names[0]]) if names else 0)


def leakage_report(subjects, split_idx, kind="subject_disjoint", missing_classes=None,
                   n_attempts=None, seed=None):
    """The Work Order's named leakage check (Task 3), for ANY split.

    Both constructions go through this one function, so the zeros of the subject-disjoint
    split and the non-zeros of the old window-level split are the same measurement rather
    than two conventions. Everything returned here is stamped into the log record by
    `harness._split_provenance`, which means a table can show "0 subjects reused" as a
    measured column instead of an assertion.
    """
    full = {n: np.asarray(v) for n, v in split_idx.items()}
    leak = subject_leak(subjects, full)
    sets = {n: set(int(s) for s in np.asarray(subjects)[v]) for n, v in full.items()}

    def _ov(a, b):
        return len(sets.get(a, set()) & sets.get(b, set()))

    out = dict(split_kind=kind,
               subjects_overlap_train_test=_ov("train", "test"),
               subjects_overlap_train_val=_ov("train", "val"),
               subjects_overlap_val_test=_ov("val", "test"),
               n_overlapping_pairs=leak["n_overlapping_pairs"],
               overlapping_subjects=leak["overlapping_subjects"],
               windows_from_overlapping_subjects=
                   leak["windows_from_overlapping_subjects"],
               n_subjects=leak["n_subjects"])
    if missing_classes is not None:
        out["missing_classes"] = missing_classes
    if n_attempts is not None:
        out["n_attempts"] = int(n_attempts)
    if seed is not None:
        out["seed"] = int(seed)
    return out


def to_block_positions(client_indices, block_idx):
    """Translate GLOBAL window ids into positions inside a block of windows.

    Every partition function in this module returns global ids (they index the
    whole bundle), while the runner is handed only one split block - `harness.input()`
    gives it `X[train_idx]`. Feeding global ids straight into that block reads past
    its end or trains on another subject's windows, so the two index spaces must be
    translated at the boundary.

    A window that is not in `block_idx` is a split bug, so this raises rather than
    clamps: silently dropping test windows from a client would invalidate the leak-free
    claim the whole Task 3 rests on.
    """
    block_idx = np.asarray(block_idx, dtype=int)
    pos = np.full(int(block_idx.max()) + 1 if len(block_idx) else 0, -1, dtype=int)
    pos[block_idx] = np.arange(len(block_idx))
    out = []
    for idx in client_indices:
        idx = np.asarray(idx, dtype=int)
        too_big = idx.size and int(idx.max()) >= pos.size
        if too_big:
            raise ValueError("client window id %d is beyond every window in the split "
                             "(largest global id %d) - the partition was built on a "
                             "different split" % (int(idx.max()), pos.size - 1))
        p = pos[idx] if idx.size else idx
        bad = int((p < 0).sum())
        if bad:
            raise ValueError("%d client window(s) are not in the requested split block" % bad)
        out.append(p)
    return out


# ── federated clients ─────────────────────────────────────────────────────────
def federated_clients_by_subject(train_idx, subjects, y=None, n_clients=None,
                                 seed=config.SEEDS[0], mode="grouped"):
    """Build K clients as SUBJECT SILOS.

    mode="one_subject_per_client" : K = number of training subjects.
    mode="grouped"                : training subjects are grouped into K silos,
        balanced by window count (greedy largest-first), so client sizes are even
        while no client ever mixes a subject with another client's subject.

    The returned `client_subjects` is the exact per-client subject id list the paper
    must report.
    """
    n_clients = n_clients if n_clients is not None else config.N_CLIENTS
    train_idx = np.asarray(train_idx)
    subs_tr = np.unique(np.asarray(subjects)[train_idx])
    if mode == "one_subject_per_client":
        groups = [[int(s)] for s in subs_tr]
    else:
        rng = np.random.RandomState(int(seed) + 7)
        order = list(subs_tr)
        rng.shuffle(order)
        sizes = {int(s): int(np.sum(subjects[train_idx] == s)) for s in order}
        groups = [[] for _ in range(int(n_clients))]
        load = np.zeros(int(n_clients))
        for s in sorted(order, key=lambda x: -sizes[int(x)]):
            k = int(np.argmin(load))
            groups[k].append(int(s))
            load[k] += sizes[int(s)]
        groups = [g for g in groups if g]
    out_idx = [train_idx[np.isin(subjects[train_idx], g)] for g in groups]
    keep = [i for i, ix in enumerate(out_idx) if len(ix) > 0]
    out_idx = [out_idx[i] for i in keep]
    groups = [groups[i] for i in keep]
    return dict(client_indices=out_idx, client_subjects=groups,
                client_sizes=[len(ix) for ix in out_idx],
                n_clients=len(out_idx), mode=mode, seed=int(seed),
                privacy_unit="client (= subject silo)",
                adjacency="add/remove all data of one client")


def dirichlet_partition_windows(train_idx, y, n_clients=None, alpha=0.5,
                                seed=config.SEEDS[0]):
    """Label-Dirichlet partition of the TRAINING WINDOWS only (Task 5).

    The rejected code called this on the full dataset including the test set, which
    both leaked and made alpha meaningless. Test windows are never partitioned.
    """
    n_clients = n_clients if n_clients is not None else config.N_CLIENTS
    train_idx = np.asarray(train_idx)
    y_tr = np.asarray(y)[train_idx]
    rng = np.random.RandomState(int(seed))
    classes = np.unique(y_tr)
    buckets = [[] for _ in range(int(n_clients))]
    for c in classes:
        pos = train_idx[y_tr == c]
        rng.shuffle(pos)
        props = rng.dirichlet(np.ones(int(n_clients)) * float(alpha))
        counts = (props * len(pos)).astype(int)
        counts[-1] = max(0, len(pos) - counts[:-1].sum())
        start = 0
        for k, cnt in enumerate(counts):
            buckets[k].extend(pos[start:start + cnt].tolist())
            start += cnt
    out = [np.array(sorted(b), dtype=int) for b in buckets]
    return [o for o in out if len(o) > 0]


def iid_partition_windows(train_idx, n_clients=None, seed=config.SEEDS[0]):
    n_clients = n_clients if n_clients is not None else config.N_CLIENTS
    train_idx = np.asarray(train_idx)
    rng = np.random.RandomState(int(seed))
    parts = np.array_split(rng.permutation(train_idx), int(n_clients))
    return [p for p in parts if len(p) > 0]


# ── heterogeneity measure ─────────────────────────────────────────────────────
def partition_heterogeneity(client_indices, y, n_class=None):
    """Label skew of a partition: mean per-client entropy deficit plus the spread of
    the dominant-class share. Reported alongside every alpha in Task 5."""
    n_class = n_class if n_class is not None else config.N_CLASS
    y = np.asarray(y)
    shares, dom = [], []
    for idx in client_indices:
        if len(idx) == 0:
            continue
        cnt = np.array([np.sum(y[idx] == c) for c in range(n_class)], float)
        p = cnt / cnt.sum()
        ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
        shares.append(ent / np.log(n_class))
        dom.append(float(cnt.max() / cnt.sum()))
    if not shares:
        return dict(normalized_entropy=0.0, dominant_class_share=0.0,
                    dominant_class_spread=0.0, n_clients=0)
    return dict(normalized_entropy=float(np.mean(shares)),
                dominant_class_share=float(np.mean(dom)),
                dominant_class_spread=float(np.std(dom)),
                n_clients=len(shares))


# ── leave-one-subject-out ─────────────────────────────────────────────────────
def loso_folds(subjects, n_folds_max=None, seed=None):
    """One fold per subject: that subject is the test set, the rest is training.

    Returns list of dicts with train_idx/test_idx and the held-out subject id.
    Val is carved from the training SUBJECTS (never from the held-out one).
    """
    subjects = np.asarray(subjects)
    subs = _subject_arrays(subjects)
    folds = []
    for s in subs:
        test_idx = np.where(subjects == s)[0]
        tr_subs = [x for x in subs if x != s]
        val_sub = tr_subs[-1]
        train_idx = np.where(np.isin(subjects, tr_subs[:-1]))[0]
        val_idx = np.where(subjects == val_sub)[0]
        folds.append(dict(holdout_subject=int(s), train_idx=train_idx,
                          val_idx=val_idx, test_idx=test_idx,
                          n_train=len(train_idx), n_test=len(test_idx)))
    if n_folds_max and len(folds) > n_folds_max:
        rng = np.random.RandomState(int(seed or config.SEEDS[0]))
        keep = rng.choice(len(folds), size=int(n_folds_max), replace=False)
        folds = [folds[i] for i in sorted(keep)]
    return folds
