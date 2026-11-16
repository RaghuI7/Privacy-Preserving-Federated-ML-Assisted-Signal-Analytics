"""Metric helpers shared by every experiment."""

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)

from . import config


def _specificity(conf):
    """One-vs-rest specificity per class, straight from the confusion matrix.

    TP/FP/FN/TN are read off the matrix rather than recomputed from the labels, so the
    number on the row is the same reading the figure and the table show. A class nobody
    predicted and nobody had gives 0/0; those cells are reported as None, not as 1.0,
    because a perfect-looking score that came from an empty class is the kind of number
    a reviewer takes apart.
    """
    conf = np.asarray(conf, dtype=float)
    n_total = float(conf.sum())
    out = []
    for i in range(conf.shape[0]):
        tp = float(conf[i, i])
        fp = float(conf[:, i].sum() - tp)
        fn = float(conf[i, :].sum() - tp)
        tn = n_total - tp - fp - fn
        denom = tn + fp
        out.append(float(tn / denom) if denom > 0 else None)
    return out


def classification_metrics(y_true, y_pred, y_proba=None, sample_loss=None,
                           n_class=None):
    n_class = n_class if n_class is not None else config.N_CLASS
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(n_class))
    conf = [[int(v) for v in row]
            for row in confusion_matrix(y_true, y_pred, labels=labels)]
    spec = _specificity(conf)
    spec_vals = [s for s in spec if s is not None]
    recall_pc = [float(x) for x in recall_score(y_true, y_pred, average=None,
                                                labels=labels, zero_division=0)]
    # Macro averages are taken over the classes that occur in this test set: a class with
    # no test windows (possible with a 2-subject PAMAP2 test split) has no recall to
    # average and must not enter as a zero. Per-class arrays keep every class.
    present = [c for c in labels if np.any(y_true == c)] or labels
    out = dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted",
                                   labels=labels, zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", labels=present,
                                zero_division=0)),
        precision=float(precision_score(y_true, y_pred, average="weighted",
                                        labels=labels, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", labels=labels,
                                  zero_division=0)),
        f1_per_class=[float(x) for x in f1_score(y_true, y_pred, average=None,
                                                 labels=labels, zero_division=0)],
        precision_per_class=[float(x) for x in precision_score(y_true, y_pred, average=None,
                                                               labels=labels, zero_division=0)],
        recall_per_class=recall_pc,
        # balanced accuracy is the mean of the per-class recalls: the class-count honest
        # version of accuracy, which matters when a HAR or PAMAP2 class is rare.
        balanced_accuracy=(float(np.mean([recall_pc[c] for c in present]))
                           if recall_pc else None),
        n_classes_in_test=len(present),
        specificity_per_class=spec,
        macro_specificity=float(np.mean(spec_vals)) if spec_vals else None,
        kappa=float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        n_test=int(len(y_true)),
        class_support=[int(np.sum(y_true == i)) for i in labels],
        confusion=conf,
        loss=float(np.mean(sample_loss)) if sample_loss is not None and
        len(np.asarray(sample_loss)) else float("nan"),
    )
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(
                np.eye(n_class)[y_true][:, present], np.asarray(y_proba)[:, present],
                average="macro"))
        except (ValueError, IndexError):
            out["roc_auc_ovr"] = None
        try:
            out["nll"] = float(log_loss(y_true, np.clip(y_proba, 1e-9, 1.0),
                                        labels=labels))
        except (ValueError, IndexError):
            out["nll"] = None
    return out


def per_round_arrays(per_round):
    if not per_round:
        return []
    return [float(p["test_acc"]) for p in per_round]


def _fmt(v, spec="%.4f"):
    return "n/a" if v is None else (spec % v)


def format_metrics(res):
    """One console line with every headline reading of a finished run.

    Used by the experiment stages so that a run's console line, its JSONL record and its
    table row all carry the same set. Nothing is computed here - it reads the result the
    loop already produced, so the line cannot disagree with the log.
    """
    parts = ["acc=%s" % _fmt(res.get("accuracy")),
             "prec=%s" % _fmt(res.get("precision")),
             "rec=%s" % _fmt(res.get("recall")),
             "F1w=%s" % _fmt(res.get("f1")),
             "F1m=%s" % _fmt(res.get("f1_macro")),
             "bal=%s" % _fmt(res.get("balanced_accuracy")),
             "spec=%s" % _fmt(res.get("macro_specificity")),
             "kappa=%s" % _fmt(res.get("kappa"))]
    auc, nll = res.get("roc_auc_ovr"), res.get("nll")
    if auc is not None:
        parts.append("auc=%s" % _fmt(auc))
    if nll is not None:
        parts.append("nll=%s" % _fmt(nll, "%.4f"))
    priv = res.get("privacy") or {}
    if priv:
        parts.append("eps=%s" % _fmt(priv.get("eps"), "%.4f"))
        parts.append("alpha*=%s" % _fmt(priv.get("alpha_star"), "%.2f"))
        parts.append("z=%s C=%s K=%s T=%s"
                     % (_fmt(res.get("dp_z"), "%.4g"), _fmt(res.get("clip_C"), "%.3f"),
                        _fmt(res.get("K"), "%d"), _fmt(res.get("rounds"), "%d")))
        snr = res.get("release_snr_mean")
        parts.append("snr=%s" % _fmt(snr, "%.4f"))
        if priv.get("adjacency"):
            parts.append("adj=%s" % priv["adjacency"])
    if res.get("accuracy_iter_avg") is not None:
        parts.append("acc_avg=%s" % _fmt(res.get("accuracy_iter_avg")))
    parts.append("wall=%.1fs" % float(res.get("wall_time_s") or 0.0))
    return "  ".join(parts)
