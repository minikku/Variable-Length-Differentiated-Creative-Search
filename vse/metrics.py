"""Held-out evaluation metrics computed from the best network on a fold.

The original ``Main_Parallel.m`` only stored the optimiser fitness for the
``*_d`` group.  Here we additionally evaluate the best network on the held-out
validation fold and record F1 (macro), AUC (macro one-vs-rest) and accuracy --
the quantities requested for downstream analysis.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from .network import Network
from .mlp import forward


def evaluate_on_fold(net: Network, x_val: np.ndarray, t_val_onehot: np.ndarray) -> dict:
    """Compute classification metrics for ``net`` on the validation fold."""
    pred = forward(net, x_val)                    # (n, n_classes) sigmoid scores
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(t_val_onehot, axis=1)
    n_classes = t_val_onehot.shape[1]

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUC: normalise the per-sample sigmoid scores into a proper score matrix.
    scores = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-12, None)
    auc = _safe_auc(y_true, scores, n_classes)

    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "auc": auc,
        "hidden_nodes": int(net.hidden_size),
    }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray, n_classes: int):
    """ROC-AUC that degrades gracefully when a fold misses a class."""
    present = np.unique(y_true)
    if present.size < 2:
        return float("nan")
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, scores[:, 1]))
        return float(
            roc_auc_score(
                y_true, scores, multi_class="ovr", average="macro",
                labels=np.arange(n_classes),
            )
        )
    except ValueError:
        # Restrict to classes actually present in this fold.
        try:
            return float(
                roc_auc_score(
                    y_true, scores[:, present], multi_class="ovr",
                    average="macro", labels=present,
                )
            )
        except ValueError:
            return float("nan")
