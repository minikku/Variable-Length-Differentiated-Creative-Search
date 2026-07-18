"""Forward pass and loss functions (ports of ``MLP_d`` and ``mcc_loss_multiclass``)."""
from __future__ import annotations

import numpy as np
from .network import Network


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def forward(net: Network, x: np.ndarray) -> np.ndarray:
    """Forward pass of the single-hidden-layer MLP.

    ``x`` is (n_samples, input_size); returns predictions (n_samples, output_size).
    Mirrors ``MLP_d`` in "optimizer" mode (weights come straight from the
    chromosome -- no gradient training happens).
    """
    w1 = net.hidden_layer.weight        # (hidn, inp)
    b1 = net.hidden_layer.bias          # (hidn, 1)
    w2 = net.output_layer.weight        # (outp, hidn)
    b2 = net.output_layer.bias          # (outp, 1)
    z1 = w1 @ x.T + b1                   # (hidn, n)
    a1 = _sigmoid(z1)
    z2 = w2 @ a1 + b2                    # (outp, n)
    y_pred = _sigmoid(z2)               # (outp, n)
    return y_pred.T                     # (n, outp)


def accuracy(pred: np.ndarray, target_onehot: np.ndarray) -> float:
    """Top-1 accuracy in percent (matches ``acc_result`` in MATLAB)."""
    idx1 = np.argmax(pred, axis=1)
    idx2 = np.argmax(target_onehot, axis=1)
    return float(np.mean(idx1 == idx2) * 100.0)


def mse(pred: np.ndarray, target_onehot: np.ndarray) -> float:
    return float(np.mean((pred - target_onehot) ** 2))


def mcc_loss_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Soft multi-class MCC loss (port of ``mcc_loss_multiclass``).

    ``y_true`` and ``y_pred`` are (N, M); returns ``1 - mean(MCC_per_class)``.
    """
    inputs = y_pred
    targets = y_true
    n, m = inputs.shape
    tp = np.sum(inputs * targets, axis=0)
    tn = np.sum((1 - inputs) * (1 - targets), axis=0)
    fp = np.sum(inputs * (1 - targets), axis=0)
    fn = np.sum((1 - inputs) * targets, axis=0)
    numerators = tp * tn - fp * fn
    denominators = np.sqrt(
        (tp + fp + 1) * (tp + fn + 1) * (tn + fp + 1) * (tn + fn + 1)
    )
    mcc_values = numerators / (denominators + 1)
    return float(1.0 - np.mean(mcc_values))
