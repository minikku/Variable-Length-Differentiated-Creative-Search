"""Fitness function (port of ``TrainFunctionsInfo_d`` / ``CostFunction``).

The optimiser minimises a training-set MCC loss plus a hidden-node complexity
penalty:

    o = mcc_loss(tTrain, forward(net, xTrain)) * 100 + ((H - H_min)/(H_max - H_min)) * 100

where ``H`` is the network's hidden size.  This matches ``Evaluator`` inside
``TrainFunctionsInfo_d.m``.  The complexity divisor uses the configured
min/max hidden sizes.

When ``min_hidden == max_hidden`` (a fixed-H baseline such as Study VI / B3),
the divisor is zero and the penalty is logically zero for every candidate.
``_penalty`` handles that case explicitly so the evaluator does not raise
``ZeroDivisionError``.
"""
from __future__ import annotations

import numpy as np
from .network import Network
from .mlp import forward, mcc_loss_multiclass, accuracy, mse


class Evaluator:
    """Callable fitness wrapper bound to one train/val fold."""

    def __init__(self, x_train, t_train, x_val, t_val,
                 min_hidden: int, max_hidden: int):
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        self.min_hidden = min_hidden
        self.max_hidden = max_hidden
        self._denom = float(max_hidden - min_hidden)

    def _penalty(self, h: int) -> float:
        """Hidden-size complexity penalty, robust to a collapsed search range.

        When ``min_hidden == max_hidden`` every candidate has the same H, so
        the per-architecture penalty is identically zero.  Returning 0 in
        that case avoids the ``ZeroDivisionError`` a literal
        ``(h - min) / (max - min)`` would raise.
        """
        if self._denom <= 0.0:
            return 0.0
        return ((h - self.min_hidden) / self._denom) * 100.0

    def __call__(self, net: Network):
        """Return ``(fitness, acc_train, mse_train)`` for a candidate network."""
        pred = forward(net, self.x_train)
        loss = mcc_loss_multiclass(self.t_train, pred)
        h = net.hidden_size
        penalty = self._penalty(h)
        fitness = loss * 100.0 + penalty
        acc = accuracy(pred, self.t_train)
        m = mse(pred, self.t_train)
        return fitness, acc, m

    def ablation(self, net: Network):
        """Return ``(fitness, acc, mse, mcc, hidden)`` for DCS_VSE_hist tracking."""
        pred = forward(net, self.x_train)
        loss = mcc_loss_multiclass(self.t_train, pred)
        h = net.hidden_size
        penalty = self._penalty(h)
        fitness = loss * 100.0 + penalty
        acc = accuracy(pred, self.t_train)
        m = mse(pred, self.t_train)
        mcc = 1.0 - loss  # mean per-class soft-MCC
        return fitness, acc, m, mcc, float(h)
