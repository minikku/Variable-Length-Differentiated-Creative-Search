"""Shared structures for the optimisation algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..network import Network


@dataclass
class Individual:
    chrom: Optional[Network] = None
    fitness: float = np.inf


@dataclass
class AlgoResult:
    best_cost: float
    best_x: Network
    convergence_curve: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def pop_sort(pop):
    """Ascending sort of a list of individuals by fitness (stable, like MATLAB)."""
    order = np.argsort([ind.fitness for ind in pop], kind="stable")
    return [pop[i] for i in order]


def base_options(net_inp: int, net_outp: int, options: dict) -> dict:
    """Build the per-call options dict expected by the operators."""
    o = dict(options)
    o["inp"] = net_inp
    o["outp"] = net_outp
    return o
