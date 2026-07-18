"""Optimisation-algorithm registry.

Each algorithm is a callable ``f(options, rng) -> AlgoResult`` where ``options``
is a dict carrying the problem definition (population size, NFE budget, bounds,
hidden-size range, input/output sizes, and the bound ``fobj`` evaluator) and
``rng`` is a ``numpy.random.Generator``.
"""
from __future__ import annotations

from .base import AlgoResult, Individual
from . import dcs, gprop, code, dba, colshade, vlpso, extended

# name -> callable(options, rng) -> AlgoResult
REGISTRY = {}

REGISTRY.update(dcs.build_registry())
REGISTRY["GProp_d"] = gprop.gprop_d
REGISTRY["CoDE_d"] = code.code_d
REGISTRY["DBA_d"] = dba.dba_d
REGISTRY["COLSHADE_d"] = colshade.colshade_d
REGISTRY["VLPSO_d"] = vlpso.vlpso_d
REGISTRY.update(extended.EXTENDED)   # recent metaheuristics via the size-gene adapter


def get_algorithm(name: str):
    if name not in REGISTRY:
        raise KeyError(
            f"unknown algorithm '{name}'. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


__all__ = ["AlgoResult", "Individual", "REGISTRY", "get_algorithm"]
