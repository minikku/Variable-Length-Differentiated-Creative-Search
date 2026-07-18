"""DBA_d -- Detective Behavior Algorithm with VSE (port of DBA_d.m).

Cheng & De Waele (2026), Knowledge-Based Systems 338:115434.
"""
from __future__ import annotations

import math
import numpy as np

from ..network import (
    initial_single_layer_network, random_hidden_size,
    network_to_vector, vector_to_network,
)
from ..rng import dba_levy
from .base import Individual, AlgoResult


def _clamp(v, lb, ub):
    return np.where(v > ub, ub, np.where(v < lb, lb, v))


def dba_d(options, rng):
    NP = int(options["popsize"])
    max_nfe = int(options["max_nfe"])
    lb, ub = float(options["lb"]), float(options["ub"])
    Max_It = round(max_nfe / NP)
    opts = dict(options)

    X = []
    for _ in range(NP):
        opts["hidn"] = random_hidden_size(opts, rng)
        chrom = initial_single_layer_network(opts, rng)
        fit, _, _ = options["fobj"](chrom)
        X.append(Individual(chrom, fit))
    pbest = [Individual(ind.chrom, ind.fitness) for ind in X]

    fits = [ind.fitness for ind in X]
    best = Individual(X[int(np.argmin(fits))].chrom, min(fits))

    convergence = []
    it = 0
    nfe = 0
    while nfe < max_nfe:
        it += 1
        for i in range(NP):
            veci, meta_i = network_to_vector(X[i].chrom)
            vecb, _ = network_to_vector(best.chrom)
            n = min(veci.size, vecb.size)
            v_tmp = veci.copy()

            A = (10.0 * rng.random() - 1.0) * math.sin(0.5 * math.pi * it / Max_It)
            if A > 0.5:
                if rng.random() < 0.5:
                    v_tmp[:n] = vecb[:n] - rng.random() * (vecb[:n] - veci[:n])
                else:
                    v_tmp[:n] = vecb[:n] + dba_levy(rng, 1.5, size=n)
            else:
                r = rng.random(n)
                v_tmp[:n] = (ub + lb) * 0.5 + r * (vecb[:n] - veci[:n])

            v_tmp = _clamp(v_tmp, lb, ub)
            cand = Individual(vector_to_network(v_tmp, meta_i), np.inf)
            cand.fitness, _, _ = options["fobj"](cand.chrom)
            nfe += 1

            opp_v = _clamp((ub + lb) - v_tmp, lb, ub)
            opp = Individual(vector_to_network(opp_v, meta_i), np.inf)
            opp.fitness, _, _ = options["fobj"](opp.chrom)
            nfe += 1

            X[i] = opp if cand.fitness > opp.fitness else cand

            if X[i].fitness < pbest[i].fitness:
                pbest[i] = X[i]
                if pbest[i].fitness < best.fitness:
                    best = pbest[i]

        convergence.append(best.fitness)

    return AlgoResult(best.fitness, best.chrom, convergence)
