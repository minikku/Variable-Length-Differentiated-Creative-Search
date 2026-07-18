"""VLPSO_d -- Variable-Length PSO for neural networks (port of VLPSO_d.m).

Based on Tran, Xue & Zhang (2019), adapted to the DCS-VSE framework.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..network import (
    initial_single_layer_network, network_to_vector, vector_to_network,
)
from .base import AlgoResult


@dataclass
class _Best:
    chrom: object = None
    vec: Optional[np.ndarray] = None
    fitness: float = np.inf
    L: int = 0


@dataclass
class _Particle:
    chrom: object = None
    fitness: float = np.inf
    vec: Optional[np.ndarray] = None
    vel: Optional[np.ndarray] = None
    L: int = 0
    opt_x: object = None
    best: _Best = field(default_factory=_Best)
    exemplar: Optional[np.ndarray] = None
    Pc: float = 0.5
    imp_count: int = 0


def _calculate_pc(pop, NP):
    fitnesses = np.array([p.fitness for p in pop])
    order = np.argsort(fitnesses, kind="stable")
    ranks = np.empty(NP, dtype=int)
    ranks[order] = np.arange(1, NP + 1)
    for i in range(NP):
        ri = ranks[i]
        num = math.exp(10 * (ri - 1) / (NP - 1)) - 1
        den = math.exp(10) - 1
        pop[i].Pc = 0.05 + 0.45 * (num / den)


def _select_valid(pop, NP, dim_req, exclude1, exclude2, rng):
    for _ in range(NP):
        kk = int(rng.integers(1, NP + 1))
        if kk != exclude1 and kk != exclude2 and pop[kk - 1].vec.size >= dim_req:
            return kk
    return -1


def _assign_exemplars_single(p, pop, p_idx, NP, rng):
    D = p.vec.size
    p.exemplar = np.zeros(D, dtype=int)
    for d in range(1, D + 1):
        if rng.random() >= p.Pc:
            p.exemplar[d - 1] = p_idx
        else:
            p1 = _select_valid(pop, NP, d, p_idx, -1, rng)
            p2 = _select_valid(pop, NP, d, p_idx, p1, rng)
            if p1 == -1 or p2 == -1:
                p.exemplar[d - 1] = p_idx
            else:
                if pop[p1 - 1].best.fitness < pop[p2 - 1].best.fitness:
                    p.exemplar[d - 1] = p1
                else:
                    p.exemplar[d - 1] = p2


def _assign_exemplars(pop, NP, rng):
    for i in range(NP):
        _assign_exemplars_single(pop[i], pop, i + 1, NP, rng)


def _length_changing(pop, gbest, NbrDiv, NP, options, rng):
    DivSize = NP // NbrDiv
    nfe_added = 0
    div_avg = np.zeros(NbrDiv)
    div_lens = np.zeros(NbrDiv, dtype=int)
    for div in range(NbrDiv):
        s = div * DivSize
        e = min((div + 1) * DivSize, NP)
        subset = pop[s:e]
        div_avg[div] = np.mean([p.fitness for p in subset])
        div_lens[div] = subset[0].L
    best_div = int(np.argmin(div_avg))
    target_h = div_lens[best_div]

    for div in range(NbrDiv):
        if div == best_div:
            continue
        s = div * DivSize
        e = min((div + 1) * DivSize, NP)
        for i in range(s, e):
            current_h = pop[i].L
            if current_h == target_h:
                continue
            local_opts = dict(options)
            local_opts["hidn"] = int(target_h)
            dummy = initial_single_layer_network(local_opts, rng)
            dummy_vec, target_meta = network_to_vector(dummy)
            target_dim = dummy_vec.size
            old_vec = pop[i].vec
            current_dim = old_vec.size
            new_vec = np.zeros(target_dim)
            if current_dim > target_dim:
                new_vec = old_vec[:target_dim].copy()
            else:
                new_vec[:current_dim] = old_vec
                new_vec[current_dim:] = rng.random(target_dim - current_dim) - 0.5
            pop[i].vec = new_vec
            pop[i].vel = np.zeros(target_dim)
            pop[i].L = int(target_h)
            pop[i].opt_x = target_meta
            pop[i].chrom = vector_to_network(new_vec, target_meta)
            pop[i].fitness, _, _ = options["fobj"](pop[i].chrom)
            nfe_added += 1
            pop[i].best = _Best(pop[i].chrom, pop[i].vec.copy(),
                                pop[i].fitness, int(target_h))
            pop[i].imp_count = 0
            if pop[i].fitness < gbest.fitness:
                gbest = _copy_best(pop[i].best)
    return gbest, nfe_added


def _copy_best(b: _Best) -> _Best:
    return _Best(b.chrom, None if b.vec is None else b.vec.copy(), b.fitness, b.L)


def vlpso_d(options, rng):
    NP = int(options["popsize"])
    max_nfe = int(options["max_nfe"])
    NbrDiv = 5
    a = 7
    beta = 10
    w = 0.729
    c = 1.49445
    min_h = int(options["min_hidden_size"])
    max_h = int(options["max_hidden_size"])

    pop = [_Particle() for _ in range(NP)]
    DivSize = NP // NbrDiv
    for div in range(1, NbrDiv + 1):
        current_h = min_h + int(math.floor((max_h - min_h) * (div / NbrDiv)))
        start = (div - 1) * DivSize
        end = div * DivSize
        if div == NbrDiv:
            end = NP
        for i in range(start, end):
            local_opts = dict(options)
            local_opts["hidn"] = current_h
            chrom = initial_single_layer_network(local_opts, rng)
            vec, meta = network_to_vector(chrom)
            p = pop[i]
            p.chrom = chrom
            p.L = current_h
            p.vec = vec
            p.opt_x = meta
            p.vel = np.zeros(vec.size)
            p.fitness, _, _ = options["fobj"](chrom)
            p.best = _Best(chrom, vec.copy(), p.fitness, current_h)

    order = np.argsort([p.fitness for p in pop], kind="stable")
    gbest = _copy_best(pop[int(order[0])].best)

    _calculate_pc(pop, NP)
    _assign_exemplars(pop, NP, rng)

    nfe = NP
    convergence = []
    gbest_stall = 0

    while nfe <= max_nfe:
        renew_flag = False
        for i in range(NP):
            p = pop[i]
            if p.imp_count >= a:
                _assign_exemplars_single(p, pop, i + 1, NP, rng)
                p.imp_count = 0
                renew_flag = True

            D = p.vec.size
            ex = p.exemplar.copy()
            ex[(ex < 1) | (ex > NP)] = i + 1
            # Default target = own pbest (also the fallback when an exemplar is
            # shorter than the current particle at dimension d).
            target = p.best.vec[:D].copy()
            for eid in np.unique(ex):
                pos = np.nonzero(ex == eid)[0]
                bv = pop[eid - 1].best.vec
                valid = pos[pos < bv.size]
                target[valid] = bv[valid]
            r = rng.random(D)
            p.vel = w * p.vel + c * r * (target - p.vec)

            p.vec = p.vec + p.vel
            p.chrom = vector_to_network(p.vec, p.opt_x)
            p.fitness, _, _ = options["fobj"](p.chrom)
            nfe += 1

            if p.fitness < p.best.fitness:
                p.best = _Best(p.chrom, p.vec.copy(), p.fitness, p.L)
                p.imp_count = 0
                if p.fitness < gbest.fitness:
                    gbest = _copy_best(p.best)
                    gbest_stall = 0
            else:
                p.imp_count += 1

        if gbest_stall >= beta:
            gbest, nfe_added = _length_changing(pop, gbest, NbrDiv, NP, options, rng)
            nfe += nfe_added
            _calculate_pc(pop, NP)
            _assign_exemplars(pop, NP, rng)
            gbest_stall = 0
        else:
            gbest_stall += 1

        if renew_flag:
            _calculate_pc(pop, NP)

        convergence.append(gbest.fitness)
        if nfe >= max_nfe:
            break

    return AlgoResult(gbest.fitness, gbest.chrom, convergence)
