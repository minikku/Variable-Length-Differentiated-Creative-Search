"""CoDE_d -- Composite Differential Evolution with VSE (port of CoDE_d.m).

Wang, Cai & Zhang (2011), IEEE TEC 15(1):55-66.
"""
from __future__ import annotations

import math
import numpy as np

from ..network import (
    initial_single_layer_network, random_hidden_size,
    network_to_vector, vector_to_network,
)
from .base import Individual, AlgoResult, pop_sort


def _reflect(v, lb, ub):
    """Boundary reflection used by CoDE (matches the vioLow/vioUpper logic)."""
    v = v.copy()
    low = v < lb
    v[low] = 2 * lb - v[low]
    v[low & (v > ub)] = ub
    high = v > ub
    v[high] = 2 * ub - v[high]
    v[high & (v < lb)] = lb
    return v


def _pick_distinct(rng, popsize, i, k):
    """MATLAB-style draw of k distinct indices in 1..popsize excluding i."""
    idxset = [j for j in range(1, popsize + 1) if j != i]
    chosen = []
    for _ in range(k):
        t = int(math.floor(rng.random() * len(idxset)))
        chosen.append(idxset.pop(t))
    return chosen


def _generator(p, i, F, CR, popsize, options, rng):
    lb, ub = float(options["lb"]), float(options["ub"])
    para = [int(math.floor(rng.random() * 3)) for _ in range(3)]  # 0-based 0..2
    trials = [None, None, None]

    veci, meta_i = network_to_vector(p[i - 1].chrom)

    # ---- strategy 1: rand/1/bin ----
    idx = _pick_distinct(rng, popsize, i, 3)
    v1vecs = [network_to_vector(p[j - 1].chrom)[0] for j in idx]
    n = min(v1vecs[0].size, v1vecs[1].size, v1vecs[2].size)
    v1 = v1vecs[0][:n] + F[para[0]] * (v1vecs[1][:n] - v1vecs[2][:n])
    v1 = _reflect(v1, lb, ub)
    j_rand = int(math.floor(rng.random() * n)) + 1
    t = rng.random(n) < CR[para[0]]
    t[j_rand - 1] = True
    u_tmp = veci.copy()
    m = min(n, u_tmp.size)
    sel = np.where(t[:m])[0]
    u_tmp[sel] = v1[sel]
    trials[0] = vector_to_network(u_tmp, meta_i)

    # ---- strategy 2: current-to-rand/1 ----
    idx = [int(math.floor(rng.random() * popsize)) + 1 for _ in range(3)]
    vi, _ = network_to_vector(p[i - 1].chrom)
    vs = [network_to_vector(p[j - 1].chrom)[0] for j in idx]
    n = min(vi.size, vs[0].size, vs[1].size, vs[2].size)
    rand_scale = rng.random()
    v2 = (vi[:n] + rand_scale * (vs[0][:n] - vi[:n])
          + F[para[1]] * (vs[1][:n] - vs[2][:n]))
    v2 = _reflect(v2, lb, ub)
    u_tmp = veci.copy()
    m = min(n, u_tmp.size)
    u_tmp[:m] = v2[:m]
    trials[1] = vector_to_network(u_tmp, meta_i)

    # ---- strategy 3: rand/2/bin ----
    idx = _pick_distinct(rng, popsize, i, 5)
    vs = [network_to_vector(p[j - 1].chrom)[0] for j in idx]
    n = min(vv.size for vv in vs)
    v3 = (vs[0][:n] + rng.random() * (vs[1][:n] - vs[2][:n])
          + F[para[2]] * (vs[3][:n] - vs[4][:n]))
    v3 = _reflect(v3, lb, ub)
    j_rand = int(math.floor(rng.random() * n)) + 1
    t = rng.random(n) < CR[para[2]]
    t[j_rand - 1] = True
    u_tmp = veci.copy()
    m = min(n, u_tmp.size)
    sel = np.where(t[:m])[0]
    u_tmp[sel] = v3[sel]
    trials[2] = vector_to_network(u_tmp, meta_i)

    return trials


def code_d(options, rng):
    popsize = int(options["popsize"])
    max_nfe = int(options["max_nfe"])
    F = [1.0, 1.0, 0.8]
    CR = [0.1, 0.9, 0.2]
    opts = dict(options)

    p = []
    for _ in range(popsize):
        opts["hidn"] = random_hidden_size(opts, rng)
        chrom = initial_single_layer_network(opts, rng)
        fit, _, _ = options["fobj"](chrom)
        p.append(Individual(chrom, fit))

    FES = popsize
    convergence = []

    while FES < max_nfe:
        p_temp = [Individual(ind.chrom, ind.fitness) for ind in p]
        u_set = []
        for i in range(1, popsize + 1):
            trials = _generator(p, i, F, CR, popsize, opts, rng)
            for tr in trials:
                u_set.append(Individual(tr, np.inf))
            FES += 3

        for ind in u_set:
            ind.fitness, _, _ = options["fobj"](ind.chrom)

        for i in range(1, popsize + 1):
            block = u_set[3 * i - 3: 3 * i]
            fits = [b.fitness for b in block]
            min_id = int(np.argmin(fits))
            best_ind = block[min_id]
            if p[i - 1].fitness >= best_ind.fitness:
                p_temp[i - 1] = Individual(best_ind.chrom, best_ind.fitness)

        p = p_temp
        fits = [ind.fitness for ind in p]
        min_idx = int(np.argmin(fits))
        convergence.append(p[min_idx].fitness)
    fits = [ind.fitness for ind in p]
    min_idx = int(np.argmin(fits))
    return AlgoResult(p[min_idx].fitness, p[min_idx].chrom, convergence)
