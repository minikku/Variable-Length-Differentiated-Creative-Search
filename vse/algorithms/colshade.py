"""COLSHADE_d -- LSHADE with Levy flights and VSE (port of COLSHADE_d.m).

Gurrola-Ramos, Hernandez-Aguirre & Dalmau-Cedeno (2020), IEEE CEC.

The original is a *constrained* optimiser, but this neuro-evolution problem is
unconstrained (the MATLAB sets ``gn = hn = 0`` and every constraint-violation
vector is identically zero).  The constraint bookkeeping therefore collapses to
a plain fitness comparison, which is what is implemented here; the adaptive
F/CR memories, Levy/pbest mutation split and external archive are reproduced
faithfully.
"""
from __future__ import annotations

import math
import numpy as np

from ..network import (
    initial_single_layer_network, random_hidden_size,
    network_to_vector, vector_to_network,
)
from ..rng import normal_rng, cauchy_rng, levy_rng
from .base import Individual, AlgoResult, pop_sort


def colshade_d(options, rng):
    NP = int(options["popsize"])
    max_fes = int(options["max_nfe"])
    xmin, xmax = float(options["lb"]), float(options["ub"])
    opts = dict(options)

    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    M_F_l = np.full(H, 0.5)
    M_CR_l = np.full(H, 0.5)
    k = 0
    k_l = 0

    prob_levy = 0.5
    momentum = 0.25
    prob_min = 1e-3
    pbest_rate = 0.11
    r_arc = 2.6
    max_ps = NP
    min_ps = NP

    # --- init population ---
    x = []
    for _ in range(NP):
        opts["hidn"] = random_hidden_size(opts, rng)
        chrom = initial_single_layer_network(opts, rng)
        fit, _, _ = options["fobj"](chrom)
        x.append(Individual(chrom, fit))
    x = pop_sort(x)
    archive = []                       # list of chroms (Networks)

    fes = NP
    ps = NP
    convergence = []

    while fes + ps < max_fes:
        as_ = len(archive)

        # ---- get_index ----
        pbest_cap = int(math.ceil(pbest_rate * ps))
        pbest_idx = rng.integers(1, pbest_cap + 1, size=ps)   # 1-based
        r = np.zeros((ps, 2), dtype=int)
        for i in range(ps):
            while True:
                r1 = int(rng.integers(1, ps + 1))
                if r1 != i + 1:
                    break
            while True:
                r2 = int(rng.integers(1, ps + as_ + 1))
                if r2 != i + 1 and r2 != r1:
                    break
            r[i] = (r1, r2)

        # ---- get_cx_params ----
        F = np.zeros(ps)
        CR = np.zeros(ps)
        pbest_flag = np.zeros(ps, dtype=bool)
        p_i_levy = rng.random(ps)
        for i in range(ps):
            ri = int(rng.integers(0, H))
            if p_i_levy[i] <= prob_levy:
                m_cr, m_f = M_CR_l[ri], M_F_l[ri]
            else:
                m_cr, m_f = M_CR[ri], M_F[ri]
                pbest_flag[i] = True
            if m_cr != -1:
                CR[i] = m_cr + float(normal_rng(rng))
            f_crit = math.sqrt(max((1 - CR[i] / 2) / ps, 0.0))
            guard = 0
            while F[i] <= f_crit and guard < 1000:
                F[i] = m_f + float(cauchy_rng(rng))
                guard += 1
        CR = np.clip(CR, 0, 1)
        F = np.minimum(F, 1.0)

        # ---- mutation + crossover ----
        u = _crossover(x, archive, r, pbest_idx, F, CR, pbest_flag,
                       xmin, xmax, ps, rng)

        f_u = np.empty(ps)
        for i in range(ps):
            u[i].fitness, _, _ = options["fobj"](u[i].chrom)
            f_u[i] = u[i].fitness
        fes += ps

        # ---- deb tournament (unconstrained: greedy with archive) ----
        f = np.array([ind.fitness for ind in x])
        delta_f = np.where(f_u < f, f - f_u, 0.0)
        dmax = delta_f.max()
        if dmax > 0:
            delta_f_norm = delta_f / dmax
        else:
            delta_f_norm = delta_f.copy()
        for i in range(ps):
            if delta_f_norm[i] > 0:
                archive.append(x[i].chrom)
                x[i] = u[i]

        # ---- memory update ----
        _update_memory(M_F, M_CR, M_F_l, M_CR_l, F, CR, pbest_flag,
                       delta_f_norm, k, k_l)

        x = pop_sort(x)

        delta_pbest = delta_f_norm * pbest_flag
        delta_levy = delta_f_norm * (~pbest_flag)
        if np.any(delta_pbest):
            k = (k + 1) % H
        if np.any(delta_levy):
            k_l = (k_l + 1) % H
        if np.any(delta_pbest > 0) or np.any(delta_levy > 0):
            sl = delta_levy.sum()
            sp = delta_pbest.sum()
            if sl + sp > 0:
                prob_levy = momentum * prob_levy + (1 - momentum) * (sl / (sl + sp))
                prob_levy = max(min(prob_levy, 1 - prob_min), prob_min)

        # ---- resize population & archive ----
        ps = round(min_ps + (1 - fes / max_fes) * (max_ps - min_ps))
        ps = max(ps, 4)
        x = x[:ps]
        max_as = round(ps * r_arc)
        if len(archive) > max_as:
            n_remove = len(archive) - max_as
            rm = rng.permutation(len(archive))[:n_remove]
            archive = [a for j, a in enumerate(archive) if j not in set(rm.tolist())]

        convergence.append(x[0].fitness)

    return AlgoResult(x[0].fitness, x[0].chrom, convergence)


def _reflect_bounds(vvi, vxi, xmin, xmax, rng):
    """Bound handling from COLSHADE crossover (zero-out then re-inject)."""
    v_upper = vvi > xmax
    v_lower = vvi < xmin
    vvi = vvi * (~v_upper)
    vvi = vvi * (~v_lower)
    base = 0.1 * rng.random(vvi.shape)
    x_upper = (1 - base) * xmax + base * vxi
    x_lower = (1 - base) * xmin + base * vxi
    vvi = vvi + v_upper * x_upper + v_lower * x_lower
    return vvi


def _crossover(x, archive, r, pbest_idx, F, CR, pbest_flag, xmin, xmax, ps, rng):
    v_list = []
    for i in range(ps):
        veci, meta_i = network_to_vector(x[i].chrom)
        p = pbest_idx[i] - 1
        vecp, _ = network_to_vector(x[p].chrom)
        v_tmp = veci.copy()
        if pbest_flag[i]:
            r1 = r[i, 0] - 1
            r2 = r[i, 1]
            vecr1, _ = network_to_vector(x[r1].chrom)
            if r2 > ps:
                arch_net = archive[r2 - ps - 1]
                vecr2, _ = network_to_vector(arch_net)
            else:
                vecr2, _ = network_to_vector(x[r2 - 1].chrom)
            d = min(veci.size, vecp.size, vecr1.size, vecr2.size)
            v_tmp[:d] = (veci[:d] + F[i] * (vecp[:d] - veci[:d])
                         + F[i] * (vecr1[:d] - vecr2[:d]))
        else:
            d = min(veci.size, vecp.size)
            lr = levy_rng(rng, size=d)
            v_tmp[:d] = veci[:d] + F[i] * lr * (vecp[:d] - veci[:d])
        v_list.append((v_tmp, meta_i))

    # bound handling
    for i in range(ps):
        vxi, _ = network_to_vector(x[i].chrom)
        vvi, meta_v = v_list[i]
        d = min(vxi.size, vvi.size)
        vvi[:d] = _reflect_bounds(vvi[:d], vxi[:d], xmin, xmax, rng)
        v_list[i] = (vvi, meta_v)

    # binomial crossover
    u = []
    for i in range(ps):
        vxi, meta_i = network_to_vector(x[i].chrom)
        vvi, _ = v_list[i]
        d = min(vxi.size, vvi.size)
        cross_ = rng.random(d)
        j_rand = int(rng.integers(0, d))
        cross_[j_rand] = 0.0
        v_values = cross_ <= CR[i]
        u_vec = vxi.copy()
        u_vec[:d] = np.where(v_values, vvi[:d], vxi[:d])
        u.append(Individual(vector_to_network(u_vec, meta_i), np.inf))
    return u


def _update_memory(M_F, M_CR, M_F_l, M_CR_l, F, CR, pbest_flag, delta_f, k, k_l):
    pb = pbest_flag & (delta_f > 0)
    lv = (~pbest_flag) & (delta_f > 0)

    if np.any(pb):
        w = delta_f[pb] / delta_f[pb].sum()
        s_cr = CR[pb]
        s_f = F[pb]
        if M_CR[k] == -1 or s_cr.max() == 0:
            M_CR[k] = -1
        else:
            M_CR[k] = np.sum(w * s_cr * s_cr) / np.sum(w * s_cr)
        M_F[k] = np.sum(w * s_f * s_f) / np.sum(w * s_f)

    if np.any(lv):
        w = delta_f[lv] / delta_f[lv].sum()
        s_cr = CR[lv]
        s_f = F[lv]
        if M_CR_l[k_l] == -1 or s_cr.max() == 0:
            M_CR_l[k_l] = -1
        else:
            M_CR_l[k_l] = np.sum(w * s_cr * s_cr) / np.sum(w * s_cr)
        M_F_l[k_l] = np.sum(w * s_f * s_f) / np.sum(w * s_f)
