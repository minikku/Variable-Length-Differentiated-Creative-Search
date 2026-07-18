"""DCS / GLOProp teaching-learning family (variable-structure encoding).

A single parameterised implementation covers:

  * ``GLOProp_d``              -- original qCR, last student reinit-or-operators
  * ``DCS_noVSE_d``            -- last student always reinitialises
  * ``DCS_VSE_DKA_opt_0_d`` .. ``DCS_VSE_DKA_opt_13_d`` -- 14 qCR variants
  * ``DCS_VSE_hist_d``         -- opt-0 plus operator-success / ablation history

The 14 ``DKA`` files in the MATLAB code are byte-for-byte identical except for
which single ``qCR`` line is uncommented; they are reproduced here as the
``QCR_VARIANTS`` table.  ``DCS_VSE_DKA_opt_0_d`` and ``GLOProp_d`` use the same
formula and therefore share the implementation.
"""
from __future__ import annotations

import math
import numpy as np

from ..network import (
    initial_single_layer_network, network_to_vector, vector_to_network,
    addition_operator, elimination_operator, substitution_operator,
    random_hidden_size,
)
from ..rng import lnf2
from .base import Individual, AlgoResult, pop_sort


def _mround(x):
    """MATLAB ``round`` (half away from zero) for scalars/arrays."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# --------------------------------------------------------------------------- #
# qCR variant formulas.  curStd is 1-based; r = curStd/NP; rq = rank_to_qCR.   #
# --------------------------------------------------------------------------- #
def _qcr_opt0(rng, r, rq):
    return (_mround(rng.random() * rq) + (rng.random() <= rq)) / 2.0


def _qcr_opt1(rng, r, rq):
    return (_mround(rng.random() * r) + (rng.random() * (1 - r) <= rq)) / 2.0


def _qcr_opt2(rng, r, rq):
    return (_mround(rng.standard_normal() * r)
            + (rng.standard_normal() * (1 - r) <= rq)) / 2.0


def _qcr_opt3(rng, r, rq):
    return (_mround(rng.random() * r) + (rng.random() * (1 - r) <= rq)
            + ((1 - r) <= 0.5) * _mround(r)) / 2.0


def _qcr_opt4(rng, r, rq):
    return (_mround(rng.random() * r) + (rng.random() <= rq)
            + ((1 - r) <= 0.5) * _mround(r)) / 2.0


def _qcr_opt5(rng, r, rq):
    return (_mround(rng.random() * r) + (rng.random() <= rq)
            + ((1 - r) <= 0.5) * _mround(rng.random() * (1 - r))) / 2.0


def _qcr_opt6(rng, r, rq):
    return (_mround(rng.random() * r) + (rng.random() <= rq)
            + (((1 - r) <= rq) * _mround(rng.random() * (1 - r)))
            + (r >= rq)) / 3.0


def _qcr_opt7(rng, r, rq):
    return (_mround(rng.random() * r) + (rng.random() <= rq)
            + (((1 - r) <= rq) * _mround(rng.random() * r))
            + (r >= rq)) / 3.0


def _qcr_opt8(rng, r, rq):
    r0 = rng.random()
    return (_mround(r0 * r) + ((1 - r0) >= rq)
            + ((1 - r) <= 0.5) * _mround(r)) / 2.0


def _qcr_opt9(rng, r, rq):
    r1 = rng.random()
    r2 = rng.random()
    return (_mround(r2 * r) + ((1 - r1) >= rq)
            + ((1 - r) <= 0.5) * _mround(r)) / 2.0


def _qcr_opt10(rng, r, rq):
    r1 = rng.random()
    r2 = rng.random()
    return (_mround(r1 * r) + ((1 - r1) <= rq)
            + ((1 - r2) <= (1 - r)) * _mround(r)) / 2.0


def _qcr_opt11(rng, r, rq):
    return (_mround(rng.random() * r) + (rng.random() <= rq)
            + ((1 - r) <= 0.5) * _mround(r)) / 3.0


def _qcr_opt12(rng, r, rq):
    return (_mround(rng.random() * r) + (rng.random() <= rq)
            + (r <= 0.5) * _mround(r)
            + (r > 0.5) * _mround(rng.random())) / 2.0


def _qcr_opt13(rng, r, rq):
    r1x = _mround(rng.random())
    a = (_mround(rng.random() * r) + (rng.random() <= rq)
         + ((1 - r) <= 0.5) * _mround(r)) / 2.0
    b = (_mround(rng.random() * r) + (rng.random() <= rq)
         + (((1 - r) <= rq) * _mround(rng.random() * (1 - r)))
         + (r >= rq)) / 3.0
    return r1x * a + (1 - r1x) * b


QCR_VARIANTS = {
    0: _qcr_opt0, 1: _qcr_opt1, 2: _qcr_opt2, 3: _qcr_opt3, 4: _qcr_opt4,
    5: _qcr_opt5, 6: _qcr_opt6, 7: _qcr_opt7, 8: _qcr_opt8, 9: _qcr_opt9,
    10: _qcr_opt10, 11: _qcr_opt11, 12: _qcr_opt12, 13: _qcr_opt13,
}


def _randi_1based(rng, hi):
    """round(hi * rand + 0.5) clamped to [1, hi] (MATLAB index draw)."""
    v = int(math.floor(hi * rng.random() + 0.5))
    return min(max(v, 1), hi)


def _run_dcs(options, rng, qcr_fn, last_student_mode, track_history=False,
             uniform_dka=False):
    NP = int(options["popsize"])
    max_nfe = int(options["max_nfe"])
    golden_ratio = 2.0 / (1.0 + math.sqrt(5.0))
    ngS = max(6, math.ceil(NP * (golden_ratio / 3.0)))

    opts = dict(options)

    # --- initialise population ---
    pop = []
    for _ in range(NP):
        opts["hidn"] = random_hidden_size(opts, rng)
        chrom = initial_single_layer_network(opts, rng)
        fit, _, _ = options["fobj"](chrom)
        pop.append(Individual(chrom, fit))

    nfe = 0
    convergence = []
    pop = pop_sort(pop)
    teacher_fitness = pop[0].fitness
    teacher_ind = 1  # 1-based

    rank_to_qCR = 0.25 + 0.55 * ((np.arange(1, NP + 1) / NP) ** 0.5)
    # Uniform-DKA ablation: strip the rank-conditioning of the update rate by
    # replacing the per-rank rate and rank fraction with their population means,
    # so every individual receives the same expected update intensity. Total
    # update mass is matched to the default; only the rank dependence is removed.
    _mean_qcr = float(rank_to_qCR.mean())

    hist = {"hidden": [], "mcc": [], "add": [], "eli": [], "sub": []}

    while nfe <= max_nfe:
        social_impact = 0.1 + 0.518 * (1.0 - (nfe / max_nfe) ** 0.5)
        add_succ = np.zeros(NP)
        eli_succ = np.zeros(NP)
        sub_succ = np.zeros(NP)
        sel_op = np.zeros(NP, dtype=int)

        for cs in range(1, NP + 1):           # 1-based current student
            ci = cs - 1
            if uniform_dka:
                r = 0.5
                rq = _mean_qcr
            else:
                r = cs / NP
                rq = rank_to_qCR[ci]
            qcr_cs = qcr_fn(rng, r, rq)

            if cs == NP:
                reinit = (last_student_mode == "reinit_only") or (rng.random() < 0.5)
                if reinit:
                    opts["hidn"] = random_hidden_size(opts, rng)
                    next_chrom = initial_single_layer_network(opts, rng)
                else:
                    rr = rng.random()
                    sel = 1 if rr <= 0.4 else (2 if rr <= 0.8 else 3)
                    if sel == 1:
                        next_chrom = addition_operator(pop[ci].chrom, opts, rng)
                        sel_op[ci] = 1
                        add_succ[ci] = 1
                    elif sel == 2:
                        next_chrom = elimination_operator(pop[ci].chrom, opts, rng)
                        sel_op[ci] = 2
                        eli_succ[ci] = 1
                    else:
                        next_chrom = substitution_operator(pop[ci].chrom, opts, rng)
                        sel_op[ci] = 3
                        sub_succ[ci] = 1

            elif cs <= ngS:                    # gifted / advanced class
                hp = cs
                for _ in range(10000):
                    hp = _randi_1based(rng, NP)
                    if hp != cs and hp != teacher_ind:
                        break
                v1, meta1 = network_to_vector(pop[ci].chrom)
                v2, _ = network_to_vector(pop[hp - 1].chrom)
                D = min(v1.size, v2.size)
                jrand = int(math.floor(D * rng.random() + 1))
                tmp = v1.copy()
                mask = rng.random(D) <= qcr_cs
                mask[jrand - 1] = True
                idx = np.nonzero(mask)[0]
                if idx.size:
                    steps = lnf2(rng, golden_ratio, 0.05, 1, idx.size).ravel()
                    tmp[idx] = v2[idx] + steps
                next_chrom = vector_to_network(tmp, meta1)

            else:                              # average class
                gf = cs
                for _ in range(10000):
                    gf = _randi_1based(rng, NP)
                    if gf != cs and gf != teacher_ind:
                        break
                fr = cs
                for _ in range(10000):
                    fr = ngS + _randi_1based(rng, NP - ngS)
                    if fr != cs and fr != teacher_ind and fr != gf:
                        break
                learning_ability = rng.random()
                v1, meta1 = network_to_vector(pop[ci].chrom)
                v2, _ = network_to_vector(pop[teacher_ind - 1].chrom)
                v3, _ = network_to_vector(pop[gf - 1].chrom)
                v4, _ = network_to_vector(pop[fr - 1].chrom)
                D = min(v1.size, v2.size, v3.size, v4.size)
                jrand = int(math.floor(D * rng.random() + 1))
                tmp = v1.copy()
                mask = rng.random(D) <= qcr_cs
                mask[jrand - 1] = True
                idx = np.nonzero(mask)[0]
                if idx.size:
                    tmp[idx] = (v2[idx]
                                + (v3[idx] - v1[idx]) * learning_ability
                                + (v4[idx] - v1[idx]) * social_impact)
                next_chrom = vector_to_network(tmp, meta1)

            new_fit, _, _ = options["fobj"](next_chrom)
            nfe += 1
            if new_fit <= pop[ci].fitness:
                pop[ci] = Individual(next_chrom, new_fit)
                if new_fit < teacher_fitness:
                    teacher_fitness = new_fit
                    teacher_ind = cs
            elif track_history:
                if sel_op[ci] == 1:
                    add_succ[ci] = 0
                elif sel_op[ci] == 2:
                    eli_succ[ci] = 0
                elif sel_op[ci] == 3:
                    sub_succ[ci] = 0

        pop = pop_sort(pop)
        teacher_ind = 1
        convergence.append(teacher_fitness)

        if track_history:
            mcc_accu = 0.0
            hid_accu = 0.0
            for ind in pop:
                _, _, _, mcc_val, hid_val = options["fobj_ablation"](ind.chrom)
                mcc_accu += mcc_val
                hid_accu += hid_val
            hist["hidden"].append(hid_accu / NP)
            hist["mcc"].append(mcc_accu / NP)
            hist["add"].append(float(add_succ.sum()))
            hist["eli"].append(float(eli_succ.sum()))
            hist["sub"].append(float(sub_succ.sum()))

    best_x = pop[0].chrom
    extra = {}
    if track_history:
        order = np.argsort(hist["mcc"])[::-1]
        extra = {
            "sorted_hist_mcc": np.array(hist["mcc"])[order],
            "sorted_hist_hidden_size": np.array(hist["hidden"])[order],
            "sorted_hist_add_operator_succ": np.array(hist["add"])[order],
            "sorted_hist_eli_operator_succ": np.array(hist["eli"])[order],
            "sorted_hist_sub_operator_succ": np.array(hist["sub"])[order],
        }
    return AlgoResult(teacher_fitness, best_x, convergence, extra)


def build_registry():
    """Return the name -> callable map for the whole DCS family."""
    reg = {}

    def make(qcr_idx, last_mode, track=False, uniform=False):
        fn = QCR_VARIANTS[qcr_idx]
        return lambda options, rng: _run_dcs(options, rng, fn, last_mode, track,
                                             uniform_dka=uniform)

    # GLOProp == DKA opt 0 (same qCR), reinit-or-operators at last student.
    reg["GLOProp_d"] = make(0, "reinit_or_operators")
    reg["DCS_noVSE_d"] = make(0, "reinit_only")
    # Uniform-DKA ablation: identical to DCS_noVSE_d but with the rank-conditioned
    # update rate flattened to its population mean (rank dependence removed).
    reg["DCS_VSE_uniformDKA_d"] = make(0, "reinit_only", uniform=True)
    reg["DCS_VSE_hist_d"] = make(0, "reinit_or_operators", track=True)
    for k in range(14):                      # DCS_VSE_DKA_opt_0_d .. opt_13_d
        reg[f"DCS_VSE_DKA_opt_{k}_d"] = make(k, "reinit_or_operators")
    return reg
