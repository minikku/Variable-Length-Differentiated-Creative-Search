"""b11_dnm.py -- Variable Structure Encoding for the Dendritic Neuron Model (DNM).

Second deployment case for the DCS-VSE variable-length optimizer. The predictor
family is a multi-output Dendritic Neuron Model whose STRUCTURAL VARIABLE is the
number of dendrite branches M, exactly analogous to the hidden-unit count H in
the single-layer network (SLNN) case. The same rank-guided arithmetic update,
cross-length alignment (shared-prefix restriction), Linnik-flight perturbation,
size-regularized MCC objective, and structural add/eliminate/substitute
operators are re-instantiated on the DNM encoding. Everything else in the
pipeline (50 stratified partitions, mutual-information front-end for d>1000,
per-fold SHA-256 seeds, macro-F1 / OVR-AUC / accuracy scoring) is shared with
b8/b9/b10 so the DNM results are paired with the SLNN family.

Model (M branches, d inputs, C classes):
  synapse    S_{m,i}(x) = sigmoid(k*(w_{m,i} x_i - q_{m,i}))
  dendrite   D_m(x)     = exp( mean_i log S_{m,i}(x) )      # geometric-mean AND
  membrane   V_c(x)     = sum_m v_{c,m} D_m(x) + b_c
  soma       O_c(x)     = sigmoid(V_c(x))
The geometric-mean dendrite is the numerically stable form of the classical
multiplicative dendrite (Todo/Gao DNM); it preserves the AND-like interaction
without underflow when the fan-in d is large.

Encoding (variable length in M), block-consistent for cross-length alignment:
  per branch m:  [ w_{m,1..d}, q_{m,1..d}, v_{1..C, m} ]      -> (2d + C)
  trailing:      [ b_{1..C} ]                                 -> C
  D(M) = M*(2d + C) + C          (parallels SLNN P(H)=H(d+C+1)+C)

Run:
  python b11_dnm.py --algos DNM_VSE                  # variable-M DCS-VSE
  python b11_dnm.py --algos DNM_VSE DNM_fixed1       # + size-locked M=1 control
Output: results_B11_dnm/summary.csv (+ results.csv, manifest.json), same schema.
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config
from b8_matched_complexity_trees import (
    derive_seed, _to_class_index, _select_features, _score,
    FEATURE_REDUCTION_THRESHOLD, FEATURE_K,
)
from vse.rng import lnf2

# --------------------------------------------------------------------------- #
# Search / model constants (matched to the SLNN DCS-VSE configuration)         #
# --------------------------------------------------------------------------- #
POP = 30
NFE = 30_000
M_MIN = 1
M_MAX = 20
LB, UB = -10.0, 10.0
SPAN = UB - LB
LAMBDA_SIZE = 0.05
LINNIK_ALPHA = 1.5
LINNIK_SCALE = 0.3 * SPAN
K_SYN = 5.0                       # synaptic slope
STRUCT_P = (0.5, 0.4, 0.4, 0.2)   # reinit; add; eliminate; substitute (worst ind.)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _log_sigmoid(z):
    # numerically stable log(sigmoid(z)) = -softplus(-z)
    z = np.clip(z, -500.0, 500.0)
    return -np.logaddexp(0.0, -z)


# --------------------------------------------------------------------------- #
# DNM predictor                                                                #
# --------------------------------------------------------------------------- #
class DNM:
    __slots__ = ("W", "Q", "V", "b")

    def __init__(self, W, Q, V, b):
        self.W = W        # (M, d) synaptic weights
        self.Q = Q        # (M, d) synaptic thresholds
        self.V = V        # (C, M) membrane->class map
        self.b = b        # (C,)   class bias

    @property
    def M(self):
        return self.W.shape[0]

    def copy(self):
        return DNM(self.W.copy(), self.Q.copy(), self.V.copy(), self.b.copy())

    def n_params(self):
        M, d = self.W.shape
        C = self.V.shape[0]
        return M * (2 * d + C) + C


def dnm_forward(net: DNM, X: np.ndarray) -> np.ndarray:
    """X: (n, d) -> O: (n, C)."""
    # z: (n, M, d)
    z = K_SYN * (X[:, None, :] * net.W[None, :, :] - net.Q[None, :, :])
    logS = _log_sigmoid(z)                 # (n, M, d)
    D = np.exp(logS.mean(axis=2))          # (n, M) geometric-mean dendrite
    Vc = D @ net.V.T + net.b[None, :]      # (n, C)
    return _sigmoid(Vc)                    # (n, C)


def _mcc_loss(y_true, y_pred):
    n, m = y_pred.shape
    tp = np.sum(y_pred * y_true, axis=0)
    tn = np.sum((1 - y_pred) * (1 - y_true), axis=0)
    fp = np.sum(y_pred * (1 - y_true), axis=0)
    fn = np.sum((1 - y_pred) * y_true, axis=0)
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp + 1) * (tp + fn + 1) * (tn + fp + 1) * (tn + fn + 1))
    return float(1.0 - np.mean(num / (den + 1.0)))


# --------------------------------------------------------------------------- #
# Variable-length encode / decode (block-consistent for cross-length align)    #
# --------------------------------------------------------------------------- #
def encode(net: DNM) -> np.ndarray:
    M, d = net.W.shape
    C = net.V.shape[0]
    per = np.concatenate(
        [np.concatenate([net.W[m], net.Q[m], net.V[:, m]]) for m in range(M)]
    )
    return np.concatenate([per, net.b])


def decode(vec: np.ndarray, M: int, d: int, C: int) -> DNM:
    stride = 2 * d + C
    W = np.empty((M, d)); Q = np.empty((M, d)); V = np.empty((C, M))
    for m in range(M):
        blk = vec[m * stride:(m + 1) * stride]
        W[m] = blk[:d]
        Q[m] = blk[d:2 * d]
        V[:, m] = blk[2 * d:2 * d + C]
    b = vec[M * stride:M * stride + C]
    return DNM(W, Q, V, b.copy())


def rand_dnm(M: int, d: int, C: int, rng) -> DNM:
    return DNM(
        LB + SPAN * rng.random((M, d)),
        LB + SPAN * rng.random((M, d)),
        LB + SPAN * rng.random((C, M)),
        LB + SPAN * rng.random(C),
    )


# --------------------------------------------------------------------------- #
# Structural operators on dendrite branches                                    #
# --------------------------------------------------------------------------- #
def add_branch(net: DNM, rng) -> DNM:
    if net.M >= M_MAX:
        return net.copy()
    d = net.W.shape[1]; C = net.V.shape[0]
    W = np.vstack([net.W, LB + SPAN * rng.random((1, d))])
    Q = np.vstack([net.Q, LB + SPAN * rng.random((1, d))])
    V = np.hstack([net.V, LB + SPAN * rng.random((C, 1))])
    return DNM(W, Q, V, net.b.copy())


def del_branch(net: DNM, rng) -> DNM:
    if net.M <= M_MIN:
        return net.copy()
    j = rng.integers(0, net.M)
    keep = [i for i in range(net.M) if i != j]
    return DNM(net.W[keep].copy(), net.Q[keep].copy(),
               net.V[:, keep].copy(), net.b.copy())


def subst_branch(net: DNM, rng) -> DNM:
    out = net.copy()
    d = out.W.shape[1]; C = out.V.shape[0]
    j = rng.integers(0, out.M)
    out.W[j] = LB + SPAN * rng.random(d)
    out.Q[j] = LB + SPAN * rng.random(d)
    out.V[:, j] = LB + SPAN * rng.random(C)
    return out


# --------------------------------------------------------------------------- #
# DKA rate (opt-0 profile from the DCS-VSE default) and fitness                 #
# --------------------------------------------------------------------------- #
def _mround(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def qcr_opt0(rng, rq):
    return (_mround(rng.random() * rq) + (rng.random() <= rq)) / 2.0


def _fitness(net: DNM, Xtr, Ytr):
    loss = _mcc_loss(Ytr, dnm_forward(net, Xtr))
    penalty = 0.0
    if M_MAX > M_MIN:
        penalty = ((net.M - M_MIN) / (M_MAX - M_MIN)) * LAMBDA_SIZE
    return loss * 100.0 + penalty * 100.0


# --------------------------------------------------------------------------- #
# Variable-length DCS-VSE search on the DNM family                             #
# --------------------------------------------------------------------------- #
def dcs_vse_dnm(Xtr, Ytr, d, C, rng, fixed_M=None):
    lo = hi = fixed_M if fixed_M is not None else None
    pop = []
    for _ in range(POP):
        M = fixed_M if fixed_M is not None else (M_MIN + int((M_MAX - M_MIN) * rng.random()))
        M = max(M_MIN, M)
        net = rand_dnm(M, d, C, rng)
        pop.append([net, _fitness(net, Xtr, Ytr)])
    evals = POP
    while evals < NFE:
        pop.sort(key=lambda z: z[1])
        leader = pop[0][0]
        NP = len(pop)
        for i in range(NP):
            r = (i + 1) / NP
            rq = 1.0 - r                         # rank-to-qCR (better rank -> larger)
            xi = pop[i][0]
            vi = encode(xi)
            # peer: a higher-ranked individual (rank-guided)
            j = rng.integers(0, max(1, i)) if i > 0 else 0
            xp = pop[j][0]
            vp = encode(xp)
            vl = encode(leader)
            # cross-length alignment: restrict arithmetic to shared prefix
            L = min(vi.size, vp.size, vl.size)
            child = vi.copy()
            gate = rng.random(L) <= qcr_opt0(rng, rq)
            step = (vl[:L] - vi[:L]) + rng.random() * (vp[:L] - vi[:L])
            perturb = lnf2(rng, LINNIK_ALPHA, LINNIK_SCALE, 1, L).ravel()
            child[:L] = np.where(gate, vi[:L] + rng.random() * step + perturb, vi[:L])
            child = np.clip(child, LB, UB)
            cand = decode(child, xi.M, d, C)
            fc = _fitness(cand, Xtr, Ytr)
            evals += 1
            if fc < pop[i][1]:
                pop[i] = [cand, fc]
            if evals >= NFE:
                break
        # structural exploration on the current worst individual
        if fixed_M is None and evals < NFE:
            pop.sort(key=lambda z: z[1])
            worst = pop[-1][0]
            u = rng.random()
            if u < STRUCT_P[0]:
                M = M_MIN + int((M_MAX - M_MIN) * rng.random())
                cand = rand_dnm(max(M_MIN, M), d, C, rng)
            elif u < STRUCT_P[0] + 0.25:
                cand = add_branch(worst, rng)
            elif u < STRUCT_P[0] + 0.40:
                cand = del_branch(worst, rng)
            else:
                cand = subst_branch(worst, rng)
            fc = _fitness(cand, Xtr, Ytr)
            evals += 1
            if fc <= pop[-1][1]:
                pop[-1] = [cand, fc]
    pop.sort(key=lambda z: z[1])
    return pop[0][0]


# --------------------------------------------------------------------------- #
# Per-fold task                                                                #
# --------------------------------------------------------------------------- #
def _onehot(y_idx, C):
    Y = np.zeros((y_idx.size, C))
    Y[np.arange(y_idx.size), y_idx] = 1.0
    return Y


def run_single(task):
    algo = task["algo"]; ds = task["dataset"]
    repeat = task["repeat"]; fold = task["fold"]
    X = task["X"]; y = task["y"]; tr = task["train_idx"]; te = task["test_idx"]
    seed = derive_seed(config.BASE_SEED, algo, ds, repeat, fold)
    rng = np.random.default_rng(seed)

    Xtr_raw, Xte_raw = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    if task.get("match_slnn"):
        # Strict match to the headline SLNN pipeline: full features (no MI) and
        # the per-feature min-max [0,1] scaling already applied at load time.
        Xtr, Xte = Xtr_raw, Xte_raw
    else:
        Xtr_raw, Xte_raw = _select_features(Xtr_raw, ytr, Xte_raw, seed)
        mu = Xtr_raw.mean(axis=0); sd = Xtr_raw.std(axis=0) + 1e-8
        Xtr = (Xtr_raw - mu) / sd
        Xte = (Xte_raw - mu) / sd

    classes = np.unique(y)                      # sorted; index order for proba columns
    C = int(classes.size)
    ytr_idx = np.searchsorted(classes, ytr)
    yte_idx = np.searchsorted(classes, yte)
    Ytr = _onehot(ytr_idx, C)
    d = Xtr.shape[1]

    fixed_M = 1 if algo.endswith("fixed1") else None
    t0 = time.time()
    net = dcs_vse_dnm(Xtr, Ytr, d, C, rng, fixed_M=fixed_M)
    runtime = time.time() - t0

    proba = dnm_forward(net, Xte)               # (n, C)
    y_pred_idx = np.argmax(proba, axis=1)
    sc = _score(yte_idx, y_pred_idx, proba)
    return {
        "algo": algo, "dataset": ds, "repeat": repeat, "fold": fold,
        "f1_macro": sc["f1_macro"], "auc": sc["auc"], "accuracy": sc["acc"],
        "dendrites": float(net.M), "params": float(net.n_params()),
        "runtime": runtime,
    }


# --------------------------------------------------------------------------- #
# Driver                                                                        #
# --------------------------------------------------------------------------- #
def _minmax01(X):
    """Per-feature min-max to [0,1] over the whole matrix (matches the SLNN
    pipeline's mapminmax; constant columns map to 0)."""
    xmin = X.min(axis=0); xmax = X.max(axis=0); span = xmax - xmin
    span_safe = np.where(span == 0, 1.0, span)
    out = (X - xmin) / span_safe
    out[:, span == 0] = 0.0
    return out


def _load_dataset(name, match_slnn=False):
    path = os.path.join(HERE, config.CSV_DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    y = df.iloc[:, -1].to_numpy()
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    if match_slnn:
        X = _minmax01(X)
    return X, y


def build_tasks(args):
    from sklearn.model_selection import RepeatedStratifiedKFold
    tasks = []
    match = bool(getattr(args, "match_slnn", False))
    datasets = args.datasets or config.SELECTED_DATASETS
    for ds in datasets:
        X, y = _load_dataset(ds, match_slnn=match)
        yc = _to_class_index(y)
        rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                       random_state=config.BASE_SEED)
        for k, (tr, te) in enumerate(rskf.split(X, yc)):
            repeat = k // args.folds
            fold = k % args.folds
            for algo in args.algos:
                tasks.append({"algo": algo, "dataset": ds, "repeat": repeat,
                              "fold": fold, "X": X, "y": y, "match_slnn": match,
                              "train_idx": tr, "test_idx": te})
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["DNM_VSE"],
                    help="DNM_VSE (variable M) and/or DNM_fixed1 (size-locked M=1)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--match-slnn", action="store_true",
                    help="Strict match to the SLNN input: full features (no MI) and "
                         "min-max [0,1] scaling only (no z-score).")
    ap.add_argument("--quick", action="store_true", help="1 repeat x 2 folds, 1 dataset")
    args = ap.parse_args(argv)

    if args.outdir is None:
        args.outdir = os.path.join(HERE, "results_B11_dnm_matched"
                                   if args.match_slnn else "results_B11_dnm")

    if args.quick:
        args.repeats = 1; args.folds = 2
        args.datasets = args.datasets or ["WDBC"]

    os.makedirs(args.outdir, exist_ok=True)
    tasks = build_tasks(args)
    print(f"[b11_dnm] {len(tasks)} fold-tasks | algos={args.algos} "
          f"| datasets={len(args.datasets or config.SELECTED_DATASETS)} | jobs={args.jobs}")
    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_single)(t) for t in tasks)
    print(f"[b11_dnm] done in {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "results.csv"), index=False)
    agg = (df.groupby(["algo", "dataset"])
             .agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
                  auc_mean=("auc", "mean"), acc_mean=("accuracy", "mean"),
                  dendrites_mean=("dendrites", "mean"), params_mean=("params", "mean"),
                  runtime_mean=("runtime", "mean"), n=("f1_macro", "size"))
             .reset_index())
    agg.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump({"algorithms": args.algos, "n_folds_per_dataset": args.repeats * args.folds,
                   "pop": POP, "nfe": NFE, "M_range": [M_MIN, M_MAX], "k_syn": K_SYN,
                   "lambda_size": LAMBDA_SIZE}, f, indent=2)
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
