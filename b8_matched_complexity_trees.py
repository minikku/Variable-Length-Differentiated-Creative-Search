#!/usr/bin/env python3
"""B8: LGBM and XGBoost under a total-leaves budget matched to the SLNN H_max.

Reviewer's matched-complexity counterfactual. The headline benchmark compares
the SLNN family at H in [1, H_max=20] against LGBM / XGBoost configurations
that produce 10^3-10^4 total leaves; the Pareto-frontier framing in Section 5.4
is honest only because the two methods operate at vastly different
architectural complexities. This script runs the symmetric question: when both
methods are constrained to the SAME maximum architectural budget (total leaves
<= H_max = 20), who wins on macro-F1?

Design (matched-complexity)
---------------------------
SLNN baseline: H in [1, H_max=20] hidden units. The hidden units are the
"decision units" of an SLNN: each hidden unit defines one nonlinear hyperplane
in the input space, and the final output is a softmax over linear combinations
of those H units. H_max=20 caps the SLNN at 20 decision units.

LGBM_matched: one tree, ``num_leaves`` in [max(2, min_leaves), H_max]. A single
CART tree with at most H_max=20 leaves produces at most H_max axis-aligned
decision regions in the input space. Total decision units = num_leaves <= H_max.
No boosting. LGBM's library minimum is 2 (a one-leaf no-split tree is a
constant predictor, which LGBM does not expose); ``--min-leaves 1`` is clamped
to 2 for LGBM with a one-line stderr notice.

XGBoost_matched: one tree, ``max_depth`` in [ceil(log2(min_leaves)),
log2(H_max)]. A single depth-d binary tree has up to 2**d leaves. With
H_max=20 the binary-tree ceiling is max_depth=4 -> 16 <= 20 leaves. Pass
``--min-leaves 1`` to include max_depth=0 in the search; that is the
constant-predictor degenerate baseline (predicts class prior, ignores input).

Note on matching units. ``H=1`` (one hidden unit) is one nonlinear sigmoid
hyperplane; ``num_leaves=1`` (zero splits) is a constant predictor with no
decision boundary. The natural matched-unit pairing is therefore
``H=1`` <-> ``num_leaves=2``: one nonlinear hyperplane vs one axis-aligned
binary split, each defining one decision boundary in the input space.
``--min-leaves 1`` is available for XGBoost to record the degenerate floor;
LGBM's API minimum is 2 and is clamped with a stderr notice.

Both tree variants are tuned with the same 40-trial Optuna budget as the
headline LGBM+Optuna / XGBoost+Optuna runs in Section 5.3, restricted to the
matched-budget search space; the inner 90/10 stratified split is preserved for
early stopping. Feature reduction (SelectKBest mutual_info_classif, k=256 on
d>1000) is preserved for parity with the headline pipeline.

Reported as a separate complexity-column for downstream analysis:
    total_leaves       : per the fitted tree (LGBM ``num_leaves``,
                         XGBoost 2**max_depth approximation)
    n_trees            : always 1 in this study
    decision_units     : alias for total_leaves (the "matched H" quantity)

Datasets: the nine healthcare benchmarks in ``config.SELECTED_DATASETS``.
Total: 9 datasets x 50 stratified CV partitions x 2 methods = 900 fold-runs.

Output
------
    results_B8_matched_complexity/results.csv
    results_B8_matched_complexity/summary.csv
    results_B8_matched_complexity/manifest.json

Usage
-----
    python b8_matched_complexity_trees.py                  # all 9 datasets, both methods, H_max=20
    python b8_matched_complexity_trees.py --h-max 10       # tighter match
    python b8_matched_complexity_trees.py --algos LGBM     # one method
    python b8_matched_complexity_trees.py --datasets LEUKEMIA1 LEUKEMIA2
    python b8_matched_complexity_trees.py --quick          # smoke test: 1 dataset x 5 folds
    python b8_matched_complexity_trees.py --n-jobs 8

Runtime estimate
----------------
    A single-tree LGBM with <= 20 leaves at 40-trial Optuna is roughly 3-5x
    faster than the unconstrained LGBM+Optuna headline run because each tree
    is much smaller; the b1 baselines (Section 5.1) reported LGBM+Optuna
    wall-clock of 200-700 s per fold (heavy boosting). At <= 20 leaves the
    expected wall-clock is 30-100 s per fold including the 40-trial Optuna
    sweep. XGBoost at max_depth <= 4, n_estimators=1 is similar.

    Sequential total: 900 folds x ~60 s = ~15 h on the reference 16-core
    hardware. With ``--n-jobs 8`` parallelism: ~2 h wall-clock. Allow ~4 h
    on slower hardware (8 cores, no SSD, background contention).

    Use ``--quick`` (5 folds on the first listed dataset, single algorithm)
    for a smoke test in 5-10 minutes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

import config
from vse.dataset import load_dataset, make_cv_splits

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    optuna = None

warnings.filterwarnings("ignore")


# Default H_max from the SLNN-family experimental protocol.
DEFAULT_H_MAX = 20

# Feature reduction parity with the headline LGBM+Optuna / XGBoost+Optuna pipeline.
FEATURE_REDUCTION_THRESHOLD = 1000
FEATURE_K = 256

# Optuna budget identical to the headline tree-ensemble runs (40 trials per fold).
OPTUNA_TRIALS = 40

ALL_ALGOS = ["LGBM", "XGBoost"]


def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def derive_seed(base_seed: int, algo: str, dataset: str, repeat: int, fold: int) -> int:
    """Same per-fold seed derivation as the rest of the project."""
    s = f"{base_seed}|{algo}|{dataset}|{repeat}|{fold}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


def _to_class_index(y: np.ndarray) -> np.ndarray:
    """ds.y is one-hot (n, K); convert to 1-D integer class indices for sklearn/LGBM/XGB."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    return y.astype(int).ravel()


def _select_features(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, seed: int
):
    d = x_train.shape[1]
    if d <= FEATURE_REDUCTION_THRESHOLD:
        return x_train, x_val
    k = min(FEATURE_K, d)
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=int(seed) % (2 ** 31 - 1)),
        k=k,
    )
    return selector.fit_transform(x_train, y_train), selector.transform(x_val)


def _inner_split(y_train: np.ndarray, seed: int):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    return next(sss.split(np.zeros(len(y_train)), y_train))


def _score(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    n_classes = int(np.max(y_true)) + 1
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.5
    return {"f1_macro": float(f1), "auc": float(auc), "acc": float(acc)}


# --------------------------------------------------------------------------- #
# LGBM at total-leaves <= H_max, n_estimators = 1                             #
# --------------------------------------------------------------------------- #
def _run_lgbm_matched(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    seed: int, h_max: int, min_leaves: int, optuna_trials: int,
) -> Dict[str, Any]:
    """Single-tree LGBM with leaves in [max(2, min_leaves), h_max] (40-trial Optuna).

    LGBM's API requires ``num_leaves >= 2`` (a tree with one leaf and no split
    is a constant predictor, which LGBM does not expose as a configuration).
    If ``min_leaves < 2`` is requested, the lower bound is clamped to 2 and a
    one-line stderr notice is emitted.
    """
    if lgb is None:
        raise RuntimeError("lightgbm not installed; pip install lightgbm")
    lgb_min = max(2, int(min_leaves))
    if lgb_min != int(min_leaves):
        print(f"  [LGBM] min_leaves={min_leaves} clamped to {lgb_min} "
              "(LGBM requires num_leaves >= 2)", file=sys.stderr)
    n_classes = int(np.max(np.concatenate([y_train, y_val]))) + 1
    inner_tr, inner_va = _inner_split(y_train, seed)
    x_tr_in, x_va_in = x_train[inner_tr], x_train[inner_va]
    y_tr_in, y_va_in = y_train[inner_tr], y_train[inner_va]

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            objective="binary" if n_classes == 2 else "multiclass",
            num_class=1 if n_classes == 2 else n_classes,
            metric="binary_logloss" if n_classes == 2 else "multi_logloss",
            class_weight="balanced",
            verbosity=-1,
            random_state=seed,
            # MATCHED-COMPLEXITY CONSTRAINT --------------------------------
            num_leaves=trial.suggest_int("num_leaves", lgb_min, int(h_max)),
            n_estimators=1,  # single tree -> total leaves <= h_max
            # ---------------------------------------------------------------
            max_depth=-1,    # let leaf-wise growth decide depth
            #learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
	    learning_rate=trial.suggest_float("learning_rate", 0.1, 0.3, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 30),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(x_tr_in, y_tr_in)
        proba = model.predict_proba(x_va_in)
        pred = np.argmax(proba, axis=1)
        return -f1_score(y_va_in, pred, average="macro", zero_division=0)

    if optuna is not None and optuna_trials > 0:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        best = study.best_params
    else:
        best = dict(num_leaves=min(20, h_max), learning_rate=0.05,
                    min_data_in_leaf=10, reg_alpha=0.0, reg_lambda=0.0)

    final_params = dict(
        objective="binary" if n_classes == 2 else "multiclass",
        num_class=1 if n_classes == 2 else n_classes,
        metric="binary_logloss" if n_classes == 2 else "multi_logloss",
        class_weight="balanced",
        verbosity=-1,
        random_state=seed,
        n_estimators=1,
        max_depth=-1,
        **best,
    )
    model = lgb.LGBMClassifier(**final_params)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_val)
    pred = np.argmax(proba, axis=1)
    metrics = _score(y_val, pred, proba)
    # Architectural-complexity reporting: total leaves in the single tree.
    n_trees = model.booster_.num_trees()
    # LGBM reports leaves via dump_model; for binary 1-tree this equals num_leaves.
    total_leaves = int(best.get("num_leaves", final_params["num_leaves"]))
    metrics["total_leaves"] = float(total_leaves)
    metrics["n_trees"] = float(n_trees)
    metrics["decision_units"] = float(total_leaves)
    return metrics


# --------------------------------------------------------------------------- #
# XGBoost at 2**max_depth <= H_max, n_estimators = 1                          #
# --------------------------------------------------------------------------- #
def _run_xgb_matched(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    seed: int, h_max: int, min_leaves: int, optuna_trials: int,
) -> Dict[str, Any]:
    """Single-tree XGBoost with 2**max_depth in [min_leaves, h_max] (40-trial Optuna).

    XGBoost accepts ``max_depth = 0`` (a single-leaf constant-predictor tree).
    If ``min_leaves == 1`` the search includes ``max_depth = 0`` as the
    degenerate floor; otherwise the lower bound on depth is the smallest d
    with 2**d >= min_leaves.
    """
    if xgb is None:
        raise RuntimeError("xgboost not installed; pip install xgboost")
    n_classes = int(np.max(np.concatenate([y_train, y_val]))) + 1
    inner_tr, inner_va = _inner_split(y_train, seed)
    x_tr_in, x_va_in = x_train[inner_tr], x_train[inner_va]
    y_tr_in, y_va_in = y_train[inner_tr], y_train[inner_va]

    # Lower bound on max_depth from min_leaves:
    #   min_leaves=1  -> depth 0 allowed (constant predictor)
    #   min_leaves=2  -> depth 1 (2 leaves)
    #   min_leaves>=3 -> ceil(log2(min_leaves))
    if int(min_leaves) <= 1:
        max_depth_floor = 0
    else:
        max_depth_floor = int(math.ceil(math.log2(max(2, int(min_leaves)))))
    # Upper bound: largest depth whose leaf ceiling (2**d) does not exceed h_max.
    # Example: h_max=20 -> max_depth_ceiling=4 (2**4=16 <= 20 < 2**5=32).
    max_depth_ceiling = max(max_depth_floor, int(math.floor(math.log2(h_max))))

    sample_w_in = compute_sample_weight("balanced", y_tr_in)
    sample_w_all = compute_sample_weight("balanced", y_train)

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            num_class=None if n_classes == 2 else n_classes,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            random_state=seed,
            # MATCHED-COMPLEXITY CONSTRAINT --------------------------------
            n_estimators=1,
            max_depth=trial.suggest_int("max_depth", max_depth_floor, max_depth_ceiling),
            # ---------------------------------------------------------------
            #learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.1, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            verbosity=0,
            n_jobs=1,
        )
        params = {k: v for k, v in params.items() if v is not None}
        model = xgb.XGBClassifier(**params)
        model.fit(x_tr_in, y_tr_in, sample_weight=sample_w_in)
        proba = model.predict_proba(x_va_in)
        pred = np.argmax(proba, axis=1)
        return -f1_score(y_va_in, pred, average="macro", zero_division=0)

    if optuna is not None and optuna_trials > 0:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        best = study.best_params
    else:
        best = dict(max_depth=max_depth_ceiling, learning_rate=0.05,
                    subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
                    reg_alpha=0.0, reg_lambda=0.0)

    final_params = dict(
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        num_class=None if n_classes == 2 else n_classes,
        eval_metric="logloss" if n_classes == 2 else "mlogloss",
        random_state=seed,
        n_estimators=1,
        verbosity=0,
        n_jobs=1,
        **best,
    )
    final_params = {k: v for k, v in final_params.items() if v is not None}
    model = xgb.XGBClassifier(**final_params)
    model.fit(x_train, y_train, sample_weight=sample_w_all)
    proba = model.predict_proba(x_val)
    pred = np.argmax(proba, axis=1)
    metrics = _score(y_val, pred, proba)
    # Architectural-complexity reporting.
    chosen_depth = int(best.get("max_depth", final_params["max_depth"]))
    leaf_ceiling = int(2 ** chosen_depth)
    metrics["total_leaves"] = float(leaf_ceiling)
    metrics["n_trees"] = 1.0
    metrics["decision_units"] = float(leaf_ceiling)
    return metrics


# --------------------------------------------------------------------------- #
# Per-fold driver                                                             #
# --------------------------------------------------------------------------- #
def run_single(task: dict) -> dict:
    ds = load_dataset(task["csv_path"])
    tr, va = task["train_idx"], task["val_idx"]
    x_train, y_train = ds.x[tr], _to_class_index(ds.y[tr])
    x_val, y_val = ds.x[va], _to_class_index(ds.y[va])

    # Feature reduction (parity with the headline tree-ensemble pipeline).
    x_train, x_val = _select_features(x_train, y_train, x_val, task["seed"])

    t0 = time.perf_counter()
    if task["algo"] == "LGBM_matched":
        metrics = _run_lgbm_matched(
            x_train, y_train, x_val, y_val,
            task["seed"], task["h_max"], task["min_leaves"], task["optuna_trials"],
        )
    elif task["algo"] == "XGBoost_matched":
        metrics = _run_xgb_matched(
            x_train, y_train, x_val, y_val,
            task["seed"], task["h_max"], task["min_leaves"], task["optuna_trials"],
        )
    else:
        raise ValueError(f"unknown algo {task['algo']}")
    runtime = time.perf_counter() - t0

    row = {
        "algo": task["algo"],
        "dataset": task["dataset"],
        "repeat": task["repeat"],
        "fold": task["fold"],
        "run_id": task["run_id"],
        "seed": task["seed"],
        "h_max_budget": int(task["h_max"]),
        "f1_macro": float(metrics["f1_macro"]),
        "auc": float(metrics["auc"]),
        "acc": float(metrics["acc"]),
        "total_leaves": float(metrics["total_leaves"]),
        "n_trees": float(metrics["n_trees"]),
        "decision_units": float(metrics["decision_units"]),
        "hidden_nodes": float(metrics["decision_units"]),  # alias for cross-study comparability
        "runtime_sec": float(runtime),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "input_size": int(ds.input_size),
        "output_size": int(ds.output_size),
    }
    return {"row": row}


def build_tasks(args) -> list:
    data_dir = args.data_dir or _here(config.CSV_DATA_DIR)
    tasks = []
    run_id = 0
    algos = args.algos or ALL_ALGOS
    algo_map = {"LGBM": "LGBM_matched", "XGBoost": "XGBoost_matched"}
    for dataset in args.datasets:
        csv_path = os.path.join(data_dir, dataset + ".csv")
        if not os.path.exists(csv_path):
            print(f"  !! missing CSV for '{dataset}'", file=sys.stderr)
            continue
        ds = load_dataset(csv_path)
        splits = make_cv_splits(
            ds, args.repeats, args.folds, args.base_seed,
            mode=args.cv_mode,
            holdout_train_fraction=config.HOLDOUT_TRAIN_FRACTION,
        )
        for repeat, fold, tr, va in splits:
            for algo_short in algos:
                algo = algo_map[algo_short]
                tasks.append({
                    "algo": algo,
                    "dataset": dataset,
                    "csv_path": csv_path,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "run_id": run_id,
                    "train_idx": tr,
                    "val_idx": va,
                    "seed": derive_seed(args.base_seed, algo, dataset, repeat, fold),
                    "h_max": int(args.h_max),
                    "min_leaves": int(args.min_leaves),
                    "optuna_trials": int(args.optuna_trials),
                })
                run_id += 1
                if args.quick and run_id >= 5:
                    return tasks
        if args.quick:
            break
    return tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=getattr(config, "SELECTED_DATASETS",
                                    ["ADENOCARCINOMA", "DLBCL", "DYSLEXIA", "DYSLEXIA_10p",
                                     "ILPD", "LEUKEMIA1", "LEUKEMIA2",
                                     "PROSTATE6033", "PROSTATE_TUMOR"]))
    ap.add_argument("--algos", nargs="*", choices=ALL_ALGOS, default=None,
                    help="Subset of {LGBM, XGBoost}. Default: both.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--h-max", type=int, default=DEFAULT_H_MAX,
                    help="Total-leaves budget per model. Default 20 (= SLNN H_max).")
    ap.add_argument("--min-leaves", type=int, default=2,
                    help=("Minimum leaves per model. Default 2 (one binary split = one "
                          "decision boundary, matched to SLNN H=1 which defines one "
                          "nonlinear hyperplane). Pass 1 to include a degenerate "
                          "constant-predictor floor (XGBoost max_depth=0); LGBM has a "
                          "library minimum of 2 and is clamped to max(2, min_leaves) "
                          "with a one-line stderr warning."))
    ap.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS,
                    help="Optuna trials per fold (default 40, matches the headline budget).")
    ap.add_argument("--n-jobs", type=int,
                    default=getattr(config, "N_JOBS", -1),
                    help="Parallel workers (-1 = all cores).")
    ap.add_argument("--out", default=_here("results_B8_matched_complexity"))
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 5 (dataset, fold, algo) combinations only.")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    tasks = build_tasks(args)
    if not tasks:
        print("No tasks built; check --datasets / --data-dir / --algos.", file=sys.stderr)
        return 2

    n_ds = len(set(t["dataset"] for t in tasks))
    n_algos = len(set(t["algo"] for t in tasks))
    print(f"Built {len(tasks)} tasks: {n_ds} datasets x {n_algos} algorithms "
          f"x {args.repeats * args.folds} folds.")
    print(f"Matched-complexity budget: total leaves <= {args.h_max} (= SLNN H_max)")
    print(f"Optuna budget per fold:    {args.optuna_trials} trials")
    print(f"Parallel workers:          {args.n_jobs}")

    t_start = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_single)(t) for t in tasks
    )
    wall = time.perf_counter() - t_start
    print(f"\nWall-clock: {wall:.1f} s ({wall / 60:.1f} min, {wall / 3600:.2f} h)")

    rows = [r["row"] for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)

    summary = (
        df.groupby(["algo", "dataset"])
        .agg(
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            auc_mean=("auc", "mean"),
            acc_mean=("acc", "mean"),
            leaves_mean=("total_leaves", "mean"),
            runtime_mean=("runtime_sec", "mean"),
            n=("f1_macro", "size"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\nPer-dataset summary:")
    print(summary.to_string(index=False))

    # Aggregate H_max-matched comparison numbers for the manuscript.
    print("\nAggregate (mean over datasets):")
    for algo, g in df.groupby("algo"):
        print(f"  {algo:18s}  F1={g['f1_macro'].mean():.4f}  "
              f"leaves={g['total_leaves'].mean():.2f}  "
              f"runtime={g['runtime_sec'].mean():.1f}s")

    manifest = {
        "script": os.path.basename(__file__),
        "h_max_budget": int(args.h_max),
        "algorithms": sorted(set(df["algo"].unique())),
        "datasets": sorted(set(df["dataset"].unique())),
        "n_folds_per_dataset": int(args.repeats * args.folds),
        "optuna_trials_per_fold": int(args.optuna_trials),
        "n_estimators": 1,
        "feature_reduction_threshold": FEATURE_REDUCTION_THRESHOLD,
        "feature_k": FEATURE_K,
        "base_seed": int(args.base_seed),
        "cv_mode": args.cv_mode,
        "wall_clock_sec": float(wall),
        "wall_clock_min": float(wall / 60),
        "n_jobs": int(args.n_jobs),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\nNext step:")
    print("  Compare results_B8_matched_complexity/summary.csv F1 against the "
          "DCS--VSE rows of Tables 3 / 5 in main_v32.tex on the same datasets. "
          "At matched architectural complexity (<= H_max leaves), the question "
          "is whether tree ensembles still beat the SLNN family on macro-F1 "
          "or whether the gap closes. The headline Pareto-frontier framing in "
          "Section 5.4 is either (a) softened to 'at matched budget the two "
          "families are comparable' if F1 converges, or (b) strengthened to "
          "'tree ensembles need 10^3-10^4 leaves to dominate, not 20' if F1 "
          "still favors the SLNN family at the matched budget. Both outcomes "
          "are publishable; the experiment closes the matched-complexity "
          "counterfactual that Reviewer 2 will otherwise raise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
