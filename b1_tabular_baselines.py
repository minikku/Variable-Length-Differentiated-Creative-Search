#!/usr/bin/env python3
"""B1: Off-the-shelf tabular baselines under the same repeated 5x10 stratified CV.

Runs three reference models under *exactly* the protocol used in main.py:
  - the same load_dataset / make_cv_splits from vse.dataset (cached, deterministic),
  - the same 50 partitions per dataset (5 repeats x 10 folds),
  - the same per-run seed derivation (sha256 of "BASE_SEED|algo|dataset|repeat|fold"),
  - the same results.csv / summary.csv / manifest.json layout (drop into a sibling
    of results_H1to20, e.g. results_B1_baselines).

What changes vs. v12 / earlier off-the-shelf reports
----------------------------------------------------
Each baseline is configured *deliberately*, not at sklearn defaults:

  LightGBM
    - class_weight = "balanced"
    - early-stopping on a 90/10 inner stratified split with patience 100
    - per-fold mean-imputation of missing values is unnecessary (datasets are dense)
    - feature selection for d > 1000: SelectKBest(mutual_info_classif, k = min(d, 256))
      using ONLY training-fold data (no leakage)
    - Optuna over (num_leaves, max_depth, learning_rate, min_data_in_leaf,
      reg_alpha, reg_lambda) with 40 trials, TPE sampler, median pruner

  XGBoost
    - scale_pos_weight = (#majority / #minority) for binary tasks
    - multi:softprob with class weights set via sample_weight for multiclass
    - early-stopping on inner 90/10 stratified split with patience 100
    - same SelectKBest feature reduction on high-d datasets
    - Optuna over (max_depth, learning_rate, subsample, colsample_bytree,
      min_child_weight, reg_alpha, reg_lambda) with 40 trials

  sklearn MLP (MLPClassifier)
    - Pipeline: StandardScaler -> MLPClassifier
    - hidden_layer_sizes selected by inner-validation F1 over {(2,), (3,), (5,), (10,), (20,)}
    - solver='adam', activation='relu', lr='adaptive', max_iter=1000
    - imbalance handled by class-balanced random oversampling of the training
      fold (MLPClassifier.fit does not accept ``sample_weight``; sklearn #14101)

Usage
-----
    python b1_tabular_baselines.py                  # all 9 datasets, all 3 methods
    python b1_tabular_baselines.py --algos LGBM     # one method
    python b1_tabular_baselines.py --datasets ILPD  # one dataset
    python b1_tabular_baselines.py --quick          # smoke test

Requires
--------
    pip install lightgbm xgboost optuna scikit-learn

Output
------
    results_B1_baselines/results.csv      (one row per algo,dataset,repeat,fold)
    results_B1_baselines/summary.csv      (mean/std per algo,dataset)
    results_B1_baselines/manifest.json    (config + wall-clock)
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import config
from vse.dataset import load_dataset, make_cv_splits

# Optional imports — gracefully skip the corresponding baseline if unavailable.
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

# Methods registered in this script (B1 set).
B1_ALGOS = ["LGBM", "XGBoost", "MLP"]

# Feature-reduction threshold: datasets with more than this many features get
# SelectKBest reduction to FEATURE_K before training.  Mirrors the "feature
# selection on high-d datasets" item in the limitations section.
FEATURE_REDUCTION_THRESHOLD = 1000
FEATURE_K = 256

# Optuna budget per fold (matches the v12 LGBM+Optuna budget for fairness).
OPTUNA_TRIALS = 40


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def derive_seed(base_seed: int, algo: str, dataset: str, repeat: int, fold: int) -> int:
    """Same derivation as in main.py so seeds are comparable across studies."""
    s = f"{base_seed}|{algo}|{dataset}|{repeat}|{fold}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


# --------------------------------------------------------------------------- #
# Preprocessing common to all three baselines                                 #
# --------------------------------------------------------------------------- #
def _select_features(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, seed: int
):
    """Fit SelectKBest on the training fold only, apply to train and val."""
    d = x_train.shape[1]
    if d <= FEATURE_REDUCTION_THRESHOLD:
        return x_train, x_val, None
    k = min(FEATURE_K, d)
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=seed),
        k=k,
    )
    x_train_sel = selector.fit_transform(x_train, y_train)
    x_val_sel = selector.transform(x_val)
    return x_train_sel, x_val_sel, selector


def _inner_split(y_train: np.ndarray, seed: int):
    """Inner 90/10 stratified split for early stopping / inner CV."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    return next(sss.split(np.zeros(len(y_train)), y_train))


def _balanced_oversample(x: np.ndarray, y: np.ndarray, seed: int):
    """Class-balanced random oversampling on (x, y).

    sklearn's MLPClassifier does not accept ``sample_weight`` (long-standing
    sklearn limitation), so we instead replicate minority-class samples until
    every class has the same count as the largest class.  This gives an
    effective balanced training distribution for SGD-based learners and is
    the standard substitute when ``sample_weight`` is unavailable.

    Deterministic given ``seed``; uses sampling with replacement on the
    minority classes only (the majority class is left untouched).
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max())
    parts_x, parts_y = [x], [y]
    for c, n in zip(classes, counts):
        if n >= target:
            continue
        cls_idx = np.where(y == c)[0]
        extra = rng.choice(cls_idx, size=target - n, replace=True)
        parts_x.append(x[extra])
        parts_y.append(y[extra])
    x_bal = np.concatenate(parts_x, axis=0)
    y_bal = np.concatenate(parts_y, axis=0)
    # Final permutation so SGD does not see all minority duplicates back-to-back.
    perm = rng.permutation(len(y_bal))
    return x_bal[perm], y_bal[perm]


def _score(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Same metrics as the rest of the project (macro-F1, macro-OvR AUC, ACC)."""
    n_classes = int(np.max(y_true)) + 1
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        # Happens when a class is absent from the validation fold; fall back to
        # the same convention as the rest of the project (worst-case 0.5).
        auc = 0.5
    return {"f1_macro": float(f1), "auc": float(auc), "acc": float(acc)}


# --------------------------------------------------------------------------- #
# LightGBM                                                                    #
# --------------------------------------------------------------------------- #
def _run_lgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    optuna_trials: int,
) -> Dict[str, Any]:
    if lgb is None:
        raise RuntimeError("lightgbm not installed; pip install lightgbm")
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
            num_leaves=trial.suggest_int("num_leaves", 7, 127),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 2, 30),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            n_estimators=5000,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_tr_in,
            y_tr_in,
            eval_set=[(x_va_in, y_va_in)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        proba = model.predict_proba(x_va_in)
        pred = np.argmax(proba, axis=1)
        return -f1_score(y_va_in, pred, average="macro", zero_division=0)

    if optuna is not None and optuna_trials > 0:
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        best = study.best_params
    else:
        best = dict(
            num_leaves=31, max_depth=-1, learning_rate=0.05,
            min_data_in_leaf=10, reg_alpha=0.0, reg_lambda=0.0,
        )

    final_params = dict(
        objective="binary" if n_classes == 2 else "multiclass",
        num_class=1 if n_classes == 2 else n_classes,
        metric="binary_logloss" if n_classes == 2 else "multi_logloss",
        class_weight="balanced",
        verbosity=-1,
        random_state=seed,
        n_estimators=5000,
        **best,
    )
    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train[inner_va], y_train[inner_va])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    proba = model.predict_proba(x_val)
    pred = np.argmax(proba, axis=1)
    metrics = _score(y_val, pred, proba)
    metrics["hidden_nodes"] = float(model.booster_.num_trees())  # tree count as complexity proxy
    return metrics


# --------------------------------------------------------------------------- #
# XGBoost                                                                     #
# --------------------------------------------------------------------------- #
def _run_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    optuna_trials: int,
) -> Dict[str, Any]:
    if xgb is None:
        raise RuntimeError("xgboost not installed; pip install xgboost")
    n_classes = int(np.max(np.concatenate([y_train, y_val]))) + 1
    inner_tr, inner_va = _inner_split(y_train, seed)
    x_tr_in, x_va_in = x_train[inner_tr], x_train[inner_va]
    y_tr_in, y_va_in = y_train[inner_tr], y_train[inner_va]

    sample_w_in = compute_sample_weight("balanced", y_tr_in)
    sample_w_all = compute_sample_weight("balanced", y_train)

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            num_class=None if n_classes == 2 else n_classes,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            tree_method="hist",
            random_state=seed,
            verbosity=0,
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 10.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            n_estimators=5000,
            early_stopping_rounds=100,
        )
        if n_classes == 2:
            params.pop("num_class")
        model = xgb.XGBClassifier(**params)
        model.fit(
            x_tr_in, y_tr_in,
            sample_weight=sample_w_in,
            eval_set=[(x_va_in, y_va_in)],
            verbose=False,
        )
        proba = model.predict_proba(x_va_in)
        pred = np.argmax(proba, axis=1)
        return -f1_score(y_va_in, pred, average="macro", zero_division=0)

    if optuna is not None and optuna_trials > 0:
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        best = study.best_params
    else:
        best = dict(
            max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
            min_child_weight=1.0, reg_alpha=0.0, reg_lambda=1.0,
        )

    final_params = dict(
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        eval_metric="logloss" if n_classes == 2 else "mlogloss",
        tree_method="hist",
        random_state=seed,
        verbosity=0,
        n_estimators=5000,
        early_stopping_rounds=100,
        **best,
    )
    if n_classes > 2:
        final_params["num_class"] = n_classes
    model = xgb.XGBClassifier(**final_params)
    model.fit(
        x_train, y_train,
        sample_weight=sample_w_all,
        eval_set=[(x_train[inner_va], y_train[inner_va])],
        verbose=False,
    )
    proba = model.predict_proba(x_val)
    pred = np.argmax(proba, axis=1)
    metrics = _score(y_val, pred, proba)
    metrics["hidden_nodes"] = float(model.get_booster().num_boosted_rounds())
    return metrics


# --------------------------------------------------------------------------- #
# sklearn MLP                                                                  #
# --------------------------------------------------------------------------- #
def _run_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    """sklearn MLPClassifier with StandardScaler and class-balanced oversampling.

    Note: ``MLPClassifier.fit`` does not accept ``sample_weight`` (this is a
    long-standing sklearn limitation, issue #14101).  We achieve the same
    "balanced" effective training distribution by oversampling minority
    classes with replacement to match the majority class size.
    """
    candidate_sizes = [(2,), (3,), (5,), (10,), (20,)]
    inner_tr, inner_va = _inner_split(y_train, seed)
    x_tr_in, x_va_in = x_train[inner_tr], x_train[inner_va]
    y_tr_in, y_va_in = y_train[inner_tr], y_train[inner_va]

    # Balanced oversampling for both the inner-CV training fold and the final
    # training fold.  Seeds are derived deterministically.
    x_tr_in_bal, y_tr_in_bal = _balanced_oversample(x_tr_in, y_tr_in, seed)
    x_all_bal, y_all_bal = _balanced_oversample(x_train, y_train, seed + 1)

    best_score, best_h = -np.inf, candidate_sizes[0]
    for h in candidate_sizes:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=h,
                        activation="relu",
                        solver="adam",
                        learning_rate="adaptive",
                        max_iter=1000,
                        random_state=seed,
                        early_stopping=False,
                    ),
                ),
            ]
        )
        try:
            pipe.fit(x_tr_in_bal, y_tr_in_bal)
            proba = pipe.predict_proba(x_va_in)
            pred = np.argmax(proba, axis=1)
            score = f1_score(y_va_in, pred, average="macro", zero_division=0)
        except Exception:
            score = -np.inf
        if score > best_score:
            best_score, best_h = score, h

    final = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=best_h,
                    activation="relu",
                    solver="adam",
                    learning_rate="adaptive",
                    max_iter=1000,
                    random_state=seed,
                    early_stopping=False,
                ),
            ),
        ]
    )
    final.fit(x_all_bal, y_all_bal)
    proba = final.predict_proba(x_val)
    pred = np.argmax(proba, axis=1)
    metrics = _score(y_val, pred, proba)
    metrics["hidden_nodes"] = float(best_h[0])
    return metrics


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #
def run_single(task: dict) -> dict:
    """One (algorithm, dataset, repeat, fold) run."""
    ds = load_dataset(task["csv_path"])
    tr, va = task["train_idx"], task["val_idx"]

    # Integer labels: vse.dataset stores y as one-hot AND y_int as ints.
    y_train = ds.y_int[tr]
    y_val = ds.y_int[va]
    x_train_raw = ds.x[tr]
    x_val_raw = ds.x[va]

    # Train-fold-only feature selection on high-d datasets.
    x_train, x_val, _ = _select_features(x_train_raw, y_train, x_val_raw, task["seed"])

    algo = task["algo"]
    t0 = time.perf_counter()
    if algo == "LGBM":
        metrics = _run_lgbm(
            x_train, y_train, x_val, y_val,
            task["seed"], task["optuna_trials"],
        )
    elif algo == "XGBoost":
        metrics = _run_xgb(
            x_train, y_train, x_val, y_val,
            task["seed"], task["optuna_trials"],
        )
    elif algo == "MLP":
        metrics = _run_mlp(x_train, y_train, x_val, y_val, task["seed"])
    else:
        raise ValueError(f"unknown algo {algo}")
    runtime = time.perf_counter() - t0

    return {
        "row": {
            "algo": algo,
            "dataset": task["dataset"],
            "repeat": task["repeat"],
            "fold": task["fold"],
            "run_id": task["run_id"],
            "seed": task["seed"],
            "f1_macro": metrics["f1_macro"],
            "auc": metrics["auc"],
            "acc": metrics["acc"],
            "hidden_nodes": metrics["hidden_nodes"],
            "runtime_sec": runtime,
            "best_fitness": np.nan,  # not applicable for tabular baselines
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "input_size": x_train.shape[1],
            "output_size": int(ds.output_size),
            "n_iters": 0,
        }
    }


def build_tasks(args) -> list:
    data_dir = args.data_dir or _here(config.CSV_DATA_DIR)
    tasks = []
    run_id = 0
    for dataset in args.datasets:
        csv_path = os.path.join(data_dir, dataset + ".csv")
        if not os.path.exists(csv_path):
            print(f"  !! missing CSV for '{dataset}' ({csv_path})", file=sys.stderr)
            continue
        ds = load_dataset(csv_path)
        splits = make_cv_splits(
            ds, args.repeats, args.folds, args.base_seed,
            mode=args.cv_mode,
            holdout_train_fraction=config.HOLDOUT_TRAIN_FRACTION,
        )
        for algo in args.algos:
            for repeat, fold, tr, va in splits:
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
                    "optuna_trials": args.optuna_trials,
                })
                run_id += 1
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--algos", nargs="*", default=B1_ALGOS, choices=B1_ALGOS)
    ap.add_argument("--datasets", nargs="*", default=config.SELECTED_DATASETS)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=_here("results_B1_baselines"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.repeats = 1
        args.folds = 3
        args.optuna_trials = 5
        args.jobs = min(args.jobs, 4)

    os.makedirs(args.results_dir, exist_ok=True)
    tasks = build_tasks(args)
    if not tasks:
        print("No runs to execute (check --datasets / CSVs).", file=sys.stderr)
        return 1

    print(f"B1 baselines | algos={args.algos} datasets={len(args.datasets)} "
          f"runs={len(tasks)} jobs={args.jobs}")

    t0 = time.perf_counter()
    outputs = Parallel(n_jobs=args.jobs, backend=config.JOBLIB_BACKEND, verbose=10)(
        delayed(run_single)(t) for t in tasks
    )
    wall = time.perf_counter() - t0

    df = pd.DataFrame([o["row"] for o in outputs]).sort_values(
        ["algo", "dataset", "repeat", "fold"]
    )
    df.to_csv(os.path.join(args.results_dir, "results.csv"), index=False)

    summary = (
        df.groupby(["algo", "dataset"])
        .agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
             auc_mean=("auc", "mean"), acc_mean=("acc", "mean"),
             hidden_mean=("hidden_nodes", "mean"),
             runtime_mean=("runtime_sec", "mean"), n=("run_id", "count"))
        .reset_index()
    )
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)

    with open(os.path.join(args.results_dir, "manifest.json"), "w") as fh:
        json.dump({
            "platform": platform.platform(),
            "python": platform.python_version(),
            "study": "B1_tabular_baselines",
            "algorithms": args.algos,
            "datasets": args.datasets,
            "cv_mode": args.cv_mode,
            "repeats": args.repeats,
            "folds": args.folds,
            "optuna_trials": args.optuna_trials,
            "feature_reduction_threshold": FEATURE_REDUCTION_THRESHOLD,
            "feature_k": FEATURE_K,
            "base_seed": args.base_seed,
            "n_runs": len(tasks),
            "wall_seconds": wall,
        }, fh, indent=2)

    print(f"\nFinished {len(tasks)} runs in {wall:.1f}s")
    print(f"  results : {args.results_dir}/results.csv")
    print(f"  summary : {args.results_dir}/summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
