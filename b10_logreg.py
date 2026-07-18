#!/usr/bin/env python3
"""B10: footprint-matched deterministic baselines (logistic regression, stump).

Answers the reviewers' load-bearing question: at the operating point DCS--VSE
returns (median hidden size H=1, i.e. essentially one nonlinear hyperplane), does
a 30,000-evaluation stochastic search buy anything over a convex solver of the
same footprint? This script runs regularized logistic regression (and, as a
one-split floor, a decision stump) under the identical pipeline so the answer is
a paired comparison, not a guess.

Pipeline parity
---------------
Inherited from the tree baseline (b8_matched_complexity_trees.py):
    * identical CV partitions           : vse.dataset.make_cv_splits(...)
    * identical per-fold seed derivation : derive_seed(base, algo, ds, r, f)
    * identical feature front-end        : SelectKBest(mutual_info_classif, k=256)
                                           on datasets with d > 1000
    * identical scoring                  : macro-F1, one-vs-rest macro AUC, accuracy
    * identical one-hot -> class-index    : _to_class_index
The only additions are (i) StandardScaler on the selected features (fit on the
training rows only), because linear models are scale-sensitive, and (ii) the
model itself.

Footprint
---------
For a binary task an H=1 sigmoid SLNN is one hyperplane, which is the same model
class as logistic regression; for C classes, multinomial LR uses C(d+1)
parameters. The reported ``n_params`` lets the reader place LR on the same
(macro-F1, parameter-count) plane as the ~3,000-parameter SLNN. LR needs no
per-dataset stochastic search: its only hyperparameter (the L2 strength C) is
chosen by an internal cross-validation on the training fold, so any accuracy it
reaches is what a deterministic convex solver delivers "for free."

Models (``--algos``)
    LogReg   : L2 logistic regression, C chosen by internal 5-fold CV
               (LogisticRegressionCV), class_weight='balanced'. [default]
    LogRegL1 : L1 (sparse) logistic regression, saga solver; a natural
               sparse-linear comparison for the microarray regime.
    Stump    : depth-1 decision tree (one axis-aligned split), the num_leaves=2
               deterministic floor.

Output
------
    results_B10_logreg/results.csv     (one row per fold)
    results_B10_logreg/summary.csv     (algo x dataset means, b8-compatible schema)
    results_B10_logreg/manifest.json

Usage
-----
    python b10_logreg.py                       # LogReg, all 18 datasets, 5x10 CV
    python b10_logreg.py --algos LogReg LogRegL1 Stump
    python b10_logreg.py --datasets WDBC ILPD
    python b10_logreg.py --quick               # smoke test: 5 folds, first dataset
    python b10_logreg.py --jobs 8              # parallel over folds (CPU)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier

import config
from vse.dataset import load_dataset, make_cv_splits

# Reuse the EXACT helpers from the tree baseline so the pipeline is identical.
from b8_matched_complexity_trees import (
    derive_seed,
    _to_class_index,
    _select_features,
    _score,
    FEATURE_REDUCTION_THRESHOLD,
    FEATURE_K,
)

warnings.filterwarnings("ignore")

DEFAULT_DATASETS = [
    "ADENOCARCINOMA", "COLON_ALON", "DLBCL", "DYSLEXIA", "DYSLEXIA_10p",
    "HABERMAN_SURVIVAL", "HEART_CLEVELAND", "HEART_FAILURE", "HEPATITIS", "ILPD",
    "LEUKEMIA1", "LEUKEMIA2", "PARKINSONS", "PIMA_DIABETES", "PROSTATE6033",
    "PROSTATE_TUMOR", "VERTEBRAL_COLUMN", "WDBC",
]
ALL_ALGOS = ["LogReg", "LogRegL1", "Stump"]


def _fit_predict(algo: str, x_train, y_train, x_val, seed: int):
    """Return (proba, n_params, n_nonzero) for the chosen deterministic model."""
    n_classes = int(np.max(y_train)) + 1
    # internal CV folds bounded by the smallest class count on the training fold
    min_class = int(np.min(np.bincount(y_train, minlength=n_classes)))
    inner_cv = max(2, min(5, min_class))

    if algo == "LogReg":
        clf = LogisticRegressionCV(
            Cs=10, cv=inner_cv, penalty="l2", solver="lbfgs",
            class_weight="balanced", scoring="f1_macro",
            max_iter=1000, n_jobs=1, random_state=int(seed),
        )
    elif algo == "LogRegL1":
        clf = LogisticRegressionCV(
            Cs=10, cv=inner_cv, penalty="l1", solver="saga",
            class_weight="balanced", scoring="f1_macro",
            max_iter=1000, n_jobs=1, random_state=int(seed),
        )
    elif algo == "Stump":
        clf = DecisionTreeClassifier(
            max_depth=1, class_weight="balanced", random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown algo {algo}")

    clf.fit(x_train, y_train)
    proba = np.asarray(clf.predict_proba(x_val))
    # align proba columns to 0..K-1 (stratified CV keeps all classes in train)
    if proba.shape[1] != n_classes and hasattr(clf, "classes_"):
        full = np.zeros((proba.shape[0], n_classes))
        for j, c in enumerate(np.asarray(clf.classes_).astype(int)):
            full[:, c] = proba[:, j]
        proba = full

    if hasattr(clf, "coef_"):
        n_params = int(clf.coef_.size + clf.intercept_.size)
        n_nonzero = int(np.count_nonzero(clf.coef_))
    else:  # stump
        n_params = int(clf.get_n_leaves())      # 2 leaves for a depth-1 split
        n_nonzero = n_params
    return proba, n_params, n_nonzero


def run_single(task: dict) -> dict:
    ds = load_dataset(task["csv_path"])
    tr, va = task["train_idx"], task["val_idx"]
    x_train, y_train = ds.x[tr], _to_class_index(ds.y[tr])
    x_val, y_val = ds.x[va], _to_class_index(ds.y[va])

    if task.get("match_slnn"):
        # Strict match to the headline SLNN pipeline: full features and the
        # per-feature min-max [0,1] scaling already applied by
        # vse.dataset.load_dataset. No MI reduction and no extra standardization,
        # so logistic regression sees the identical input the SLNN receives.
        pass
    else:
        # feature reduction (identical to the tree/headline pipeline)
        x_train, x_val = _select_features(x_train, y_train, x_val, task["seed"])
        # standardize (linear models are scale-sensitive); fit on training rows only
        scaler = StandardScaler().fit(x_train)
        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)

    t0 = time.perf_counter()
    proba, n_params, n_nonzero = _fit_predict(
        task["algo"], x_train, y_train, x_val, task["seed"])
    runtime = time.perf_counter() - t0

    pred = np.argmax(proba, axis=1)
    metrics = _score(y_val, pred, proba)
    row = {
        "algo": task["algo"],
        "dataset": task["dataset"],
        "repeat": task["repeat"],
        "fold": task["fold"],
        "run_id": task["run_id"],
        "seed": task["seed"],
        "f1_macro": float(metrics["f1_macro"]),
        "auc": float(metrics["auc"]),
        "acc": float(metrics["acc"]),
        "n_params": int(n_params),
        "n_nonzero": int(n_nonzero),
        "runtime_sec": float(runtime),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "input_size": int(ds.input_size),
        "output_size": int(ds.output_size),
        "n_features_used": int(x_train.shape[1]),
    }
    return {"row": row}


def build_tasks(args) -> list:
    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), config.CSV_DATA_DIR)
    tasks, run_id = [], 0
    algos = args.algos or ["LogReg"]
    for dataset in args.datasets:
        csv_path = os.path.join(data_dir, dataset + ".csv")
        if not os.path.exists(csv_path):
            print(f"  !! missing CSV for '{dataset}'", file=sys.stderr)
            continue
        ds = load_dataset(csv_path)
        splits = make_cv_splits(
            ds, args.repeats, args.folds, args.base_seed,
            mode=args.cv_mode, holdout_train_fraction=config.HOLDOUT_TRAIN_FRACTION)
        for repeat, fold, tr, va in splits:
            for algo in algos:
                tasks.append({
                    "algo": algo, "dataset": dataset, "csv_path": csv_path,
                    "repeat": int(repeat), "fold": int(fold), "run_id": run_id,
                    "train_idx": tr, "val_idx": va,
                    "match_slnn": bool(getattr(args, "match_slnn", False)),
                    "seed": derive_seed(args.base_seed, algo, dataset, repeat, fold),
                })
                run_id += 1
                if args.quick and run_id >= 5:
                    return tasks
        if args.quick:
            break
    return tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--algos", nargs="*", choices=ALL_ALGOS, default=["LogReg"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--jobs", type=int, default=getattr(config, "N_JOBS", -1),
                    help="Parallel workers over folds (-1 = all cores).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--match-slnn", action="store_true",
                    help="Strict match to the SLNN input: full features (no MI) and "
                         "min-max [0,1] scaling only (no z-score).")
    ap.add_argument("--quick", action="store_true", help="Smoke test: 5 folds, first dataset.")
    args = ap.parse_args(argv)

    if args.out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.out = os.path.join(here, "results_B10_logreg_matched"
                                if args.match_slnn else "results_B10_logreg")
    os.makedirs(args.out, exist_ok=True)
    tasks = build_tasks(args)
    if not tasks:
        print("No tasks built; check --datasets / --data-dir / --algos.", file=sys.stderr)
        return 2

    n_ds = len(set(t["dataset"] for t in tasks))
    print(f"Built {len(tasks)} tasks: {n_ds} datasets x {len(set(t['algo'] for t in tasks))} "
          f"algorithms x {args.repeats*args.folds} folds.")
    print(f"Models: {args.algos} | feature front-end: MI k={FEATURE_K} on d>{FEATURE_REDUCTION_THRESHOLD}; "
          f"features standardized. No stochastic search (deterministic convex fits).")

    t0 = time.perf_counter()
    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_single)(t) for t in tasks)
    wall = time.perf_counter() - t0
    print(f"\nWall-clock: {wall:.1f} s ({wall/60:.1f} min)")

    df = pd.DataFrame([r["row"] for r in results])
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    summary = (df.groupby(["algo", "dataset"]).agg(
        f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
        auc_mean=("auc", "mean"), acc_mean=("acc", "mean"),
        params_mean=("n_params", "mean"), nonzero_mean=("n_nonzero", "mean"),
        runtime_mean=("runtime_sec", "mean"), n=("f1_macro", "size"),
    ).reset_index())
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\nPer-dataset summary:")
    print(summary[["algo", "dataset", "f1_mean", "auc_mean", "acc_mean",
                   "params_mean", "runtime_mean"]].to_string(index=False))
    for algo, g in df.groupby("algo"):
        print(f"\nAggregate {algo}: F1={g['f1_macro'].mean():.4f}  AUC={g['auc'].mean():.4f}  "
              f"ACC={g['acc'].mean():.4f}  mean_params={g['n_params'].mean():.0f}  "
              f"runtime={g['runtime_sec'].mean():.2f}s/fold")

    manifest = {
        "script": os.path.basename(__file__), "algorithms": args.algos,
        "datasets": sorted(set(df["dataset"].unique())),
        "n_folds_per_dataset": int(args.repeats * args.folds),
        "tuning": "internal CV for L2/L1 strength; no stochastic search",
        "feature_reduction_threshold": FEATURE_REDUCTION_THRESHOLD,
        "feature_k": FEATURE_K,
        "match_slnn": bool(args.match_slnn),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
