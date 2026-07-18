#!/usr/bin/env python3
"""B4: Tuned MLP at fixed H=1 -- head-to-head against B3 (plain DCS at H=1).

The Optuna suggestion names MUST exactly match scikit-learn's MLPClassifier
parameter names, because ``study.best_params`` is unpacked directly into
``MLPClassifier(...)`` via ``**best``.  Using short Optuna names like "lr"
produces:
    TypeError: MLPClassifier got unexpected keyword argument 'lr'
The canonical sklearn names ``learning_rate_init``, ``alpha``, ``beta_1`` are
used instead.  A defensive whitelist filter after the search ensures no stray
unknown key ever reaches MLPClassifier.
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

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from vse.dataset import load_dataset, make_cv_splits
from b1_tabular_baselines import (
    _balanced_oversample,
    _select_features,
    _score,
    _inner_split,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

warnings.filterwarnings("ignore")

ALGO_NAME = "MLP_H1_Tuned"
FIXED_HIDDEN_SIZE = 1
DEFAULT_OPTUNA_TRIALS = 40
_MLP_ALLOWED_PARAMS = {
    "activation", "learning_rate_init", "alpha", "beta_1", "beta_2",
    "epsilon", "batch_size", "learning_rate",
}


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def derive_seed(base_seed, algo, dataset, repeat, fold):
    s = f"{base_seed}|{algo}|{dataset}|{repeat}|{fold}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


def _run_tuned_mlp_h1(x_train, y_train, x_val, y_val, seed, optuna_trials):
    inner_tr, inner_va = _inner_split(y_train, seed)
    x_tr_in, x_va_in = x_train[inner_tr], x_train[inner_va]
    y_tr_in, y_va_in = y_train[inner_tr], y_train[inner_va]

    x_tr_in_bal, y_tr_in_bal = _balanced_oversample(x_tr_in, y_tr_in, seed)
    x_all_bal, y_all_bal = _balanced_oversample(x_train, y_train, seed + 1)

    def make_mlp(params):
        clean = {k: v for k, v in params.items() if k in _MLP_ALLOWED_PARAMS}
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(FIXED_HIDDEN_SIZE,),
                solver="adam",
                max_iter=2000,
                random_state=seed,
                early_stopping=False,
                **clean,
            )),
        ])

    def objective(trial):
        params = dict(
            activation=trial.suggest_categorical(
                "activation", ["relu", "tanh", "logistic"]),
            learning_rate_init=trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-1, log=True),
            alpha=trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            beta_1=trial.suggest_float("beta_1", 0.5, 0.99),
        )
        try:
            pipe = make_mlp(params)
            pipe.fit(x_tr_in_bal, y_tr_in_bal)
            proba = pipe.predict_proba(x_va_in)
            pred = np.argmax(proba, axis=1)
            return -f1_score(y_va_in, pred, average="macro", zero_division=0)
        except Exception:
            return 0.0

    if optuna is not None and optuna_trials > 0:
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        best = dict(study.best_params)
    else:
        best = dict(activation="relu", learning_rate_init=0.01,
                    alpha=1e-4, beta_1=0.9)

    final = make_mlp(best)
    final.fit(x_all_bal, y_all_bal)
    proba = final.predict_proba(x_val)
    pred = np.argmax(proba, axis=1)
    metrics = _score(y_val, pred, proba)
    metrics["hidden_nodes"] = float(FIXED_HIDDEN_SIZE)
    return metrics


def run_single(task):
    ds = load_dataset(task["csv_path"])
    tr, va = task["train_idx"], task["val_idx"]
    y_train, y_val = ds.y_int[tr], ds.y_int[va]
    x_train, x_val, _ = _select_features(ds.x[tr], y_train, ds.x[va], task["seed"])
    t0 = time.perf_counter()
    metrics = _run_tuned_mlp_h1(
        x_train, y_train, x_val, y_val, task["seed"], task["optuna_trials"])
    runtime = time.perf_counter() - t0
    return {
        "row": {
            "algo": ALGO_NAME,
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
            "best_fitness": np.nan,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "input_size": x_train.shape[1],
            "output_size": int(ds.output_size),
            "n_iters": 0,
        }
    }


def build_tasks(args):
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
        for repeat, fold, tr, va in splits:
            tasks.append({
                "dataset": dataset,
                "csv_path": csv_path,
                "repeat": int(repeat),
                "fold": int(fold),
                "run_id": run_id,
                "train_idx": tr,
                "val_idx": va,
                "seed": derive_seed(args.base_seed, ALGO_NAME, dataset, repeat, fold),
                "optuna_trials": args.optuna_trials,
            })
            run_id += 1
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="B4: Tuned MLP at fixed H=1, head-to-head against B3.",
    )
    ap.add_argument("--datasets", nargs="*", default=config.SELECTED_DATASETS)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=_here("results_B4_mlp_fixed_H1"))
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
        print("No runs to execute.", file=sys.stderr)
        return 1

    print(f"B4 MLP_H1_Tuned | datasets={len(args.datasets)} runs={len(tasks)} "
          f"jobs={args.jobs} optuna_trials={args.optuna_trials}")
    if optuna is None:
        print("  !! Optuna not installed; using fixed defaults.", file=sys.stderr)

    t0 = time.perf_counter()
    outputs = Parallel(
        n_jobs=args.jobs, backend=config.JOBLIB_BACKEND, verbose=10,
    )(delayed(run_single)(t) for t in tasks)
    wall = time.perf_counter() - t0

    df = pd.DataFrame([o["row"] for o in outputs]).sort_values(
        ["dataset", "repeat", "fold"]
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

    manifest = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "study": "B4_mlp_fixed_H1",
        "algorithm": ALGO_NAME,
        "fixed_hidden_size": FIXED_HIDDEN_SIZE,
        "datasets": args.datasets,
        "cv_mode": args.cv_mode,
        "repeats": args.repeats,
        "folds": args.folds,
        "optuna_trials": args.optuna_trials,
        "base_seed": args.base_seed,
        "n_runs": len(tasks),
        "wall_seconds": wall,
        "paired_with": "results_B3_plain_dcs_fixed_H1",
    }
    with open(os.path.join(args.results_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nFinished {len(tasks)} runs in {wall:.1f}s")
    print(f"  results : {args.results_dir}/results.csv")
    print(f"  summary : {args.results_dir}/summary.csv")
    print(f"  Next   : python b4_compare_with_b3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
