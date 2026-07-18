#!/usr/bin/env python3
"""B7: Sparse-prior front-end for DCS--VSE on the six microarray datasets.

Reviewer's Triage B Priority 1 experiment. Study~G in main_v32.tex diagnoses the
LEUKEMIA macro-F1 gap as a structural mismatch between the dense-SLNN encoding
and the sparse-causal structure of microarray data (Lorenz curves of trained
weights track the uniform reference; LGBM/XGBoost feature-importance curves
concentrate >90 % of mass on <10 % of features). This script tests the natural
remedy: a univariate mutual-information feature-selection front-end that
reduces d to top-k before the DCS--VSE search.

Pipeline per CV fold (no leakage):
    1. Fit ``SelectKBest(mutual_info_classif, k=256)`` on the training split
       only.
    2. Transform train and validation splits with the fitted selector.
    3. Run DCS--VSE (algorithm key ``DCS_noVSE_d``, the proposed method) on
       the reduced (n, 256) tabular problem with identical hyperparameters,
       seed derivation, and CV partitions as the headline benchmark.
    4. Record macro-F1, AUC, accuracy, final hidden size, wall-clock, plus
       the kept-feature indices (for downstream stability analysis).

Datasets (microarray subgroup, d/n >= 50):
    ADENOCARCINOMA  (n = 76,  d = 9672)
    DLBCL           (n = 77,  d = 5469)
    LEUKEMIA1       (n = 72,  d = 5147)
    LEUKEMIA2       (n = 72,  d = 11225)
    PROSTATE6033    (n = 102, d = 6033)
    PROSTATE_TUMOR  (n = 102, d = 10509)

Total: 6 datasets x 50 stratified CV partitions = 300 fold-level runs.

Output
------
    results_B7_sparse_prior/results.csv
    results_B7_sparse_prior/summary.csv
    results_B7_sparse_prior/kept_features.json
    results_B7_sparse_prior/manifest.json

Usage
-----
    python b7_sparse_prior_frontend.py                  # default: 6 datasets, k=256
    python b7_sparse_prior_frontend.py --k 128          # try a tighter filter
    python b7_sparse_prior_frontend.py --datasets LEUKEMIA1 LEUKEMIA2
    python b7_sparse_prior_frontend.py --n-jobs 8       # control parallelism
    python b7_sparse_prior_frontend.py --quick          # 1 dataset x 5 folds smoke test

Runtime estimate
----------------
    Headline-benchmark wall-clock per fold for DCS--VSE on the microarray
    subgroup (Table 3, Table 5) is in the range 45-210 s (mean ~118 s at full
    feature dimensionality). With d reduced from 5K-11K to 256, the SLNN
    parameter count drops by roughly 20-40x, so the fitness-evaluation cost
    drops proportionally; expected per-fold wall-clock is 4-12 s.

    Sequential total: 300 folds x ~8 s = ~40 min on the reference 16-core
    hardware. With ``--n-jobs 8`` parallelism: ~5-10 min wall-clock.

    Allow up to 2 h on slower hardware (8 cores, no SSD, contention).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import config
from vse.dataset import load_dataset, make_cv_splits
from vse.evaluator import Evaluator
from vse.metrics import evaluate_on_fold


ALGO_NAME = "DCS_noVSE_d"  # proposed DCS--VSE (random-reinitialization length primitive)


# Microarray subgroup (d/n >= 50). Selection rationale: Study~G identifies these
# six datasets as the regime where the dense-SLNN encoding underperforms the
# tree-ensemble references on macro-F1 because the EA-driven search cannot
# recover the sparse-causal structure of the data-generating process.
DEFAULT_MICROARRAY_DATASETS = [
    "ADENOCARCINOMA",
    "DLBCL",
    "LEUKEMIA1",
    "LEUKEMIA2",
    "PROSTATE6033",
    "PROSTATE_TUMOR",
]


def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def derive_seed(base_seed: int, algo: str, dataset: str, repeat: int, fold: int) -> int:
    """Reproducible per-fold seed shared with the headline benchmark."""
    s = f"{base_seed}|{algo}|{dataset}|{repeat}|{fold}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


def _to_class_index(y: np.ndarray) -> np.ndarray:
    """Convert (n, K) one-hot labels (as ds.y is stored) to (n,) class indices.

    The DCS--VSE evaluator accepts the one-hot encoding; sklearn's
    ``mutual_info_classif`` expects a 1-D integer class vector.
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    return y.astype(int).ravel()


def apply_sparse_prior(
    x_train: np.ndarray,
    y_train_onehot: np.ndarray,
    x_val: np.ndarray,
    k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit SelectKBest(mutual_info_classif) on TRAIN ONLY; transform val with the fitted selector.

    Returns the reduced (x_train', x_val', kept_idx). No information leaks from
    the validation split into the feature-ranking computation.
    """
    y_train_idx = _to_class_index(y_train_onehot)
    k_eff = min(k, x_train.shape[1])
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=int(seed) % (2 ** 31 - 1)),
        k=k_eff,
    )
    x_train_red = selector.fit_transform(x_train, y_train_idx)
    x_val_red = selector.transform(x_val)
    kept_idx = selector.get_support(indices=True)
    return x_train_red, x_val_red, kept_idx


def run_single(task: dict) -> dict:
    """Run one CV fold of (sparse-prior filter -> DCS--VSE) end-to-end."""
    from vse.algorithms import get_algorithm

    t_total = time.perf_counter()
    ds = load_dataset(task["csv_path"])
    tr, va = task["train_idx"], task["val_idx"]
    x_train_full, y_train = ds.x[tr], ds.y[tr]
    x_val_full, y_val = ds.x[va], ds.y[va]

    # Sparse-prior front-end (TRAIN-ONLY fit, no leakage).
    t0 = time.perf_counter()
    x_train, x_val, kept_idx = apply_sparse_prior(
        x_train_full, y_train, x_val_full, task["k"], task["seed"]
    )
    filter_sec = time.perf_counter() - t0
    d_eff = x_train.shape[1]

    # Build the evaluator on the REDUCED feature space.
    evaluator = Evaluator(
        x_train, y_train, x_val, y_val,
        task["min_hidden"], task["max_hidden"],
    )
    options = {
        "popsize": task["popsize"],
        "max_nfe": task["max_nfe"],
        "lb": task["lb"],
        "ub": task["ub"],
        "min_hidden_size": task["min_hidden"],
        "max_hidden_size": task["max_hidden"],
        "inp": d_eff,           # reduced input dimensionality
        "outp": ds.output_size,
        "hidn": task["min_hidden"],
        "fobj": evaluator,
        "fobj_ablation": evaluator.ablation,
    }
    rng = np.random.default_rng(task["seed"])
    algo_fn = get_algorithm(ALGO_NAME)

    t1 = time.perf_counter()
    result = algo_fn(options, rng)
    search_sec = time.perf_counter() - t1
    metrics = evaluate_on_fold(result.best_x, x_val, y_val)
    total_sec = time.perf_counter() - t_total

    row = {
        # Distinct algo label so the rows do not collide with prior CSVs.
        "algo": f"DCS_VSE_sparse_k{task['k']}",
        "dataset": task["dataset"],
        "repeat": task["repeat"],
        "fold": task["fold"],
        "run_id": task["run_id"],
        "seed": task["seed"],
        "k_requested": int(task["k"]),
        "k_effective": int(d_eff),
        "f1_macro": float(metrics["f1_macro"]),
        "auc": float(metrics["auc"]),
        "acc": float(metrics["acc"]),
        "hidden_nodes": float(metrics["hidden_nodes"]),
        "runtime_sec": float(total_sec),
        "runtime_filter_sec": float(filter_sec),
        "runtime_search_sec": float(search_sec),
        "best_fitness": float(result.best_cost),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "input_size_full": ds.input_size,
        "input_size_reduced": int(d_eff),
        "output_size": ds.output_size,
        "n_iters": len(result.convergence_curve),
    }
    return {"row": row, "kept_features": [int(i) for i in kept_idx]}


def build_tasks(args) -> list:
    data_dir = args.data_dir or _here(config.CSV_DATA_DIR)
    tasks = []
    run_id = 0
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
            tasks.append({
                "algo": f"DCS_VSE_sparse_k{args.k}",
                "dataset": dataset,
                "csv_path": csv_path,
                "repeat": int(repeat),
                "fold": int(fold),
                "run_id": run_id,
                "train_idx": tr,
                "val_idx": va,
                "seed": derive_seed(
                    args.base_seed, f"DCS_VSE_sparse_k{args.k}",
                    dataset, repeat, fold,
                ),
                "k": int(args.k),
                "popsize": args.popsize,
                "max_nfe": args.max_nfe,
                "lb": args.lb,
                "ub": args.ub,
                "min_hidden": args.min_hidden,
                "max_hidden": args.max_hidden,
            })
            run_id += 1
            if args.quick and run_id >= 5:
                return tasks
        if args.quick:
            break
    return tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_MICROARRAY_DATASETS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--popsize", type=int, default=config.POPSIZE)
    ap.add_argument("--max-nfe", type=int, default=config.MAX_NFE)
    ap.add_argument("--lb", type=float, default=config.LB)
    ap.add_argument("--ub", type=float, default=config.UB)
    ap.add_argument("--min-hidden", type=int, default=1)
    ap.add_argument("--max-hidden", type=int, default=20)
    ap.add_argument("--k", type=int, default=256,
                    help="SelectKBest target dimensionality (default 256).")
    ap.add_argument("--n-jobs", type=int, default=config.N_JOBS if hasattr(config, "N_JOBS") else -1,
                    help="Parallel workers (-1 = all cores).")
    ap.add_argument("--out", default=_here("results_B7_sparse_prior"))
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 5 folds on the first listed dataset only.")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    tasks = build_tasks(args)
    if not tasks:
        print("No tasks built; check --datasets and --data-dir.", file=sys.stderr)
        return 2

    print(f"Built {len(tasks)} tasks across {len(set(t['dataset'] for t in tasks))} datasets.")
    print(f"Sparse-prior front-end: SelectKBest(mutual_info_classif, k={args.k}) "
          f"fit per training fold, no leakage.")
    print(f"Parallel workers: {args.n_jobs}")

    t_start = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_single)(t) for t in tasks
    )
    wall = time.perf_counter() - t_start
    print(f"\nWall-clock: {wall:.1f} s ({wall / 60:.1f} min)")

    # Per-fold rows.
    rows = [r["row"] for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)

    # Kept-feature sets (one entry per (dataset, repeat, fold)).
    kept = {}
    for r in results:
        key = f"{r['row']['dataset']}__r{r['row']['repeat']}_f{r['row']['fold']}"
        kept[key] = r["kept_features"]
    with open(os.path.join(args.out, "kept_features.json"), "w") as fh:
        json.dump(kept, fh)

    # Per-dataset summary.
    summary = (
        df.groupby(["algo", "dataset"])
        .agg(
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            auc_mean=("auc", "mean"),
            acc_mean=("acc", "mean"),
            hidden_mean=("hidden_nodes", "mean"),
            runtime_mean=("runtime_sec", "mean"),
            runtime_filter_mean=("runtime_filter_sec", "mean"),
            runtime_search_mean=("runtime_search_sec", "mean"),
            n=("f1_macro", "size"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\nPer-dataset summary:")
    print(summary.to_string(index=False))

    # Manifest for reproducibility.
    manifest = {
        "script": os.path.basename(__file__),
        "algo": f"DCS_VSE_sparse_k{args.k}",
        "datasets": list(df["dataset"].unique()),
        "k_requested": int(args.k),
        "n_folds_per_dataset": int(args.repeats * args.folds),
        "popsize": int(args.popsize),
        "max_nfe": int(args.max_nfe),
        "min_hidden": int(args.min_hidden),
        "max_hidden": int(args.max_hidden),
        "lb": float(args.lb),
        "ub": float(args.ub),
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

    # Helpful pointer for the manuscript update.
    print("\nNext step:")
    print(f"  Compare {os.path.join(args.out, 'summary.csv')} F1 against the "
          "DCS--VSE rows of Tables 3 and 5 in main_v32.tex on the six microarray "
          "datasets. If the sparse-prior F1 closes the gap to LGBM+Optuna by "
          "more than half on at least four of the six datasets, the Study~G "
          "future-work item is upgraded to a positive contribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
