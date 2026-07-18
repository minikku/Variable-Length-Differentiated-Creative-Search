#!/usr/bin/env python3
"""Parallel experiment runner -- Python port of ``Main_Parallel.m``.

For every (algorithm, dataset) pair it runs a *repeated stratified K-fold*
cross-validation; each fold is one independent run with its own deterministic
seed.  Per run it records F1 (macro), AUC (macro OvR), accuracy, wall-clock
runtime, the best network's hidden-node count, and the optimiser fitness.

Runs execute in parallel (``--jobs``, default 52) and are fully reproducible
across Windows/Linux from ``--base-seed``.

Examples
--------
    python convert_datasets.py            # once, to build the CSVs
    python main.py                        # full run from config.py defaults
    python main.py --algos DCS_VSE_DKA_opt_0_d CoDE_d --datasets IRIS \\
        --repeats 2 --folds 5 --jobs 8
    python main.py --quick                # tiny smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import time
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from vse.dataset import load_dataset, make_cv_splits
from vse.evaluator import Evaluator
from vse.metrics import evaluate_on_fold
from vse.algorithms import get_algorithm


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def derive_seed(base_seed, algo, dataset, repeat, fold) -> int:
    s = f"{base_seed}|{algo}|{dataset}|{repeat}|{fold}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


def run_single(task: dict) -> dict:
    """Execute one (algorithm, dataset, repeat, fold) run. Top-level for joblib."""
    ds = load_dataset(task["csv_path"])
    tr = task["train_idx"]
    va = task["val_idx"]
    x_train, t_train = ds.x[tr], ds.y[tr]
    x_val, t_val = ds.x[va], ds.y[va]

    evaluator = Evaluator(
        x_train, t_train, x_val, t_val,
        task["min_hidden"], task["max_hidden"],
    )

    options = {
        "popsize": task["popsize"],
        "max_nfe": task["max_nfe"],
        "lb": task["lb"],
        "ub": task["ub"],
        "min_hidden_size": task["min_hidden"],
        "max_hidden_size": task["max_hidden"],
        "inp": ds.input_size,
        "outp": ds.output_size,
        "hidn": task["min_hidden"],
        "fobj": evaluator,
        "fobj_ablation": evaluator.ablation,
    }

    rng = np.random.default_rng(task["seed"])
    algo_fn = get_algorithm(task["algo"])

    t0 = time.perf_counter()
    result = algo_fn(options, rng)
    runtime = time.perf_counter() - t0

    metrics = evaluate_on_fold(result.best_x, x_val, t_val)

    row = {
        "algo": task["algo"],
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
        "best_fitness": float(result.best_cost),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "input_size": ds.input_size,
        "output_size": ds.output_size,
        "n_iters": len(result.convergence_curve),
    }
    return {
        "row": row,
        "convergence": np.asarray(result.convergence_curve, dtype=float),
        "tag": f"{task['algo']}__{task['dataset']}__r{task['repeat']}_f{task['fold']}",
    }


def build_tasks(args) -> list:
    data_dir = args.data_dir or _here(config.CSV_DATA_DIR)
    tasks = []
    run_id = 0
    for dataset in args.datasets:
        csv_path = os.path.join(data_dir, dataset + ".csv")
        if not os.path.exists(csv_path):
            print(f"  !! missing CSV for '{dataset}' ({csv_path}); "
                  f"run convert_datasets.py first.", file=sys.stderr)
            continue
        ds = load_dataset(csv_path)
        splits = make_cv_splits(
            ds, args.repeats, args.folds, args.base_seed,
            mode=args.cv_mode, holdout_train_fraction=config.HOLDOUT_TRAIN_FRACTION,
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
                    "popsize": args.popsize,
                    "max_nfe": args.max_nfe,
                    "lb": args.lb,
                    "ub": args.ub,
                    "min_hidden": args.min_hidden,
                    "max_hidden": args.max_hidden,
                })
                run_id += 1
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the repeated-CV neuro-evolution experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--algos", nargs="*", default=config.SELECTED_ALGORITHMS)
    ap.add_argument("--datasets", nargs="*", default=config.SELECTED_DATASETS)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--popsize", type=int, default=config.POPSIZE)
    ap.add_argument("--max-nfe", type=int, default=config.MAX_NFE)
    ap.add_argument("--min-hidden", type=int, default=config.MIN_HIDDEN_SIZE)
    ap.add_argument("--max-hidden", type=int, default=config.MAX_HIDDEN_SIZE)
    ap.add_argument("--lb", type=float, default=config.LB)
    ap.add_argument("--ub", type=float, default=config.UB)
    ap.add_argument("--cv-mode", default=config.CV_MODE,
                    choices=["repeated_stratified_kfold",
                             "repeated_stratified_holdout"])
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--no-curves", action="store_true",
                    help="Do not save convergence curves (smaller output).")
    ap.add_argument("--list-algos", action="store_true",
                    help="Print the available algorithm names and exit.")
    ap.add_argument("--quick", action="store_true",
                    help="Tiny smoke test (overrides budget/CV for a fast run).")
    args = ap.parse_args(argv)

    if args.list_algos:
        from vse.algorithms import REGISTRY
        print("\n".join(sorted(REGISTRY)))
        return 0

    if args.quick:
        args.popsize = 16
        args.max_nfe = 320
        args.repeats = 1
        args.folds = 3
        args.jobs = min(args.jobs, 4)

    results_dir = args.results_dir or _here(config.RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    curves_dir = os.path.join(results_dir, "curves")
    if not args.no_curves:
        os.makedirs(curves_dir, exist_ok=True)

    tasks = build_tasks(args)
    if not tasks:
        print("No runs to execute (check --datasets / CSVs).", file=sys.stderr)
        return 1

    print(f"Platform: {platform.system()} {platform.release()} | "
          f"Python {platform.python_version()}")
    print(f"Algorithms : {len(args.algos)}  {args.algos}")
    print(f"Datasets   : {len(args.datasets)}  {args.datasets}")
    print(f"CV         : {args.cv_mode} -> {args.repeats}x{args.folds} "
          f"= {args.repeats * args.folds} runs/pair")
    print(f"Budget     : popsize={args.popsize} max_nfe={args.max_nfe} "
          f"hidden=[{args.min_hidden},{args.max_hidden}]")
    print(f"Total runs : {len(tasks)}  | parallel jobs: {args.jobs}\n")

    t0 = time.perf_counter()
    outputs = Parallel(n_jobs=args.jobs, backend=config.JOBLIB_BACKEND, verbose=10)(
        delayed(run_single)(t) for t in tasks
    )
    wall = time.perf_counter() - t0

    rows = [o["row"] for o in outputs]
    df = pd.DataFrame(rows).sort_values(["algo", "dataset", "repeat", "fold"])
    results_csv = os.path.join(results_dir, "results.csv")
    df.to_csv(results_csv, index=False)

    if not args.no_curves:
        by_pair = {}
        for o in outputs:
            algo = o["row"]["algo"]
            dataset = o["row"]["dataset"]
            by_pair.setdefault((algo, dataset), {})[
                f"r{o['row']['repeat']}_f{o['row']['fold']}"
            ] = o["convergence"]
        for (algo, dataset), arrays in by_pair.items():
            np.savez_compressed(
                os.path.join(curves_dir, f"{algo}__{dataset}.npz"), **arrays
            )

    # Per-(algo,dataset) summary table for quick inspection.
    summary = (
        df.groupby(["algo", "dataset"])
        .agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
             auc_mean=("auc", "mean"), acc_mean=("acc", "mean"),
             hidden_mean=("hidden_nodes", "mean"),
             runtime_mean=("runtime_sec", "mean"), n=("run_id", "count"))
        .reset_index()
    )
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    manifest = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "algorithms": args.algos,
        "datasets": args.datasets,
        "cv_mode": args.cv_mode,
        "repeats": args.repeats,
        "folds": args.folds,
        "popsize": args.popsize,
        "max_nfe": args.max_nfe,
        "min_hidden": args.min_hidden,
        "max_hidden": args.max_hidden,
        "lb": args.lb,
        "ub": args.ub,
        "base_seed": args.base_seed,
        "n_runs": len(tasks),
        "wall_seconds": wall,
    }
    with open(os.path.join(results_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nFinished {len(tasks)} runs in {wall:.1f}s")
    print(f"  results : {results_csv}")
    print(f"  summary : {os.path.join(results_dir, 'summary.csv')}")
    if not args.no_curves:
        print(f"  curves  : {curves_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())