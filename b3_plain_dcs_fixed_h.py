#!/usr/bin/env python3
"""B3: Plain DCS at fixed H with no VSE machinery and no length search.

The simplest possible variant of the proposed method, asked for by the advisor:
  - hidden size fixed at a chosen H (default H=1, since the H1to20 evidence
    shows the regularization frontier is at H=1 for these datasets),
  - no Addition/Elimination/Substitution operators (so no structural search),
  - no cross-length alignment (because all individuals share the same length
    by construction, the alignment rule is vacuous),
  - identical parametric DCS learning otherwise.

We obtain this exact behavior by running ``DCS_noVSE_d`` with
``min_hidden = max_hidden = H``.  ``DCS_noVSE_d`` already disables the three
structural operators (Addition / Elimination / Substitution); restricting
the hidden-size range to a single value freezes the variable-length search
to a fixed-length search.  This is the clean fixed-H baseline.

What this experiment answers
----------------------------
"Does the variable-length apparatus (alignment + structural operators) add
anything beyond a fixed-H DCS at the regularization sweet spot?"

If macro-F1 from this script matches or exceeds the DCS-VSE numbers in the
manuscript's headline Tables 3 and 4, then the variable-length apparatus is
not necessary on these datasets (the strongest version of the paper becomes
a fixed-H weight-search paper).  If the variable-length version is at least
as good without being told H in advance, that is the value-add of Study VI.

Usage
-----
    python b3_plain_dcs_fixed_h.py                  # H=1, default 9 datasets
    python b3_plain_dcs_fixed_h.py --hidden 2       # try H=2
    python b3_plain_dcs_fixed_h.py --hidden 1 2 3   # sweep
    python b3_plain_dcs_fixed_h.py --quick

Output
------
    results_B3_plain_dcs_fixed_H<h>/results.csv
    results_B3_plain_dcs_fixed_H<h>/summary.csv
    results_B3_plain_dcs_fixed_H<h>/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from vse.dataset import load_dataset, make_cv_splits
from vse.evaluator import Evaluator
from vse.metrics import evaluate_on_fold

ALGO_NAME = "DCS_noVSE_d"  # operator-disabled DCS already in the registry


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def derive_seed(base_seed, algo, dataset, repeat, fold) -> int:
    s = f"{base_seed}|{algo}|{dataset}|{repeat}|{fold}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


def run_single(task: dict) -> dict:
    from vse.algorithms import get_algorithm

    ds = load_dataset(task["csv_path"])
    tr, va = task["train_idx"], task["val_idx"]
    x_train, t_train = ds.x[tr], ds.y[tr]
    x_val, t_val = ds.x[va], ds.y[va]

    h = int(task["hidden"])
    evaluator = Evaluator(x_train, t_train, x_val, t_val, h, h)

    options = {
        "popsize": task["popsize"],
        "max_nfe": task["max_nfe"],
        "lb": task["lb"],
        "ub": task["ub"],
        "min_hidden_size": h,
        "max_hidden_size": h,
        "inp": ds.input_size,
        "outp": ds.output_size,
        "hidn": h,
        "fobj": evaluator,
        "fobj_ablation": evaluator.ablation,
    }
    rng = np.random.default_rng(task["seed"])

    algo_fn = get_algorithm(ALGO_NAME)
    t0 = time.perf_counter()
    result = algo_fn(options, rng)
    runtime = time.perf_counter() - t0
    metrics = evaluate_on_fold(result.best_x, x_val, t_val)

    row = {
        # Labeled as "PlainDCS_H<h>" so it does not collide with the existing
        # DCS_noVSE rows from results_H1to20 / results_H1to2.
        "algo": f"PlainDCS_H{h}",
        "dataset": task["dataset"],
        "repeat": task["repeat"],
        "fold": task["fold"],
        "run_id": task["run_id"],
        "seed": task["seed"],
        "fixed_hidden": h,
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
    return {"row": row}


def build_tasks(args, hidden) -> list:
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
                "algo": f"PlainDCS_H{hidden}",
                "dataset": dataset,
                "csv_path": csv_path,
                "repeat": int(repeat),
                "fold": int(fold),
                "run_id": run_id,
                "train_idx": tr,
                "val_idx": va,
                "seed": derive_seed(
                    args.base_seed, f"PlainDCS_H{hidden}",
                    dataset, repeat, fold,
                ),
                "popsize": args.popsize,
                "max_nfe": args.max_nfe,
                "lb": args.lb,
                "ub": args.ub,
                "hidden": int(hidden),
            })
            run_id += 1
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--hidden", nargs="+", type=int, default=[1],
                    help="Fixed hidden size(s). One results dir per value.")
    ap.add_argument("--datasets", nargs="*", default=config.SELECTED_DATASETS)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--popsize", type=int, default=config.POPSIZE)
    ap.add_argument("--max-nfe", type=int, default=config.MAX_NFE)
    ap.add_argument("--lb", type=float, default=config.LB)
    ap.add_argument("--ub", type=float, default=config.UB)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-root", default=None,
                    help="Parent directory for results_B3_plain_dcs_fixed_H<h> dirs.")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.popsize = 16
        args.max_nfe = 320
        args.repeats = 1
        args.folds = 3
        args.jobs = min(args.jobs, 4)

    results_root = args.results_root or os.path.dirname(_here(""))
    overall_t0 = time.perf_counter()

    for h in args.hidden:
        results_dir = os.path.join(
            args.results_root or _here(""),
            f"results_B3_plain_dcs_fixed_H{h}",
        )
        os.makedirs(results_dir, exist_ok=True)
        tasks = build_tasks(args, h)
        if not tasks:
            print(f"No runs to execute for H={h}.", file=sys.stderr)
            continue

        print(f"\nB3 plain-DCS H={h} | datasets={len(args.datasets)} "
              f"runs={len(tasks)} jobs={args.jobs}")

        t0 = time.perf_counter()
        outputs = Parallel(
            n_jobs=args.jobs, backend=config.JOBLIB_BACKEND, verbose=10,
        )(delayed(run_single)(t) for t in tasks)
        wall = time.perf_counter() - t0

        df = pd.DataFrame([o["row"] for o in outputs]).sort_values(
            ["algo", "dataset", "repeat", "fold"]
        )
        df.to_csv(os.path.join(results_dir, "results.csv"), index=False)

        summary = (
            df.groupby(["algo", "dataset"])
            .agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
                 auc_mean=("auc", "mean"), acc_mean=("acc", "mean"),
                 hidden_mean=("hidden_nodes", "mean"),
                 runtime_mean=("runtime_sec", "mean"), n=("run_id", "count"))
            .reset_index()
        )
        summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

        with open(os.path.join(results_dir, "manifest.json"), "w") as fh:
            json.dump({
                "platform": platform.platform(),
                "python": platform.python_version(),
                "study": "B3_plain_dcs_fixed_H",
                "algorithm": ALGO_NAME,
                "fixed_hidden": int(h),
                "datasets": args.datasets,
                "cv_mode": args.cv_mode,
                "repeats": args.repeats,
                "folds": args.folds,
                "popsize": args.popsize,
                "max_nfe": args.max_nfe,
                "lb": args.lb,
                "ub": args.ub,
                "base_seed": args.base_seed,
                "n_runs": len(tasks),
                "wall_seconds": wall,
            }, fh, indent=2)

        print(f"  H={h}: {len(tasks)} runs in {wall:.1f}s -> {results_dir}/")

    overall_wall = time.perf_counter() - overall_t0
    print(f"\nB3 total wall time across {len(args.hidden)} H value(s): {overall_wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
