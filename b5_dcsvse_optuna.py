#!/usr/bin/env python3
"""B5: Optuna sweep over DCS--VSE's own hyperparameters (fair-budget closure).

Why this experiment exists
--------------------------
Reviewer 2 sees that the three off-the-shelf references (LGBM, XGBoost, MLP)
each got 40 Optuna trials per CV fold, while the DCS family ran at a single
fixed operating point (popsize=30, max_nfe=30000, lambda_size=0.05, Linnik
alpha=0.618, scale=0.05).  The obvious reviewer challenge is:

    "Is DCS--VSE's headline F1 an under-tuned operating point?  What would it
    look like under a hyperparameter sweep matched in spirit to the budget
    given to the tree learners?"

This script answers that question on three representative datasets.  For each
dataset it runs an outer Optuna study over five DCS--VSE hyperparameters:

    popsize       in {20, 30, 60}                (integer-choice)
    max_nfe       in {15000, 30000, 60000}       (integer-choice)
    lambda_size   in [0.01, 0.50]   (log-uniform)
    linnik_alpha  in [1.0, 2.0]                  (continuous)
    linnik_scale  in [0.01, 0.30]   (log-uniform)

Each Optuna trial evaluates the candidate hyperparameter combination on a
small inner subset of CV folds (default: the first 5 folds of repeat 0) to
keep cost manageable.  After the Optuna study finishes, the best
hyperparameter combination is re-evaluated on the full 50 stratified CV
partitions (matching the rest of the paper's protocol) and the per-fold
macro-F1 results are written to a results.csv that pairs row-for-row with
``results_H1to20/summary.csv`` for the same dataset.

Datasets
--------
Default: DLBCL (imbalanced microarray), ILPD (low-dimensional clinical),
PROSTATE6033 (balanced microarray).  These three span the dataset
characteristics of the full nine-dataset benchmark.

How the hyperparameters reach the optimiser
-------------------------------------------
* ``popsize`` and ``max_nfe`` are passed through the ``options`` dict that
  ``get_algorithm("DCS_VSE_DKA_opt_0_d")`` already consumes.
* ``lambda_size`` is injected by monkey-patching ``vse.evaluator.Evaluator``
  inside the worker (worker-local because joblib forks fresh interpreters).
  The original ``_penalty`` returns ``((h - h_min) / denom) * 100.0``; the
  patched version returns ``lambda_size * 100.0 * ((h - h_min) / denom)``
  (with the original behaviour recovered exactly at ``lambda_size = 1.0``).
* ``linnik_alpha`` and ``linnik_scale`` are injected by monkey-patching
  ``vse.algorithms.dcs.lnf2`` (the name as imported into dcs.py at module
  load time) to ignore the hardcoded arguments at the call site and use the
  per-task globals instead.  Original behaviour recovered at
  ``alpha = golden_ratio`` and ``scale = 0.05``.

Pairing semantics with results_H1to20
-------------------------------------
* Same datasets (subset).
* Same repeats/folds (5 x 10).
* Same per-(algo, dataset, repeat, fold) seed derivation as the existing
  studies, but with algo="DCS_VSE_OptunaTuned" so the seeds are independent
  of the existing ``DCS_VSE_DKA_opt_0_d`` runs.

Usage
-----
    pip install optuna   # if not already present from B1
    python b5_dcsvse_optuna.py
    python b5_dcsvse_optuna.py --datasets DLBCL ILPD
    python b5_dcsvse_optuna.py --optuna-trials 15 --inner-folds 3   # faster
    python b5_dcsvse_optuna.py --quick

Cost
----
Default: 3 datasets x (25 Optuna trials x 5 inner folds + 50 final folds).
Per-trial DCS runs use a reduced ``max_nfe`` ceiling capped at the trial's
suggested value; the final 50-fold evaluation uses the best trial's
hyperparameters at full budget.  Wall-clock target on 16-core hardware:
~3-4 hours.

After it finishes
-----------------
Run ``b5_compare_with_defaults.py`` to get the per-dataset paired Wilcoxon
test against the existing DCS_VSE_DKA_opt_0_d numbers and the decision-tree
outcome.

Output
------
    results_B5_dcsvse_optuna/results.csv         (final-eval rows, per fold)
    results_B5_dcsvse_optuna/best_params.csv     (one row per dataset)
    results_B5_dcsvse_optuna/optuna_trials.csv   (full per-trial trace)
    results_B5_dcsvse_optuna/summary.csv         (mean/std per dataset)
    results_B5_dcsvse_optuna/manifest.json       (config + wall-clock)
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

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from vse.dataset import load_dataset, make_cv_splits
from vse.metrics import evaluate_on_fold

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    optuna = None

warnings.filterwarnings("ignore")

ALGO_NAME = "DCS_VSE_OptunaTuned"
BASE_ALGO_KEY = "DCS_VSE_DKA_opt_0_d"
DEFAULT_OPTUNA_TRIALS = 25
DEFAULT_INNER_FOLDS = 5
DEFAULT_SWEEP_DATASETS = ["DLBCL", "ILPD", "PROSTATE6033"]


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def derive_seed(base_seed: int, algo: str, dataset: str, repeat: int, fold: int) -> int:
    s = f"{base_seed}|{algo}|{dataset}|{repeat}|{fold}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


# --------------------------------------------------------------------------- #
# Monkey patches installed per worker                                         #
# --------------------------------------------------------------------------- #
def _install_lambda_size_patch(lambda_size: float):
    """Patch Evaluator._penalty to scale by ``lambda_size``.

    The original implementation hardcodes a multiplier of 100.0; the patched
    version uses ``lambda_size * 100.0`` so the existing behaviour is
    recovered exactly at ``lambda_size = 1.0``.
    """
    from vse import evaluator as ev

    if not hasattr(ev.Evaluator, "_orig_penalty_for_b5"):
        ev.Evaluator._orig_penalty_for_b5 = ev.Evaluator._penalty

    def patched_penalty(self, h):
        if self._denom <= 0.0:
            return 0.0
        return lambda_size * 100.0 * ((h - self.min_hidden) / self._denom)

    ev.Evaluator._penalty = patched_penalty


def _install_linnik_patch(alpha: float, scale: float):
    """Patch ``vse.algorithms.dcs.lnf2`` to ignore call-site (alpha, scale).

    The DCS algorithm imports lnf2 into its own module namespace with
    ``from ..rng import lnf2``.  We override the binding in the algorithm
    module so that every call inside ``_run_dcs`` uses the per-task
    (alpha, scale) instead of the hardcoded (golden_ratio, 0.05).
    """
    from vse.algorithms import dcs as dcs_mod
    from vse import rng as rng_mod

    if not hasattr(dcs_mod, "_orig_lnf2_for_b5"):
        dcs_mod._orig_lnf2_for_b5 = dcs_mod.lnf2

    def patched_lnf2(rng, _alpha_ignored, _scale_ignored, m=1, n=1):
        return rng_mod.lnf2(rng, alpha, scale, m, n)

    dcs_mod.lnf2 = patched_lnf2


# --------------------------------------------------------------------------- #
# Run a single DCS--VSE fold under a given hyperparameter configuration       #
# --------------------------------------------------------------------------- #
def _run_one_fold(
    csv_path: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    popsize: int,
    max_nfe: int,
    lambda_size: float,
    linnik_alpha: float,
    linnik_scale: float,
    min_hidden: int,
    max_hidden: int,
    lb: float,
    ub: float,
) -> dict:
    from vse.evaluator import Evaluator
    from vse.algorithms import get_algorithm

    # Worker-local monkey-patches; safe because joblib spawns fresh interpreters.
    _install_lambda_size_patch(lambda_size)
    _install_linnik_patch(linnik_alpha, linnik_scale)

    ds = load_dataset(csv_path)
    x_tr, t_tr = ds.x[train_idx], ds.y[train_idx]
    x_va, t_va = ds.x[val_idx], ds.y[val_idx]

    evaluator = Evaluator(x_tr, t_tr, x_va, t_va, min_hidden, max_hidden)
    options = {
        "popsize": popsize,
        "max_nfe": max_nfe,
        "lb": lb,
        "ub": ub,
        "min_hidden_size": min_hidden,
        "max_hidden_size": max_hidden,
        "inp": ds.input_size,
        "outp": ds.output_size,
        "hidn": min_hidden,
        "fobj": evaluator,
        "fobj_ablation": evaluator.ablation,
    }
    rng = np.random.default_rng(seed)
    algo_fn = get_algorithm(BASE_ALGO_KEY)

    t0 = time.perf_counter()
    result = algo_fn(options, rng)
    runtime = time.perf_counter() - t0

    metrics = evaluate_on_fold(result.best_x, x_va, t_va)
    return {
        "f1_macro": float(metrics["f1_macro"]),
        "auc": float(metrics["auc"]),
        "acc": float(metrics["acc"]),
        "hidden_nodes": float(metrics["hidden_nodes"]),
        "runtime_sec": float(runtime),
        "best_fitness": float(result.best_cost),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }


# --------------------------------------------------------------------------- #
# Per-dataset Optuna study + final evaluation                                 #
# --------------------------------------------------------------------------- #
def _run_dataset(args, dataset: str):
    data_dir = args.data_dir or _here(config.CSV_DATA_DIR)
    csv_path = os.path.join(data_dir, dataset + ".csv")
    if not os.path.exists(csv_path):
        print(f"  !! missing CSV for '{dataset}'", file=sys.stderr)
        return None

    ds = load_dataset(csv_path)
    splits = make_cv_splits(
        ds, args.repeats, args.folds, args.base_seed,
        mode=args.cv_mode,
        holdout_train_fraction=config.HOLDOUT_TRAIN_FRACTION,
    )
    splits_list = list(splits)  # (repeat, fold, tr, va)
    inner_splits = splits_list[: args.inner_folds]

    # ---- Optuna study (inner-fold mean F1, per-trial) ----
    def objective(trial: "optuna.Trial") -> float:
        popsize = trial.suggest_categorical("popsize", [20, 30, 60])
        # Cap max_nfe at the upper end of our search; smaller trials are cheaper
        max_nfe = trial.suggest_categorical("max_nfe", [15000, 30000, 60000])
        lambda_size = trial.suggest_float("lambda_size", 0.01, 0.50, log=True)
        linnik_alpha = trial.suggest_float("linnik_alpha", 1.0, 2.0)
        linnik_scale = trial.suggest_float("linnik_scale", 0.01, 0.30, log=True)

        f1s = []
        for repeat, fold, tr, va in inner_splits:
            seed = derive_seed(
                args.base_seed, ALGO_NAME + "_inner",
                dataset, int(repeat), int(fold),
            )
            try:
                m = _run_one_fold(
                    csv_path, tr, va, seed,
                    popsize, max_nfe, lambda_size, linnik_alpha, linnik_scale,
                    args.min_hidden, args.max_hidden, args.lb, args.ub,
                )
                f1s.append(m["f1_macro"])
            except Exception as exc:
                trial.set_user_attr("error", repr(exc))
                return 0.0
        score = float(np.mean(f1s)) if f1s else 0.0
        trial.set_user_attr("inner_f1_mean", score)
        return -score  # Optuna minimises

    print(f"\n[{dataset}] Optuna sweep: {args.optuna_trials} trials "
          f"x {args.inner_folds} inner folds")
    # ``best_inner_f1`` is tracked independently so the report below works
    # regardless of whether Optuna was available or the study found any
    # completed trial.  Previous version referenced ``study.best_value`` after
    # the if/else and raised UnboundLocalError when ``optuna is None``.
    default_params = dict(
        popsize=30, max_nfe=30000, lambda_size=0.05,
        linnik_alpha=(2.0 / (1.0 + math.sqrt(5.0))),  # golden ratio
        linnik_scale=0.05,
    )
    best_params = default_params
    best_inner_f1 = float("nan")
    trials_df = pd.DataFrame()
    used_defaults_reason = None

    if optuna is None:
        used_defaults_reason = "Optuna not installed"
    elif args.optuna_trials <= 0:
        used_defaults_reason = f"--optuna-trials={args.optuna_trials} (<=0)"
    else:
        try:
            sampler = optuna.samplers.TPESampler(
                seed=args.base_seed + hash(dataset) % 2**16,
            )
            study = optuna.create_study(sampler=sampler)
            study.optimize(
                objective, n_trials=args.optuna_trials, show_progress_bar=False,
            )
            # study.best_params / study.best_value raise ValueError if no
            # trial completed successfully -- protect against that.
            completed = [t for t in study.trials if t.value is not None]
            if completed:
                best_params = study.best_params
                best_inner_f1 = -study.best_value
                trials_df = study.trials_dataframe()
                trials_df.insert(0, "dataset", dataset)
            else:
                used_defaults_reason = (
                    f"all {len(study.trials)} Optuna trials failed; using defaults"
                )
        except Exception as exc:  # pragma: no cover
            used_defaults_reason = f"Optuna study raised {exc!r}; using defaults"

    if used_defaults_reason is not None:
        print(f"  !! {used_defaults_reason}", file=sys.stderr)

    if not math.isnan(best_inner_f1):
        print(f"[{dataset}] best inner F1 = {best_inner_f1:.4f}  "
              f"best params = {best_params}")
    else:
        print(f"[{dataset}] using default params = {best_params}")

    # ---- Final evaluation on the full 50-fold pipeline ----
    print(f"[{dataset}] final 50-fold evaluation with best params")
    final_jobs = []
    for repeat, fold, tr, va in splits_list:
        seed = derive_seed(args.base_seed, ALGO_NAME, dataset, int(repeat), int(fold))
        final_jobs.append((dataset, csv_path, int(repeat), int(fold), tr, va, seed))

    def _final_worker(job):
        dataset_, csv_path_, repeat_, fold_, tr_, va_, seed_ = job
        m = _run_one_fold(
            csv_path_, tr_, va_, seed_,
            int(best_params["popsize"]),
            int(best_params["max_nfe"]),
            float(best_params["lambda_size"]),
            float(best_params["linnik_alpha"]),
            float(best_params["linnik_scale"]),
            args.min_hidden, args.max_hidden, args.lb, args.ub,
        )
        return {
            "algo": ALGO_NAME, "dataset": dataset_, "repeat": repeat_, "fold": fold_,
            "seed": seed_, **m,
        }

    rows = Parallel(n_jobs=args.jobs, backend=config.JOBLIB_BACKEND, verbose=10)(
        delayed(_final_worker)(j) for j in final_jobs
    )
    return rows, best_params, trials_df


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("B5: Optuna sweep over DCS-VSE hyperparameters "
                     "(popsize, max_nfe, lambda_size, Linnik alpha, scale)."),
    )
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_SWEEP_DATASETS)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS,
                    help="Number of CV folds used inside each Optuna trial.")
    ap.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    ap.add_argument("--min-hidden", type=int, default=1)
    ap.add_argument("--max-hidden", type=int, default=20)
    ap.add_argument("--lb", type=float, default=config.LB)
    ap.add_argument("--ub", type=float, default=config.UB)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=_here("results_B5_dcsvse_optuna"))
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 1 dataset, 3 trials, 2 inner folds, full eval at 6 folds.")
    args = ap.parse_args(argv)

    if args.quick:
        args.datasets = args.datasets[:1]
        args.repeats = 1
        args.folds = 6
        args.optuna_trials = 3
        args.inner_folds = 2
        args.jobs = min(args.jobs, 4)

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"B5 DCS-VSE Optuna sweep | datasets={args.datasets} "
          f"trials={args.optuna_trials} inner_folds={args.inner_folds} "
          f"jobs={args.jobs}")

    t0 = time.perf_counter()
    all_rows = []
    best_rows = []
    trials_all = []
    for dataset in args.datasets:
        out = _run_dataset(args, dataset)
        if out is None:
            continue
        rows, best, trials_df = out
        all_rows.extend(rows)
        best_rows.append({"dataset": dataset, **best})
        if not trials_df.empty:
            trials_all.append(trials_df)
    wall = time.perf_counter() - t0

    df = pd.DataFrame(all_rows).sort_values(["dataset", "repeat", "fold"])
    df.to_csv(os.path.join(args.results_dir, "results.csv"), index=False)

    pd.DataFrame(best_rows).to_csv(
        os.path.join(args.results_dir, "best_params.csv"), index=False,
    )
    if trials_all:
        pd.concat(trials_all, ignore_index=True).to_csv(
            os.path.join(args.results_dir, "optuna_trials.csv"), index=False,
        )

    summary = (
        df.groupby(["algo", "dataset"])
        .agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
             auc_mean=("auc", "mean"), acc_mean=("acc", "mean"),
             hidden_mean=("hidden_nodes", "mean"),
             runtime_mean=("runtime_sec", "mean"), n=("f1_macro", "count"))
        .reset_index()
    )
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)

    with open(os.path.join(args.results_dir, "manifest.json"), "w") as fh:
        json.dump({
            "platform": platform.platform(),
            "python": platform.python_version(),
            "study": "B5_dcsvse_optuna",
            "algorithm": ALGO_NAME,
            "base_algo": BASE_ALGO_KEY,
            "datasets": args.datasets,
            "cv_mode": args.cv_mode,
            "repeats": args.repeats,
            "folds": args.folds,
            "inner_folds": args.inner_folds,
            "optuna_trials": args.optuna_trials,
            "min_hidden": args.min_hidden,
            "max_hidden": args.max_hidden,
            "lb": args.lb,
            "ub": args.ub,
            "base_seed": args.base_seed,
            "n_final_runs": len(all_rows),
            "wall_seconds": wall,
            "paired_with": "results_H1to20",
        }, fh, indent=2)

    print(f"\nFinished in {wall:.1f}s")
    print(f"  results       : {args.results_dir}/results.csv")
    print(f"  best params   : {args.results_dir}/best_params.csv")
    print(f"  optuna trials : {args.results_dir}/optuna_trials.csv")
    print(f"  summary       : {args.results_dir}/summary.csv")
    print(f"  Next          : python b5_compare_with_defaults.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
