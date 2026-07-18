#!/usr/bin/env python3
"""B5b: Optuna sweep over the proposed DCS--VSE (no structural operators).

This is the v17-reframed counterpart to ``b5_dcsvse_optuna.py``.  Under the
v17 framing the proposed method is the simpler variant that handles length
transitions by random reinitialisation of worst-ranked individuals; the
structural-operator (SO) machinery is reported only as an ablation
(``DCS_VSE_DKA_opt_0_d``).  Study~F's fair-budget Optuna closure must
therefore run against the proposed method, not against the +SO ablation.

What is identical to ``b5_dcsvse_optuna.py``
-------------------------------------------
* Search space: popsize, max_nfe, lambda_size, Linnik alpha, Linnik scale.
* Search ranges, TPE sampler, inner-CV objective, paired-fold final eval.
* Worker-local monkey-patches for lambda_size (Evaluator._penalty) and Linnik
  (vse.algorithms.dcs.lnf2).
* Output schema (results.csv / best_params.csv / optuna_trials.csv /
  summary.csv / manifest.json) so the existing analysis tooling carries over.

What differs
------------
* ``BASE_ALGO_KEY = "DCS_noVSE_d"``   (proposed method under v17 framing)
* ``ALGO_NAME = "DCS_VSE_NoSO_OptunaTuned"``  (independent seed namespace)
* Output directory: ``results_B5b_dcsvse_noso_optuna``
* Paired-with key in the manifest: ``results_H1to20`` row where
  ``algo == "DCS_noVSE_d"`` (NOT ``DCS_VSE_DKA_opt_0_d``).
* Defaults updated to 40 trials and all nine SELECTED_DATASETS, matching
  the v17 Study~F protocol.

Companion analysis script
-------------------------
After this finishes, run ``b5b_compare_with_defaults.py`` for the paired
Wilcoxon test against the ``DCS_noVSE_d`` default-hyperparameter run.

Usage
-----
    python b5b_dcsvse_noso_optuna.py                   # full 9-dataset, 40-trial run
    python b5b_dcsvse_noso_optuna.py --quick           # smoke test
    python b5b_dcsvse_noso_optuna.py --optuna-trials 25
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

ALGO_NAME = "DCS_VSE_NoSO_OptunaTuned"
BASE_ALGO_KEY = "DCS_noVSE_d"
DEFAULT_OPTUNA_TRIALS = 40
DEFAULT_INNER_FOLDS = 5
DEFAULT_SWEEP_DATASETS = list(config.SELECTED_DATASETS)


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
    """Patch Evaluator._penalty to scale the size term by ``lambda_size``."""
    from vse import evaluator as ev

    if not hasattr(ev.Evaluator, "_orig_penalty_for_b5b"):
        ev.Evaluator._orig_penalty_for_b5b = ev.Evaluator._penalty

    def patched_penalty(self, h):
        if self._denom <= 0.0:
            return 0.0
        return lambda_size * 100.0 * ((h - self.min_hidden) / self._denom)

    ev.Evaluator._penalty = patched_penalty


def _install_linnik_patch(alpha: float, scale: float):
    """Patch ``vse.algorithms.dcs.lnf2`` to ignore call-site (alpha, scale).

    ``DCS_noVSE_d`` is registered in the same ``vse.algorithms.dcs`` module as
    the +SO variant, so the existing patch target carries over verbatim.  We
    additionally patch ``vse.rng.lnf2`` and any other algorithm module that
    happens to have re-imported ``lnf2`` at module load time -- defensive,
    cheap, and survives future reorganisations of the algorithm package.
    """
    from vse import rng as rng_mod

    if not hasattr(rng_mod, "_orig_lnf2_for_b5b"):
        rng_mod._orig_lnf2_for_b5b = rng_mod.lnf2
    orig = rng_mod._orig_lnf2_for_b5b

    def patched_lnf2(rng, _alpha_ignored, _scale_ignored, m=1, n=1):
        return orig(rng, alpha, scale, m, n)

    rng_mod.lnf2 = patched_lnf2

    import sys as _sys
    for name, mod in list(_sys.modules.items()):
        if name and name.startswith("vse.algorithms.") and hasattr(mod, "lnf2"):
            if not hasattr(mod, "_orig_lnf2_for_b5b"):
                mod._orig_lnf2_for_b5b = mod.lnf2
            mod.lnf2 = patched_lnf2


# --------------------------------------------------------------------------- #
# Run a single fold of the proposed method under a hyperparameter config      #
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

    # Resolve the algorithm first so its module is in sys.modules before we
    # walk through and patch every lnf2 binding.
    algo_fn = get_algorithm(BASE_ALGO_KEY)

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

    print(f"\n[{dataset}] Optuna sweep on {BASE_ALGO_KEY}: "
          f"{args.optuna_trials} trials x {args.inner_folds} inner folds")

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
        description=("B5b: Optuna sweep over DCS-VSE (no-SO) hyperparameters "
                     "(popsize, max_nfe, lambda_size, Linnik alpha, scale). "
                     "v17 fair-budget closure on the proposed method."),
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
    ap.add_argument("--results-dir", default=_here("results_B5b_dcsvse_noso_optuna"))
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 1 dataset, 3 trials, 2 inner folds, 6 final folds.")
    args = ap.parse_args(argv)

    if args.quick:
        args.datasets = args.datasets[:1]
        args.repeats = 1
        args.folds = 6
        args.optuna_trials = 3
        args.inner_folds = 2
        args.jobs = min(args.jobs, 4)

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"B5b DCS-VSE (no-SO) Optuna sweep | base_algo={BASE_ALGO_KEY} "
          f"datasets={args.datasets} trials={args.optuna_trials} "
          f"inner_folds={args.inner_folds} jobs={args.jobs}")

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
            "study": "B5b_dcsvse_noso_optuna",
            "algorithm": ALGO_NAME,
            "base_algo": BASE_ALGO_KEY,
            "framing": ("v17: proposed method is DCS-VSE (no structural operators); "
                        "structural-operator machinery moved to ablation only."),
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
            "paired_with": "results_H1to20 row where algo == DCS_noVSE_d",
        }, fh, indent=2)

    print(f"\nFinished in {wall:.1f}s")
    print(f"  results       : {args.results_dir}/results.csv")
    print(f"  best params   : {args.results_dir}/best_params.csv")
    print(f"  optuna trials : {args.results_dir}/optuna_trials.csv")
    print(f"  summary       : {args.results_dir}/summary.csv")
    print(f"  Next          : python b5b_compare_with_defaults.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
