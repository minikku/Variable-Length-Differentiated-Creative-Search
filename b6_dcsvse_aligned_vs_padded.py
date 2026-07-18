#!/usr/bin/env python3
"""B6: Aligned-vs-padded counterfactual on the proposed DCS-VSE method.

Closes the open methodological question identified in pre-submission review:
the original Study B (B2) ran the aligned-vs-padded counterfactual on CoDE,
COLSHADE, and DBA -- transfer-target algorithms -- but never on the proposed
method itself.  B6 plugs that gap by running ``DCS_noVSE_d`` (the v22 proposed
method, formerly DCS-noVSE) under TWO encodings on the same 5x10 stratified
CV pipeline.  Each row of the output differs from its pair only in the
cross-length interaction rule.

Two modes
---------
  aligned (the rule used throughout the paper)
    Every coordinate-wise arithmetic interaction in the DCS update is
    restricted to the shared prefix of the encoding,
        D_eff = min(D(L_i), {D(L_j) : j in R_i})
    Coordinates beyond D_eff are left unchanged.

  padded (the counterfactual)
    Every individual is encoded as a fixed-length D_max vector,
        D_max = D(H_max) = H_max * (inp + 1) + outp * (H_max + 1)
    Coordinates beyond the individual's true D(H_i) are filled with fresh
    samples from U[lb, ub] before each operator call and discarded on decode.
    The DCS arithmetic update now operates on all D_max coordinates because
    every individual reports the same dimensionality.

How it is implemented
---------------------
This script does NOT edit ``vse/algorithms/dcs.py``.  Instead, the two
network-side encoding functions that every DCS step uses are monkey-patched
inside the worker:

    vse.network.network_to_vector  ->  network_to_padded_vector
    vse.network.vector_to_network  ->  padded_vector_to_network

The padded variants encode every SLNN as a D_max vector with fresh
U[lb, ub] padding in the tail beyond the individual's true D(H_i).  Because
every individual now lives at D_max, the prefix restriction
``D_eff = min(D(L_i), D(L_j))`` collapses to D_max for every pair, which is
exactly the max-embedding-padding semantics the counterfactual demands.

The swap is local to the worker process (joblib spawns a fresh interpreter
per task), so an aligned and a padded run can execute concurrently without
contamination.

Usage
-----
    python b6_dcsvse_aligned_vs_padded.py
    python b6_dcsvse_aligned_vs_padded.py --modes padded     # only padded rows
    python b6_dcsvse_aligned_vs_padded.py --quick

Output
------
    results_B6_dcsvse_aligned_vs_padded/results.csv
    results_B6_dcsvse_aligned_vs_padded/summary.csv
    results_B6_dcsvse_aligned_vs_padded/manifest.json

Rows are labelled algo = "DCS_VSE_aligned" and "DCS_VSE_padded".  After this
finishes, run ``b6_compare_aligned_vs_padded.py`` to obtain the paired
Wilcoxon test, the per-dataset delta table, and the LaTeX block for the
revised Study B section.

Cost
----
1 method x 2 modes x 8 datasets x 50 folds = 800 DCS--VSE evaluations.
~24 hours wall-clock on 16-core hardware; quick mode for sanity-checking
finishes in ~5 minutes.
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

# Single algorithm, two modes.  ``DCS_noVSE_d`` is the v22 proposed method
# (cross-length alignment + random reinitialisation for length transitions).
B6_ALGO = "DCS_noVSE_d"
B6_MODES = ["aligned", "padded"]


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def derive_seed(base_seed, algo, mode, dataset, repeat, fold) -> int:
    s = f"{base_seed}|{algo}|{mode}|{dataset}|{repeat}|{fold}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


def _d_of(h: int, inp: int, outp: int) -> int:
    """Encoding length D(H) of an (inp, h, outp) SLNN."""
    return h * (inp + 1) + outp * (h + 1)


# --------------------------------------------------------------------------- #
# Padded encoding installed by monkey-patch inside each padded worker         #
# --------------------------------------------------------------------------- #
def _install_padded_encoding(max_hidden: int, lb: float, ub: float,
                             base_rng: np.random.Generator):
    """Replace network_to_vector / vector_to_network with padded variants.

    Must be called from inside the worker, after imports.  DCS reads encoded
    vectors through ``from ..network import network_to_vector / vector_to_network``,
    so we patch the *module* attribute that the algorithm looks up at call time.

    The padding RNG is seeded deterministically from ``base_rng`` so the
    padded values are reproducible across machines for a given fold.
    """
    from vse import network as netmod

    orig_n2v = netmod.network_to_vector
    orig_v2n = netmod.vector_to_network

    pad_rng = np.random.default_rng(base_rng.integers(0, 2 ** 32 - 1))

    def network_to_padded_vector(net):
        vec, meta = orig_n2v(net)
        inp, _hidn_orig, outp = meta
        d_max = _d_of(max_hidden, inp, outp)
        if vec.size >= d_max:
            return vec[:d_max], (inp, max_hidden, outp)
        # Fresh per-call padding from U[lb,ub] for the missing tail.
        pad = lb + (ub - lb) * pad_rng.random(d_max - vec.size)
        return np.concatenate([vec, pad]), (inp, max_hidden, outp)

    def padded_vector_to_network(vec, meta):
        # Every individual now reports max_hidden; the DCS arithmetic update
        # therefore operates on D(max_hidden) coordinates with no prefix
        # restriction (Deff = D_max for every pair).  Decoding always uses
        # max_hidden, so the rebuilt SLNN has H = max_hidden coordinates --
        # exactly the max-embedding-padded semantics demanded by the
        # counterfactual.
        inp, hidn, outp = meta
        return orig_v2n(vec, (inp, hidn, outp))

    netmod.network_to_vector = network_to_padded_vector
    netmod.vector_to_network = padded_vector_to_network


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #
def run_single(task: dict) -> dict:
    from vse.algorithms import get_algorithm  # import inside the worker

    ds = load_dataset(task["csv_path"])
    tr, va = task["train_idx"], task["val_idx"]
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

    if task["mode"] == "padded":
        # Worker-local monkey-patch of the encode/decode functions.  The DCS
        # algorithm itself is unmodified; the swap forces every individual to
        # live at the same D_max dimensionality, so the prefix-restriction
        # rule Deff = min(D_i, D_j) collapses to D_max and the arithmetic
        # operator runs on every coordinate by construction.
        _install_padded_encoding(
            task["max_hidden"], task["lb"], task["ub"], rng,
        )

    algo_fn = get_algorithm(task["algo"])
    t0 = time.perf_counter()
    result = algo_fn(options, rng)
    runtime = time.perf_counter() - t0
    metrics = evaluate_on_fold(result.best_x, x_val, t_val)

    row = {
        "algo": f"{task['algo']}:{task['mode']}",
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
    return {"row": row}


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
        for mode in args.modes:
            for repeat, fold, tr, va in splits:
                tasks.append({
                    "algo": B6_ALGO,
                    "mode": mode,
                    "dataset": dataset,
                    "csv_path": csv_path,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "run_id": run_id,
                    "train_idx": tr,
                    "val_idx": va,
                    "seed": derive_seed(
                        args.base_seed, B6_ALGO, mode,
                        dataset, repeat, fold,
                    ),
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("B6: Aligned-vs-padded counterfactual on the proposed "
                     "DCS-VSE method itself (closes the Study B gap)."),
    )
    ap.add_argument("--modes", nargs="*", default=B6_MODES, choices=B6_MODES)
    ap.add_argument("--datasets", nargs="*",
                    default=list(config.SELECTED_DATASETS))
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--popsize", type=int, default=config.POPSIZE)
    ap.add_argument("--max-nfe", type=int, default=config.MAX_NFE)
    ap.add_argument("--min-hidden", type=int, default=1)
    ap.add_argument("--max-hidden", type=int, default=20)
    ap.add_argument("--lb", type=float, default=config.LB)
    ap.add_argument("--ub", type=float, default=config.UB)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir",
                    default=_here("results_B6_dcsvse_aligned_vs_padded"))
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: 1 dataset, 1 repeat, 3 folds.")
    args = ap.parse_args(argv)

    if args.quick:
        args.datasets = args.datasets[:1]
        args.repeats = 1
        args.folds = 3
        args.jobs = min(args.jobs, 4)

    os.makedirs(args.results_dir, exist_ok=True)
    tasks = build_tasks(args)
    if not tasks:
        print("No runs to execute.", file=sys.stderr)
        return 1

    print(f"B6 DCS-VSE aligned vs padded | algo={B6_ALGO} "
          f"modes={args.modes} datasets={len(args.datasets)} "
          f"runs={len(tasks)} jobs={args.jobs}")

    t0 = time.perf_counter()
    outputs = Parallel(
        n_jobs=args.jobs, backend=config.JOBLIB_BACKEND, verbose=10,
    )(delayed(run_single)(t) for t in tasks)
    wall = time.perf_counter() - t0

    df = pd.DataFrame([o["row"] for o in outputs]).sort_values(
        ["algo", "dataset", "repeat", "fold"]
    )
    df.to_csv(os.path.join(args.results_dir, "results.csv"), index=False)

    summary = (
        df.groupby(["algo", "dataset"])
        .agg(f1_mean=("f1_macro", "mean"),
             f1_std=("f1_macro", "std"),
             auc_mean=("auc", "mean"),
             acc_mean=("acc", "mean"),
             hidden_mean=("hidden_nodes", "mean"),
             runtime_mean=("runtime_sec", "mean"),
             n=("run_id", "count"))
        .reset_index()
    )
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)

    manifest = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "study": "B6_dcsvse_aligned_vs_padded",
        "algorithm": B6_ALGO,
        "modes": args.modes,
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
        "purpose": ("Closes pre-submission review Major Issue #1: run the "
                    "aligned-vs-padded counterfactual on the proposed method "
                    "itself, not only on CoDE/COLSHADE/DBA (B2)."),
    }
    with open(os.path.join(args.results_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nFinished {len(tasks)} runs in {wall:.1f}s")
    print(f"  results : {args.results_dir}/results.csv")
    print(f"  summary : {args.results_dir}/summary.csv")
    print(f"  Next    : python b6_compare_aligned_vs_padded.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
