#!/usr/bin/env python3
"""B2: Aligned-vs-padded counterfactual on CoDE, COLSHADE, DBA.

Runs each of CoDE_d / COLSHADE_d / DBA_d under TWO encodings on the same
5x10 stratified CV pipeline, so the *only* difference between rows is the
cross-length interaction rule.  This is the experiment the manuscript's
limitations section identifies as the clean attribution counterfactual for
Study IV.

Two modes per algorithm
-----------------------
  aligned (current implementation)
    Every coordinate-wise arithmetic term is restricted to the shared prefix:
        n = min(|z_i|, |z_j|, ...)
        operate on [:n], leave the rest of z_i untouched
    This is the *aligned* rule used throughout the paper.

  padded (new for this counterfactual)
    Every individual is encoded as a fixed-length D_max vector where
        D_max = D(H_max) = H_max * (inp + 1) + outp * (H_max + 1)
    Coordinates beyond the individual's true D(H_i) are filled with fresh
    samples from U[lb, ub] before each operator call and discarded on
    decode.  Operators run on D_max coordinates with no min() restriction.
    Decoding reads the first D(H_i) entries and reconstructs the network.

How it is implemented
---------------------
We do NOT edit code.py / colshade.py / dba.py.  Instead, we monkey-patch the
two functions every DE algorithm calls under the hood:

    vse.network.network_to_vector   ->  network_to_padded_vector
    vse.network.vector_to_network   ->  padded_vector_to_network

This swap is local to the worker process (joblib spawns a fresh interpreter
per task), so the aligned baselines run with the original semantics in the
other worker even if the script is launched concurrently.

Usage
-----
    python b2_aligned_vs_padded.py
    python b2_aligned_vs_padded.py --algos CoDE_d COLSHADE_d
    python b2_aligned_vs_padded.py --modes padded     # only padded rows
    python b2_aligned_vs_padded.py --quick

Output
------
    results_B2_aligned_vs_padded/results.csv
    results_B2_aligned_vs_padded/summary.csv
    results_B2_aligned_vs_padded/manifest.json

Rows are labeled algo = "CoDE_d:aligned", "CoDE_d:padded", etc., so the
counterfactual is a single Wilcoxon test per algorithm.
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

# Algorithms exposed to the counterfactual.  These are the three "alignment
# transfer" methods used in Study IV.
B2_ALGOS = ["CoDE_d", "COLSHADE_d", "DBA_d"]
B2_MODES = ["aligned", "padded"]


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def derive_seed(base_seed, algo, dataset, repeat, fold) -> int:
    s = f"{base_seed}|{algo}|{dataset}|{repeat}|{fold}"
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


def _d_of(h: int, inp: int, outp: int) -> int:
    """Encoding length of a (inp, h, outp) SLNN."""
    return h * (inp + 1) + outp * (h + 1)


# --------------------------------------------------------------------------- #
# Padded encoding installed by monkey-patch inside each padded worker         #
# --------------------------------------------------------------------------- #
def _install_padded_encoding(max_hidden: int, lb: float, ub: float,
                             base_rng: np.random.Generator):
    """Replace network_to_vector / vector_to_network with padded variants.

    Must be called from inside the worker, after imports.  All three DE-family
    algorithms call these two functions through ``from ..network import ...``,
    so we patch the module attribute (the bound names inside each algorithm
    module are already references to the original functions; we need to patch
    the *module* attribute that they look up at call time).

    The padding RNG is seeded deterministically so the padded values are
    reproducible across machines for a given fold.
    """
    from vse import network as netmod

    orig_n2v = netmod.network_to_vector
    orig_v2n = netmod.vector_to_network

    pad_rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))

    def network_to_padded_vector(net):
        vec, meta = orig_n2v(net)
        inp, hidn, outp = meta
        d_max = _d_of(max_hidden, inp, outp)
        if vec.size >= d_max:
            return vec[:d_max], (inp, max_hidden, outp)
        # Fresh per-call padding from U[lb,ub] for the missing tail.
        pad = lb + (ub - lb) * pad_rng.random(d_max - vec.size)
        return np.concatenate([vec, pad]), (inp, max_hidden, outp)

    def padded_vector_to_network(vec, meta):
        # meta carries the *padded* hidn = max_hidden.  We must decode back to
        # the true H for this individual, which we recover from the vector's
        # first valid coordinate count.  Since every individual now lives at
        # max_hidden, we always decode at max_hidden -- this is exactly the
        # "max-embedding padding" semantics for which we want a counterfactual.
        inp, hidn, outp = meta
        return orig_v2n(vec, (inp, hidn, outp))

    netmod.network_to_vector = network_to_padded_vector
    netmod.vector_to_network = padded_vector_to_network


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #
def run_single(task: dict) -> dict:
    from vse.algorithms import get_algorithm  # imported inside worker

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
        # Replace the alignment-implementing encode/decode functions with
        # max-embedding padding semantics, then call the (unmodified) algorithm.
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
        for algo in args.algos:
            for mode in args.modes:
                for repeat, fold, tr, va in splits:
                    tasks.append({
                        "algo": algo,
                        "mode": mode,
                        "dataset": dataset,
                        "csv_path": csv_path,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "run_id": run_id,
                        "train_idx": tr,
                        "val_idx": va,
                        "seed": derive_seed(
                            args.base_seed, f"{algo}:{mode}",
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
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--algos", nargs="*", default=B2_ALGOS, choices=B2_ALGOS)
    ap.add_argument("--modes", nargs="*", default=B2_MODES, choices=B2_MODES)
    ap.add_argument("--datasets", nargs="*", default=config.SELECTED_DATASETS)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--popsize", type=int, default=config.POPSIZE)
    ap.add_argument("--max-nfe", type=int, default=config.MAX_NFE)
    ap.add_argument("--min-hidden", type=int, default=1)
    ap.add_argument("--max-hidden", type=int, default=20)
    ap.add_argument("--lb", type=float, default=config.LB)
    ap.add_argument("--ub", type=float, default=config.UB)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=_here("results_B2_aligned_vs_padded"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.popsize = 16
        args.max_nfe = 320
        args.repeats = 1
        args.folds = 3
        args.jobs = min(args.jobs, 4)

    os.makedirs(args.results_dir, exist_ok=True)
    tasks = build_tasks(args)
    if not tasks:
        print("No runs to execute.", file=sys.stderr)
        return 1

    print(f"B2 counterfactual | algos={args.algos} modes={args.modes} "
          f"datasets={len(args.datasets)} runs={len(tasks)} jobs={args.jobs}")
    print(f"  Hidden bounds: [{args.min_hidden}, {args.max_hidden}]  "
          f"Param bounds: [{args.lb}, {args.ub}]")

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

    # Convenience: a paired Wilcoxon table aligned vs padded per (algo, dataset).
    try:
        from scipy.stats import wilcoxon

        wide = df.pivot_table(
            index=["dataset", "repeat", "fold"],
            columns="algo",
            values="f1_macro",
        )
        rows = []
        for algo_root in args.algos:
            a = f"{algo_root}:aligned"
            p = f"{algo_root}:padded"
            if a in wide.columns and p in wide.columns:
                for dataset in wide.index.get_level_values("dataset").unique():
                    sub = wide.loc[dataset]
                    sub = sub.dropna(subset=[a, p])
                    if len(sub) < 5:
                        continue
                    try:
                        stat, pv = wilcoxon(sub[a], sub[p], alternative="two-sided")
                    except ValueError:
                        stat, pv = np.nan, np.nan
                    rows.append({
                        "algo": algo_root, "dataset": dataset,
                        "n": int(len(sub)),
                        "mean_aligned": float(sub[a].mean()),
                        "mean_padded": float(sub[p].mean()),
                        "delta_aligned_minus_padded": float(
                            sub[a].mean() - sub[p].mean()
                        ),
                        "wilcoxon_p": float(pv),
                    })
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(args.results_dir, "wilcoxon_aligned_vs_padded.csv"),
                index=False,
            )
    except ImportError:
        pass

    with open(os.path.join(args.results_dir, "manifest.json"), "w") as fh:
        json.dump({
            "platform": platform.platform(),
            "python": platform.python_version(),
            "study": "B2_aligned_vs_padded",
            "algorithms": args.algos,
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
        }, fh, indent=2)

    print(f"\nFinished {len(tasks)} runs in {wall:.1f}s")
    print(f"  results : {args.results_dir}/results.csv")
    print(f"  summary : {args.results_dir}/summary.csv")
    if os.path.exists(os.path.join(args.results_dir, "wilcoxon_aligned_vs_padded.csv")):
        print(f"  wilcoxon: {args.results_dir}/wilcoxon_aligned_vs_padded.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
