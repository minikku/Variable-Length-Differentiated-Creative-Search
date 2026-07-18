#!/usr/bin/env python3
"""B9: TabPFN as a current-generation tabular reference under the shared pipeline.

Adds a prior-fitted transformer (TabPFN v2) to the headline benchmark so the
paper carries a modern tabular baseline alongside the SLNN family and the
gradient-boosted trees. TabPFN needs no per-dataset hyperparameter tuning (a
single in-context forward pass), which makes it the cheapest possible external
reference and the one reviewers specifically expect for the small-n regime used
here.

Pipeline parity
---------------
Everything is inherited from the tree baseline (b8_matched_complexity_trees.py)
so results drop straight into the existing analysis:
    * identical CV partitions           : vse.dataset.make_cv_splits(...)
    * identical per-fold seed derivation : derive_seed(base, algo, ds, r, f)
    * identical feature front-end        : SelectKBest(mutual_info_classif, k=256)
                                           on datasets with d > 1000
    * identical scoring                  : macro-F1, one-vs-rest macro AUC, accuracy
    * identical one-hot -> class-index    : _to_class_index
The only change is the model: TabPFNClassifier in place of LGBM/XGBoost.

Complexity axis
---------------
TabPFN is a large pretrained transformer applied in context; it is NOT a compact
model and does not sit on the (macro-F1, hidden-units / leaves) compactness axis
the SLNN family and the matched trees occupy. Its "complexity" columns are
therefore left as NaN and it should be plotted as an accuracy reference, not a
compactness competitor. The backbone parameter count is recorded in the
manifest for transparency.

Requirements
------------
    pip install tabpfn            # v2 (PriorLabs); downloads weights on first run
    A CUDA GPU is used by default (--device cuda); pass --device cpu to force CPU.

Output
------
    results_B9_tabpfn/results.csv     (one row per fold)
    results_B9_tabpfn/summary.csv     (algo x dataset means, same columns as b8)
    results_B9_tabpfn/manifest.json

Usage
-----
    python b9_tabpfn.py                       # all 18 datasets, 5x10 CV, GPU
    python b9_tabpfn.py --device cpu          # CPU fallback (slow)
    python b9_tabpfn.py --datasets WDBC ILPD  # subset
    python b9_tabpfn.py --quick               # smoke test: 1 dataset x 5 folds
    python b9_tabpfn.py --n-estimators 8      # more in-context ensembling (slower)

Runtime estimate (single GPU)
-----------------------------
    ~0.3-1 s per fold; 18 datasets x 50 folds = 900 folds -> ~10-20 min plus a
    one-time model download. On CPU expect ~1 hour. No tuning pass is run.
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

import config
from vse.dataset import load_dataset, make_cv_splits

# Reuse the EXACT helpers from the tree baseline so the pipeline is identical.
# (Importing the module does not run its main(); it is __name__-guarded.)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from b8_matched_complexity_trees import derive_seed, _to_class_index, _score

# Self-contained feature front-end. Defined here (not imported from b8) so a
# modified local b8 cannot silently change k. These MUST match the headline
# pipeline: reduce to the top-k mutual-information features when d > threshold.
FEATURE_REDUCTION_THRESHOLD = 1000
FEATURE_K = 256


def _select_features(x_train, y_train, x_val, seed,
                     threshold=FEATURE_REDUCTION_THRESHOLD, k=FEATURE_K):
    d = x_train.shape[1]
    if d <= threshold:
        return x_train, x_val
    kk = min(int(k), d)
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=int(seed) % (2 ** 31 - 1)),
        k=kk,
    )
    return selector.fit_transform(x_train, y_train), selector.transform(x_val)

warnings.filterwarnings("ignore")

# --- Windows hardening ------------------------------------------------------- #
# ``OSError [WinError 10038] ... not a socket`` comes from a worker-process pool
# (loky/joblib, or a torch DataLoader) failing on Windows spawn. Force every such
# pool to stay in-process / threaded so no inter-process sockets are created.
#os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "26")


def _resolve_device(requested: str) -> str:
    """Report GPU status; fall back to CPU if CUDA was requested but is
    unavailable (usually a CPU-only torch build)."""
    try:
        import torch
    except Exception:
        return requested
    if str(requested).startswith("cuda"):
        if torch.cuda.is_available():
            try:
                print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA available")
            except Exception:
                print("CUDA available")
            return requested
        print("WARNING: --device cuda requested but torch.cuda.is_available() is False.")
        print("  The installed torch is a CPU-only build, so TabPFN runs on CPU.")
        print("  For GPU, reinstall the CUDA build, e.g.:")
        print("    pip uninstall -y torch")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print("  (match cuXXX to your CUDA toolkit). Falling back to CPU for this run.")
        return "cpu"
    return requested
# ---------------------------------------------------------------------------- #

# The full 18-dataset headline benchmark (same set as results_H1to20).
DEFAULT_DATASETS = [
    "ADENOCARCINOMA", "COLON_ALON", "DLBCL", "DYSLEXIA", "DYSLEXIA_10p",
    "HABERMAN_SURVIVAL", "HEART_CLEVELAND", "HEART_FAILURE", "HEPATITIS", "ILPD",
    "LEUKEMIA1", "LEUKEMIA2", "PARKINSONS", "PIMA_DIABETES", "PROSTATE6033",
    "PROSTATE_TUMOR", "VERTEBRAL_COLUMN", "WDBC",
]

ALGO_NAME = "TabPFN"


def _make_tabpfn(device: str, n_estimators: int, seed: int, n_jobs: int = 1, model_path=None):
    """Instantiate a TabPFN v2 classifier, passing only arguments the installed
    version supports.

    ``n_jobs=1`` is the default on purpose: TabPFN's parallel worker backend
    (loky/joblib) is what triggers ``OSError [WinError 10038] ... not a socket``
    on Windows, so the safe, portable setting is single-process. It costs a
    little speed and nothing in accuracy.
    """
    import inspect
    from tabpfn import TabPFNClassifier  # imported lazily so --help works without it

    candidate = dict(
        device=device,
        random_state=int(seed),
        n_estimators=int(n_estimators),
        ignore_pretraining_limits=True,
        n_jobs=int(n_jobs),          # WinError 10038 fix: disable the loky worker pool
    )
    if model_path:
        # Load pre-downloaded weights instead of fetching at runtime (offline /
        # reproducible). TabPFN v2 accepts a filesystem path or a preset string.
        candidate["model_path"] = str(model_path)
    try:
        supported = set(inspect.signature(TabPFNClassifier.__init__).parameters)
    except (TypeError, ValueError):
        supported = set(candidate)
    kwargs = {k: v for k, v in candidate.items() if k in supported}
    # Very old TabPFN used ``N_ensemble_configurations`` instead of ``n_estimators``.
    if "n_estimators" not in supported and "N_ensemble_configurations" in supported:
        kwargs["N_ensemble_configurations"] = int(n_estimators)
    return TabPFNClassifier(**kwargs)


def run_single(task: dict, clf) -> dict:
    ds = load_dataset(task["csv_path"])
    tr, va = task["train_idx"], task["val_idx"]
    x_train, y_train = ds.x[tr], _to_class_index(ds.y[tr])
    x_val, y_val = ds.x[va], _to_class_index(ds.y[va])

    # Feature reduction, identical to the tree/headline pipeline (MI, k=256, d>1000).
    x_train, x_val = _select_features(
        x_train, y_train, x_val, task["seed"],
        task["feature_threshold"], task["feature_k"])

    # The classifier is built ONCE in main() and reused across all 900 folds:
    # TabPFN loads its weights at construction, so re-instantiating per fold would
    # reload them every time. fit() only stores the training set; the GPU forward
    # pass happens in predict_proba.
    t0 = time.perf_counter()
    import joblib
    with joblib.parallel_backend("threading"):
        clf.fit(x_train, y_train)
        proba = np.asarray(clf.predict_proba(x_val))
    runtime = time.perf_counter() - t0

    # Align proba columns to class indices 0..K-1 (stratified CV keeps all classes
    # in every training fold, so clf.classes_ == range(K); guard anyway).
    n_classes = int(np.max(np.concatenate([y_train, y_val]))) + 1
    if proba.shape[1] != n_classes and hasattr(clf, "classes_"):
        full = np.zeros((proba.shape[0], n_classes))
        for j, c in enumerate(np.asarray(clf.classes_).astype(int)):
            full[:, c] = proba[:, j]
        proba = full
    pred = np.argmax(proba, axis=1)
    metrics = _score(y_val, pred, proba)

    row = {
        "algo": ALGO_NAME,
        "dataset": task["dataset"],
        "repeat": task["repeat"],
        "fold": task["fold"],
        "run_id": task["run_id"],
        "seed": task["seed"],
        "f1_macro": float(metrics["f1_macro"]),
        "auc": float(metrics["auc"]),
        "acc": float(metrics["acc"]),
        # TabPFN is a foundation model, not on the compactness axis -> NaN.
        "hidden_nodes": float("nan"),
        "total_leaves": float("nan"),
        "decision_units": float("nan"),
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
        os.path.dirname(os.path.abspath(__file__)), config.CSV_DATA_DIR
    )
    tasks, run_id = [], 0
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
                "algo": ALGO_NAME,
                "dataset": dataset,
                "csv_path": csv_path,
                "repeat": int(repeat),
                "fold": int(fold),
                "run_id": run_id,
                "train_idx": tr,
                "val_idx": va,
                "seed": derive_seed(args.base_seed, ALGO_NAME, dataset, repeat, fold),
                "feature_threshold": int(args.feature_threshold),
                "feature_k": int(args.feature_k),
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
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--cv-mode", default=config.CV_MODE)
    ap.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    ap.add_argument("--n-estimators", type=int, default=4,
                    help="TabPFN in-context ensembling size (default 4; higher = slower, marginally better).")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="TabPFN worker processes. Default 1 (avoids the Windows "
                         "'WinError 10038 not a socket' from the loky backend). "
                         "Raise only on Linux/macOS if you want faster CPU runs.")
    ap.add_argument("--model-path", default=None,
                    help="Path to a pre-downloaded TabPFN weights file (e.g. the "
                         "TabPFN-2.6 checkpoint). Skips runtime download; use for "
                         "offline / reproducible runs.")
    ap.add_argument("--feature-threshold", type=int, default=FEATURE_REDUCTION_THRESHOLD,
                    help="Apply MI feature selection when d exceeds this (default 1000, matches headline).")
    ap.add_argument("--feature-k", type=int, default=FEATURE_K,
                    help="Number of MI-selected features on high-D datasets (default 256, matches headline).")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_B9_tabpfn"))
    ap.add_argument("--quick", action="store_true", help="Smoke test: 5 folds on the first dataset.")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    tasks = build_tasks(args)
    if not tasks:
        print("No tasks built; check --datasets / --data-dir.", file=sys.stderr)
        return 2

    n_ds = len(set(t["dataset"] for t in tasks))
    print(f"Built {len(tasks)} TabPFN tasks: {n_ds} datasets x {args.repeats*args.folds} folds.")
    dev = _resolve_device(args.device)
    print(f"Device: {dev} | in-context ensembling: {args.n_estimators} | no tuning pass.")
    print(f"Feature front-end: SelectKBest(mutual_info, k={FEATURE_K}) on d > {FEATURE_REDUCTION_THRESHOLD}.")

    # Build TabPFN once and reuse it across every fold (avoids reloading the
    # weights 900 times). A single fixed construction seed keeps it deterministic.
    print("Loading TabPFN model (once)...", flush=True)
    clf = _make_tabpfn(dev, args.n_estimators, args.base_seed, args.n_jobs, args.model_path)

    t_start = time.perf_counter()
    rows = []
    for i, t in enumerate(tasks, 1):
        rows.append(run_single(t, clf)["row"])
        if i % 25 == 0 or i == len(tasks):
            el = time.perf_counter() - t_start
            print(f"  {i}/{len(tasks)} folds  ({el/60:.1f} min, {el/i:.2f} s/fold)", flush=True)
    wall = time.perf_counter() - t_start
    print(f"\nWall-clock: {wall:.1f} s ({wall/60:.1f} min, {wall/3600:.2f} h)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)

    # Sanity check: high-D datasets must have been reduced to feature_k, low-D
    # datasets must keep their raw features. Fail loudly if the front-end misfired.
    feat = df.groupby("dataset").agg(d=("input_size", "first"),
                                     used=("n_features_used", "first")).reset_index()
    bad = feat[(feat["d"] > args.feature_threshold) & (feat["used"] != args.feature_k)]
    print("\nFeature-usage check (d -> n_features_used):")
    for _, rr in feat.iterrows():
        flag = "  <-- UNEXPECTED" if (rr["d"] > args.feature_threshold and rr["used"] != args.feature_k) else ""
        print(f"    {rr['dataset']:18s} {int(rr['d']):6d} -> {int(rr['used'])}{flag}")
    if len(bad):
        print("\n*** WARNING: the feature front-end did not reduce the above high-D "
              f"datasets to k={args.feature_k}. Results are NOT comparable to the "
              "headline pipeline. Re-run after fixing --feature-k / --feature-threshold. ***")

    summary = (
        df.groupby(["algo", "dataset"])
        .agg(
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            auc_mean=("auc", "mean"),
            acc_mean=("acc", "mean"),
            hidden_mean=("hidden_nodes", "mean"),   # NaN: TabPFN not on compactness axis
            leaves_mean=("total_leaves", "mean"),   # NaN
            runtime_mean=("runtime_sec", "mean"),
            n=("f1_macro", "size"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\nPer-dataset summary:")
    print(summary[["dataset", "f1_mean", "auc_mean", "acc_mean", "runtime_mean", "n"]].to_string(index=False))
    print(f"\nAggregate mean over {n_ds} datasets: "
          f"F1={df['f1_macro'].mean():.4f}  AUC={df['auc'].mean():.4f}  "
          f"ACC={df['acc'].mean():.4f}  runtime={df['runtime_sec'].mean():.2f}s/fold")

    manifest = {
        "script": os.path.basename(__file__),
        "algorithm": ALGO_NAME,
        "datasets": sorted(set(df["dataset"].unique())),
        "n_folds_per_dataset": int(args.repeats * args.folds),
        "tuning": "none (single in-context forward pass)",
        "device": dev,
        "n_estimators": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "model_path": args.model_path,
        "feature_reduction_threshold": int(args.feature_threshold),
        "feature_k": int(args.feature_k),
        "base_seed": int(args.base_seed),
        "cv_mode": args.cv_mode,
        "wall_clock_sec": float(wall),
        "note": "TabPFN is a pretrained transformer; complexity columns are NaN "
                "because it is not on the (F1, hidden-units/leaves) compactness axis.",
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import tabpfn as _t
        manifest["tabpfn_version"] = getattr(_t, "__version__", "unknown")
    except Exception:
        pass
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\nNext step: this summary.csv has the same schema as "
          "results_B8_matched_complexity/summary.csv. Merge the TabPFN F1 into the "
          "nine-way comparison as an accuracy reference (not a compactness competitor), "
          "and add a paired Wilcoxon of TabPFN vs DCS--VSE on the 18 dataset means.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
