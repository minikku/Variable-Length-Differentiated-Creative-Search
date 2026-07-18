#!/usr/bin/env python3
"""One-time converter: MATLAB ``.mat`` datasets  ->  tidy CSV files.

Each source ``.mat`` carries ``data_input`` (n_samples x n_features) and
``data_output`` (n_samples x 1, 0-indexed integer class labels).  This script
writes ``data/<NAME>.csv`` with columns ``f0..f{D-1}`` plus a final ``label``
column.  Normalisation and one-hot encoding are deliberately *not* baked into
the CSV -- they are applied at load time (see ``vse/dataset.py``) so the CSV is
the raw, inspectable source of truth.

Run this once before running experiments:

    python convert_datasets.py                 # convert everything in config
    python convert_datasets.py --only IRIS LEUKEMIA1
    python convert_datasets.py --mat-dir /path/to/Datasets
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

import config


def _here(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def convert_one(name: str, mat_stem: str, mat_dir: str, out_dir: str) -> str:
    mat_path = os.path.join(mat_dir, mat_stem + ".mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)
    m = loadmat(mat_path)
    if "data_input" not in m or "data_output" not in m:
        raise KeyError(
            f"{mat_path} is missing data_input/data_output "
            f"(keys: {[k for k in m if not k.startswith('__')]})"
        )
    x = np.asarray(m["data_input"], dtype=float)
    y = np.asarray(m["data_output"]).ravel().astype(int)
    if x.shape[0] != y.shape[0]:
        # Some files store features transposed; align to label length.
        if x.shape[1] == y.shape[0]:
            x = x.T
        else:
            raise ValueError(
                f"{name}: shape mismatch x={x.shape} y={y.shape}"
            )
    cols = [f"f{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=cols)
    df["label"] = y
    out_path = os.path.join(out_dir, name + ".csv")
    df.to_csv(out_path, index=False)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert .mat datasets to CSV.")
    ap.add_argument("--mat-dir", default=None,
                    help="Folder containing the source .mat files "
                         "(default: config.MAT_SOURCE_DIR)")
    ap.add_argument("--out-dir", default=None,
                    help="Output folder for CSVs (default: config.CSV_DATA_DIR)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Convert only these logical dataset names")
    ap.add_argument("--force", action="store_true",
                    help="Re-convert even if the CSV already exists")
    args = ap.parse_args(argv)

    mat_dir = args.mat_dir or _here(config.MAT_SOURCE_DIR)
    out_dir = args.out_dir or _here(config.CSV_DATA_DIR)
    os.makedirs(out_dir, exist_ok=True)

    names = args.only if args.only else list(config.DATASET_MAT_MAP.keys())
    ok, skipped, failed = 0, 0, 0
    for name in names:
        if name not in config.DATASET_MAT_MAP:
            print(f"  ?  {name}: not in DATASET_MAT_MAP, skipping", file=sys.stderr)
            failed += 1
            continue
        out_path = os.path.join(out_dir, name + ".csv")
        if os.path.exists(out_path) and not args.force:
            print(f"  =  {name}: already converted")
            skipped += 1
            continue
        try:
            p = convert_one(name, config.DATASET_MAT_MAP[name], mat_dir, out_dir)
            df = pd.read_csv(p)
            print(f"  OK {name}: {df.shape[0]} rows x {df.shape[1]-1} features, "
                  f"{df['label'].nunique()} classes -> {os.path.relpath(p)}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  !! {name}: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone. converted={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
