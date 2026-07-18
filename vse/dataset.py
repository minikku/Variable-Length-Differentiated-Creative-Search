"""Dataset loading, normalisation, one-hot encoding and repeated CV splitting.

The original MATLAB ``DatasetInit`` loaded a ``.mat`` file, min-max normalised
each feature to [0, 1] (``mapminmax``), one-hot encoded the (0-indexed) integer
labels, then produced a single stratified 80/20 split.

Here we:
  * read the converted CSV (features + a final ``label`` column),
  * apply the same per-feature min-max normalisation,
  * one-hot encode the labels,
  * yield *repeated stratified K-fold* splits for a fair comparison.

Datasets are cached in-process so that the 52 parallel runs of one dataset only
parse/normalise the CSV once per worker.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
)


def mapminmax_01(x: np.ndarray) -> np.ndarray:
    """Per-feature min-max scaling to [0, 1] (matches MATLAB ``mapminmax(.,0,1)``).

    Constant features (max == min) are mapped to 0, matching ``mapminmax`` which
    leaves such rows at the lower target value.
    """
    x = np.asarray(x, dtype=float)
    xmin = x.min(axis=0)
    xmax = x.max(axis=0)
    span = xmax - xmin
    span_safe = np.where(span == 0, 1.0, span)
    out = (x - xmin) / span_safe
    out[:, span == 0] = 0.0
    return out


def one_hot(labels: np.ndarray) -> np.ndarray:
    """One-hot encode 0-indexed integer labels into (N, n_classes)."""
    labels = np.asarray(labels).astype(int).ravel()
    classes = np.unique(labels)
    n_classes = int(classes.max()) + 1 if classes.min() == 0 else len(classes)
    # Re-map labels onto a contiguous 0..C-1 range to be safe.
    remap = {c: i for i, c in enumerate(classes)}
    idx = np.array([remap[v] for v in labels])
    out = np.zeros((labels.shape[0], len(classes)), dtype=float)
    out[np.arange(labels.shape[0]), idx] = 1.0
    return out


class Dataset:
    """Normalised features + one-hot targets + integer labels for stratification."""

    def __init__(self, name: str, x: np.ndarray, y_onehot: np.ndarray,
                 y_int: np.ndarray):
        self.name = name
        self.x = x
        self.y = y_onehot
        self.y_int = y_int

    @property
    def input_size(self) -> int:
        return self.x.shape[1]

    @property
    def output_size(self) -> int:
        return self.y.shape[1]

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]


@lru_cache(maxsize=None)
def load_dataset(csv_path: str) -> Dataset:
    """Load and prepare a dataset from its converted CSV (cached)."""
    df = pd.read_csv(csv_path)
    label_col = df.columns[-1]
    y_int = df[label_col].to_numpy()
    x_raw = df.drop(columns=[label_col]).to_numpy(dtype=float)
    x = mapminmax_01(x_raw)
    y = one_hot(y_int)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    return Dataset(name, x, y, np.asarray(y_int).astype(int).ravel())


def make_cv_splits(ds: Dataset, n_repeats: int, n_folds: int, base_seed: int,
                   mode: str = "repeated_stratified_kfold",
                   holdout_train_fraction: float = 0.8):
    """Return a list of ``(repeat, fold, train_idx, val_idx)`` tuples.

    ``mode`` is either ``"repeated_stratified_kfold"`` (RepeatedStratifiedKFold)
    or ``"repeated_stratified_holdout"`` (n_repeats * n_folds independent
    stratified shuffle splits).  Both are deterministic given ``base_seed``.
    """
    splits = []
    if mode == "repeated_stratified_kfold":
        rskf = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=base_seed
        )
        for run_i, (tr, va) in enumerate(rskf.split(ds.x, ds.y_int)):
            repeat = run_i // n_folds
            fold = run_i % n_folds
            splits.append((repeat, fold, tr, va))
    elif mode == "repeated_stratified_holdout":
        total = n_repeats * n_folds
        for run_i in range(total):
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=holdout_train_fraction,
                random_state=base_seed + run_i,
            )
            tr, va = next(sss.split(ds.x, ds.y_int))
            splits.append((run_i // n_folds, run_i % n_folds, tr, va))
    else:
        raise ValueError(f"unknown CV mode: {mode}")
    return splits
