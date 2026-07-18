#!/usr/bin/env python3
"""Add 8 healthcare tabular datasets to the existing nine-dataset benchmark.

Lifts n_ds = 9 -> 17 to improve Holm-correction statistical power in the
matched-complexity comparison. Each new dataset is fetched, cleaned, label-
encoded, and saved to experiment_py/data/ in the same ``f0,f1,...,fN,label``
CSV format used by the existing benchmark (see ``vse/dataset.py::load_dataset``,
which expects the last column to be the integer-encoded class label).

Datasets added (8)
------------------
Low-D clinical (d <= 30):
    PIMA_DIABETES     n=768,  d=8,  2-class   diabetes screening      (UCI 43)
    WDBC              n=569,  d=30, 2-class   breast cancer diag.     (sklearn)
    HEART_CLEVELAND   n=303,  d=13, 2-class   coronary disease        (UCI 45)
    HEPATITIS         n=155,  d=19, 2-class   liver disease prognosis (UCI 46)
    PARKINSONS        n=195,  d=22, 2-class   voice-based PD detect.  (UCI 174)
    HEART_FAILURE     n=299,  d=12, 2-class   mortality prediction    (UCI 519)
    VERTEBRAL_COLUMN  n=310,  d=6,  2-class   orthopedic screening    (UCI 212)

High-D microarray (complements the existing 6 microarray sets):
    COLON_ALON        n=62,   d=2000, 2-class colon-cancer microarray (OpenML)

Domain diversity covered: endocrinology, oncology, cardiology, hepatology,
neurology, orthopedics. Class-balance ratios span ~20% (HEPATITIS minority) to
~50% (LBPS, HEART_CLEVELAND after binarization).

Dependencies
------------
    pip install pandas numpy scikit-learn ucimlrepo

Usage
-----
    python add_healthcare_datasets.py
    python add_healthcare_datasets.py --datasets PIMA_DIABETES WDBC
    python add_healthcare_datasets.py --out experiment_py/data --force

After running
-------------
1. Open ``experiment_py/config.py``: add the eight new names to
   ``DATASET_MAT_MAP`` (mat-file value can be "" because the CSVs already exist)
   and append them to ``SELECTED_DATASETS``.

2. Re-run ``python main.py`` for the seven matched-SLNN methods on the new
   datasets (50 folds each).

3. Re-run ``python b8_matched_complexity_trees.py`` so LGBM/XGBoost are
   recomputed at total leaves <= H_max = 20 on the new datasets.

4. Re-compute the Friedman / Holm-Wilcoxon tables with n_ds = 17 and
   regenerate Tables 8, 10, 11, 12.

Why these eight
---------------
The existing nine-dataset benchmark is dominated by microarray (six of nine)
and dyslexia / liver datasets (the three low-D points). The eight new sets:
(a) increase low-D clinical coverage from 3 to 9, improving Holm power on
the cross-family comparison; (b) widen domain coverage to endocrinology,
cardiology, hepatology, neurology, and orthopedics; (c) add one additional
microarray set (COLON_ALON) so the high-D subgroup stays at 7 and the d/n
regime tested by Study G remains represented; (d) keep all sample sizes
above n = 60 so the existing 5x10 stratified CV protocol applies unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.datasets import load_breast_cancer, fetch_openml
except ImportError:
    load_breast_cancer = None
    fetch_openml = None

try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    LabelEncoder = None

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None


CSV_HEADER_FN = "f{}"
LABEL_COL = "label"


def _encode_categorical(X: pd.DataFrame) -> pd.DataFrame:
    """Encode any non-numeric column with LabelEncoder; preserve column order."""
    if LabelEncoder is None:
        raise RuntimeError("Install scikit-learn: pip install scikit-learn")
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return X


def save_csv(out_dir: str, name: str, X: pd.DataFrame, y: pd.Series, force: bool) -> None:
    """Write the dataset to ``<out_dir>/<name>.csv`` in the f0..fN,label format."""
    path = os.path.join(out_dir, f"{name}.csv")
    if os.path.exists(path) and not force:
        print(f"  [skip] {path} already exists; pass --force to overwrite", file=sys.stderr)
        return

    X = _encode_categorical(X)
    X.columns = [CSV_HEADER_FN.format(i) for i in range(X.shape[1])]

    # Defensively squeeze y to 1-D. Some ucimlrepo versions return targets as
    # a (n, 2) DataFrame when the dataset has both a target column and a
    # subject-id column tagged as target; np.asarray + squeeze + col-0 fallback
    # handles all shapes (n,), (n, 1), and (n, 2).
    if isinstance(y, pd.DataFrame):
        # Prefer a column named 'status', 'class', 'target', or 'label' if any.
        for pref in ("status", "Class", "class", "target", "label", "Class_att"):
            if pref in y.columns:
                y = y[pref]
                break
        else:
            y = y.iloc[:, 0]
    y_arr = np.asarray(y)
    if y_arr.ndim > 1:
        y_arr = y_arr.squeeze()
        if y_arr.ndim > 1:
            y_arr = y_arr[:, 0]
    y_int = LabelEncoder().fit_transform(pd.Series(y_arr).astype(str))

    df = pd.concat([X.reset_index(drop=True),
                    pd.Series(y_int, name=LABEL_COL)], axis=1)
    df = df.dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    df.to_csv(path, index=False)

    n, d = len(df), X.shape[1]
    K = int(df[LABEL_COL].nunique())
    counts = df[LABEL_COL].value_counts(normalize=True).sort_index().tolist()
    minfrac = min(counts) * 100
    print(f"  [save] {name}: n={n}, d={d}, K={K}, "
          f"minority={minfrac:.2f}%  -> {path}")


def _fetch_uci(uci_id: int) -> Tuple[pd.DataFrame, pd.Series]:
    if fetch_ucirepo is None:
        raise RuntimeError("Install ucimlrepo: pip install ucimlrepo")
    ds = fetch_ucirepo(id=uci_id)
    X = ds.data.features
    y = ds.data.targets
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return X, y


# --------------------------------------------------------------------------- #
# Per-dataset preparation routines                                            #
# --------------------------------------------------------------------------- #
def prepare_pima(out_dir: str, force: bool) -> None:
    """Pima Indians Diabetes Database.

    Pima was removed from the new UCI repository and is no longer fetchable
    via ``ucimlrepo``. UCI id 43 in the new system is Haberman's Survival
    (different dataset). The canonical Pima Indians Diabetes is available on
    OpenML as ``name="diabetes"`` (data_id=37); the script tries OpenML by name
    first and falls back to direct id, then to a GitHub mirror as a last resort.
    """
    if fetch_openml is None:
        raise RuntimeError("Install scikit-learn: pip install scikit-learn")

    X = y = None
    for kwargs in [
        dict(name="diabetes", version=1),
        dict(data_id=37),
        dict(name="Diabetes", version=1),
    ]:
        try:
            d = fetch_openml(**kwargs, as_frame=True)
            X, y = d.data, d.target
            break
        except Exception:
            continue

    if X is None:
        # Last-resort raw-CSV mirror (Brian Lewis / Plotly mirror of the
        # Pima Indians Diabetes Database).
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
               "pima-indians-diabetes.data.csv")
        df = pd.read_csv(url, header=None)
        cols = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
        df.columns = cols
        X = df.drop(columns=["class"])
        y = df["class"]

    save_csv(out_dir, "PIMA_DIABETES", X, y, force)


def prepare_haberman(out_dir: str, force: bool) -> None:
    """Haberman's Survival (UCI id 43): 3 features, 306 instances, 2-class.

    Included separately so the misfetched-as-Pima CSV can be relabeled
    without losing the data, and so this benchmark gets a third low-feature
    (d=3) clinical-survival dataset alongside the other low-D clinical sets.
    """
    X, y = _fetch_uci(43)
    save_csv(out_dir, "HABERMAN_SURVIVAL", X, y, force)


def prepare_wdbc(out_dir: str, force: bool) -> None:
    """Wisconsin Diagnostic Breast Cancer. sklearn bundle or UCI id 17."""
    if load_breast_cancer is not None:
        d = load_breast_cancer(as_frame=True)
        X = d.data
        y = d.target
    else:
        X, y = _fetch_uci(17)
    save_csv(out_dir, "WDBC", X, y, force)


def prepare_heart_cleveland(out_dir: str, force: bool) -> None:
    """Heart Disease (Cleveland). UCI id 45. Multi-class severity binarized."""
    X, y = _fetch_uci(45)
    y_bin = (pd.to_numeric(y, errors="coerce").fillna(0).astype(int) > 0).astype(int)
    save_csv(out_dir, "HEART_CLEVELAND", X, y_bin, force)


def prepare_hepatitis(out_dir: str, force: bool) -> None:
    """Hepatitis prognosis. UCI id 46. Class 1 = DIE (minority), 2 = LIVE."""
    X, y = _fetch_uci(46)
    y_bin = (pd.to_numeric(y, errors="coerce") == 1).astype(int)
    save_csv(out_dir, "HEPATITIS", X, y_bin, force)


def prepare_parkinsons(out_dir: str, force: bool) -> None:
    """Parkinsons voice-based detection.

    Order of attempts:
      1. Direct UCI archive CSV at parkinsons.data (clean 195x24 schema with
         columns: name, MDVP:Fo(Hz), ..., status, ... -- unambiguous about
         which column is the target).
      2. ``ucimlrepo.fetch_ucirepo(id=174)`` as a fallback when the URL is
         blocked. Some ucimlrepo versions return targets as a (n, 2) frame
         containing both ``status`` and ``name``; ``save_csv`` defensively
         squeezes that down to 1-D, preferring the ``status`` column.

    Either way the subject-identifier ``name`` column is dropped from X
    before saving (it is a phon_R01_S01_* string, not a feature).
    """
    X = y = None
    last_err = None

    # 1) Direct UCI URL.
    url = ("https://archive.ics.uci.edu/ml/"
           "machine-learning-databases/parkinsons/parkinsons.data")
    try:
        df = pd.read_csv(url)
        y = df["status"]
        X = df.drop(columns=["status"])
    except Exception as e1:
        last_err = e1
        print(f"  [info] direct UCI URL failed for Parkinsons "
              f"({type(e1).__name__}); trying ucimlrepo", file=sys.stderr)

    # 2) ucimlrepo fallback.
    if X is None:
        try:
            X, y = _fetch_uci(174)
        except Exception as e2:
            raise RuntimeError(
                f"Both Parkinsons fetch paths failed. "
                f"Direct URL: {type(last_err).__name__}: {last_err}. "
                f"ucimlrepo: {type(e2).__name__}: {e2}")

    # Drop the subject-identifier column if present (string, not a feature).
    for c in ("name", "Name", "subject", "id", "ID"):
        if c in X.columns:
            X = X.drop(columns=[c])

    save_csv(out_dir, "PARKINSONS", X, y, force)


def prepare_heart_failure(out_dir: str, force: bool) -> None:
    """Heart Failure Clinical Records. UCI id 519."""
    X, y = _fetch_uci(519)
    save_csv(out_dir, "HEART_FAILURE", X, y, force)


def prepare_vertebral_column(out_dir: str, force: bool) -> None:
    """Vertebral Column (UCI id 212).

    Same underlying clinical-orthopedic data as the Kaggle "Lower Back Pain
    Symptoms" dataset; UCI hosts it directly and reliably, while the Kaggle
    GitHub mirrors are unstable. The original target has three classes
    (Hernia, Spondylolisthesis, Normal); it is binarized to Normal vs.\
    Abnormal so the split matches the 2-class clinical-screening convention
    used by the other low-D datasets in this benchmark.
    """
    try:
        X, y = _fetch_uci(212)
    except Exception as e:
        print(f"  [info] ucimlrepo failed for Vertebral Column ({type(e).__name__}); "
              "trying direct UCI URL", file=sys.stderr)
        # UCI 212 ships a column file with no header.
        url = ("https://archive.ics.uci.edu/ml/"
               "machine-learning-databases/00212/vertebral_column_data.zip")
        raise RuntimeError(
            "ucimlrepo unreachable; download UCI dataset 212 manually from "
            f"{url} and place the unzipped column_2C.dat at experiment_py/data/, "
            "or install ucimlrepo: pip install ucimlrepo.")

    # Binarize: "Normal" -> 0, anything else -> 1 (abnormal: Hernia or Spondylolisthesis).
    y_str = pd.Series(y).astype(str).str.strip().str.upper()
    y_bin = (y_str != "NORMAL").astype(int)
    save_csv(out_dir, "VERTEBRAL_COLUMN", X, y_bin, force)


def prepare_colon_alon(out_dir: str, force: bool) -> None:
    """Alon colon-cancer microarray. OpenML name='Colon'."""
    if fetch_openml is None:
        raise RuntimeError("Install scikit-learn: pip install scikit-learn")
    # Common OpenML aliases for the Alon dataset.
    for kwargs in [
        dict(name="Colon", version=1),
        dict(name="colon", version=1),
        dict(data_id=45065),
        dict(data_id=1431),
    ]:
        try:
            d = fetch_openml(**kwargs, as_frame=True)
            break
        except Exception:
            d = None
            continue
    if d is None:
        raise RuntimeError(
            "Could not locate the Alon colon-cancer dataset on OpenML. "
            "Download manually from http://genomics-pubs.princeton.edu/oncology/affydata/index.html "
            "and adapt this routine.")
    X = d.data
    y = d.target
    save_csv(out_dir, "COLON_ALON", X, y, force)


DATASETS = {
    "PIMA_DIABETES":    prepare_pima,
    "HABERMAN_SURVIVAL": prepare_haberman,
    "WDBC":             prepare_wdbc,
    "HEART_CLEVELAND":  prepare_heart_cleveland,
    "HEPATITIS":        prepare_hepatitis,
    "PARKINSONS":       prepare_parkinsons,
    "HEART_FAILURE":    prepare_heart_failure,
    "VERTEBRAL_COLUMN": prepare_vertebral_column,
    "COLON_ALON":       prepare_colon_alon,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS.keys()),
                    default=None, help="Subset of dataset names to prepare (default: all 8).")
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="Output directory (default: experiment_py/data).",
    )
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing CSVs.")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    targets = args.datasets or list(DATASETS.keys())

    print(f"Output: {args.out}")
    print(f"Datasets to prepare: {', '.join(targets)}\n")

    n_ok = 0
    for name in targets:
        try:
            DATASETS[name](args.out, args.force)
            n_ok += 1
        except Exception as e:
            print(f"  [error] {name}: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\nDone: {n_ok}/{len(targets)} datasets written to {args.out}.")
    print("\nNext steps:")
    print("  1. Append the new names to experiment_py/config.py::SELECTED_DATASETS.")
    print("  2. python main.py                                  # run the 7 matched-SLNN methods")
    print("  3. python b8_matched_complexity_trees.py           # run matched-LGBM and matched-XGBoost")
    print("  4. Re-aggregate Friedman / Holm-Wilcoxon tables at n_ds = 17.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
