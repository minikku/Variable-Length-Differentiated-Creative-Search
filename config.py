"""Central experiment configuration.

Edit the values here (or override them on the command line, see ``main.py``)
to control which algorithms / datasets are run, the optimisation budget, the
repeated cross-validation design, and the degree of parallelism.

This is the Python equivalent of the header section of ``Main_Parallel.m``.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Algorithms                                                                   #
# --------------------------------------------------------------------------- #
# Every name below maps to an implementation registered in
# ``vse.algorithms.registry``.  These are the MATLAB ``*_d`` algorithms ported
# to Python (the "direct optimisation" / variable-structure-encoding group).
ALL_ALGORITHMS = [
    "GProp_d",
    "GLOProp_d",
    "VLPSO_d",
    "DCS_noVSE_d",
    "DCS_VSE_hist_d",
    "CoDE_d",
    "DBA_d",
    "COLSHADE_d",
    "DCS_VSE_DKA_opt_0_d",
    "DCS_VSE_DKA_opt_1_d",
    "DCS_VSE_DKA_opt_2_d",
    "DCS_VSE_DKA_opt_3_d",
    "DCS_VSE_DKA_opt_4_d",
    "DCS_VSE_DKA_opt_5_d",
    "DCS_VSE_DKA_opt_6_d",
    "DCS_VSE_DKA_opt_7_d",
    "DCS_VSE_DKA_opt_8_d",
    "DCS_VSE_DKA_opt_9_d",
    "DCS_VSE_DKA_opt_10_d",
    "DCS_VSE_DKA_opt_11_d",
    "DCS_VSE_DKA_opt_12_d",
    "DCS_VSE_DKA_opt_13_d",
]

# Algorithms actually executed by default (mirrors ``selected_algo`` in MATLAB).
SELECTED_ALGORITHMS = [
    "GProp_d",
    "VLPSO_d",
    "DCS_noVSE_d",
    "CoDE_d",
    "DBA_d",
    "COLSHADE_d",
    "DCS_VSE_DKA_opt_0_d",
    "DCS_VSE_DKA_opt_1_d",
    "DCS_VSE_DKA_opt_2_d",
    "DCS_VSE_DKA_opt_3_d",
    "DCS_VSE_DKA_opt_4_d",
    "DCS_VSE_DKA_opt_5_d",
    "DCS_VSE_DKA_opt_6_d",
    "DCS_VSE_DKA_opt_7_d",
    "DCS_VSE_DKA_opt_8_d",
    "DCS_VSE_DKA_opt_9_d",
    "DCS_VSE_DKA_opt_10_d",
    "DCS_VSE_DKA_opt_11_d",
    "DCS_VSE_DKA_opt_12_d",
    "DCS_VSE_DKA_opt_13_d",
]

# --------------------------------------------------------------------------- #
# Datasets                                                                     #
# --------------------------------------------------------------------------- #
# Dataset *logical* names (left) -> source ``.mat`` filename stem (right).
# The converter (``convert_datasets.py``) reads ``Datasets/<stem>.mat`` and
# writes ``data/<logical name>.csv``.  Add or remove entries freely.
DATASET_MAT_MAP = {
    "DYSLEXIA": "DYSLEXIA_data",
    "CANCER": "CANCER_data",
    "IRIS": "IRIS_data",
    "GLASS": "GLASS_data",
    "DNAHELICASES": "DNAHELICASES_data",
    "DIABETES": "DIABETES_data",
    "DYSLEXIA_10p": "DYSLEXIA_10p_data",
    "BANKNOTE": "BANKNOTE_data",
    "BANKNOTE_10p": "BANKNOTE_10p_data",
    "WINEQUALITY_RED": "WINEQUALITY_RED_data",
    "WINEQUALITY_WHITE": "WINEQUALITY_WHITE_data",
    "ELECTRICITYTHEFT": "ELECTRICITYTHEFT_data",
    "ELECTRICITYTHEFT_BL": "ELECTRICITYTHEFT_BL_data",
    "LEUKEMIA1": "LEUKEMIA1_data",
    "LEUKEMIA2": "LEUKEMIA2_data",
    "HCV": "HCV_data",
    "HEPATITIS": "HEPATITIS_data",
    "ILPD": "ILPD_data",
    "LD": "LD_data",
    "11_TUMORS": "11_TUMORS_data",
    "9_TUMORS": "9_TUMORS_data",
    "ADENOCARCINOMA": "ADENOCARCINOMA_data",
    "BRAIN_TUMOR1": "BRAIN_TUMOR1_data",
    "BRAIN_TUMOR2": "BRAIN_TUMOR2_data",
    "BREAST3": "BREAST3_data",
    "DLBCL": "DLBCL_data",
    "LUNG_CANCER": "LUNG_CANCER_data",
    "LYMPHOMA": "LYMPHOMA_data",
    "NCI": "NCI_data",
    "PROSTATE6033": "PROSTATE6033_data",
    "PROSTATE_TUMOR": "PROSTATE_TUMOR_data",
    "SRBCT": "SRBCT_data",
    # ---- New healthcare datasets fetched directly to CSV by ----
    # ---- experiment_py/add_healthcare_datasets.py             ----
    # The .mat-stem value is unused for these because the CSV is written
    # directly by the fetch script; convert_datasets.py will skip them.
    "PIMA_DIABETES":    "",   # Pima Indians Diabetes (OpenML name=diabetes, id=37)
    "WDBC":             "",   # Wisconsin Diagnostic Breast Cancer (sklearn / UCI 17)
    "HEART_CLEVELAND":  "",   # Heart Disease, Cleveland (UCI 45, binarized)
    "PARKINSONS":       "",   # Parkinsons voice biomarker (UCI 174, drop 'name')
    "HEART_FAILURE":    "",   # Heart Failure Clinical Records (UCI 519)
    "VERTEBRAL_COLUMN": "",   # Vertebral Column, orthopedic (UCI 212, binarized)
    "COLON_ALON":       "",   # Colon cancer microarray (OpenML name=Colon)
    "HABERMAN_SURVIVAL":"",   # Haberman's Survival (UCI 43, 3 features, 306 instances)
}

# Datasets actually executed by default (mirrors ``selected_problem`` in MATLAB).
SELECTED_DATASETS = [
    # ---- Original 9-dataset benchmark ----
    "DYSLEXIA",
    "DYSLEXIA_10p",
    "LEUKEMIA1",
    "LEUKEMIA2",
    "ILPD",
    "ADENOCARCINOMA",
    "DLBCL",
    "PROSTATE6033",
    "PROSTATE_TUMOR",
    # ---- Added by add_healthcare_datasets.py (lifts n_ds=9 -> 17) ----
    "PIMA_DIABETES",     # Pima Indians Diabetes (8 features, 768 instances)
    "WDBC",              # Wisconsin Diagnostic Breast Cancer (30 features, 569 instances)
    "HEART_CLEVELAND",   # Heart Disease, Cleveland (13 features, 303 instances)
    "HEPATITIS",         # Hepatitis prognosis (19 features, 155 instances)
    "PARKINSONS",        # Parkinsons voice biomarker (22 features, 195 instances)
    "HEART_FAILURE",     # Heart Failure Clinical Records (12 features, 299 instances)
    "VERTEBRAL_COLUMN",  # Vertebral Column, orthopedic (6 features, 310 instances)
    "COLON_ALON",        # Colon cancer microarray (2000 features, 62 instances)
    "HABERMAN_SURVIVAL", # Haberman's Survival, breast-cancer survival (3 features, 306 instances)
]

# --------------------------------------------------------------------------- #
# Optimisation parameters (mirror ``options`` in Main_Parallel.m)             #
# --------------------------------------------------------------------------- #
POPSIZE = 30
MAX_NFE = 30000
MIN_HIDDEN_SIZE = 2 #1
MAX_HIDDEN_SIZE = 2 #20
LB = -10.0
UB = 10.0

# --------------------------------------------------------------------------- #
# Repeated stratified cross-validation                                        #
# --------------------------------------------------------------------------- #
# Total independent runs per (algorithm, dataset) = N_REPEATS * N_FOLDS.
# The original experiment used 52 parallel runs on a single split; here every
# run trains on a different stratified fold for a fair comparison.
N_REPEATS = 5
N_FOLDS = 10
# Fraction of each class assigned to training inside a fold (used only when
# CV_MODE == "holdout").  K-fold ignores this.
CV_MODE = "repeated_stratified_kfold"  # or "repeated_stratified_holdout"
HOLDOUT_TRAIN_FRACTION = 0.8

# Base seed; per-run seeds are derived deterministically from this so the whole
# experiment is reproducible across machines and OSes.
BASE_SEED = 20240611

# --------------------------------------------------------------------------- #
# Parallelism / execution                                                     #
# --------------------------------------------------------------------------- #
N_JOBS = 26           # number of parallel worker processes (adjustable)
JOBLIB_BACKEND = "loky"

# --------------------------------------------------------------------------- #
# Paths (relative to the experiment_py package root)                          #
# --------------------------------------------------------------------------- #
MAT_SOURCE_DIR = "../experiment_codes/Datasets"   # where the .mat files live
CSV_DATA_DIR = "data"                              # converted CSVs land here
RESULTS_DIR = "results_x"                            # experiment outputs land here
