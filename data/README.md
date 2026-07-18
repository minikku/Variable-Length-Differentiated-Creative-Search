# Datasets

The eighteen benchmark datasets are public biomedical tabular datasets. They are **not
committed** to this repository (see the top-level `.gitignore`); regenerate them locally.

## Dataset groups

- **Low-dimensional clinical** (tens of features): e.g. HABERMAN_SURVIVAL, WDBC,
  HEART_CLEVELAND, HEART_FAILURE, PIMA_DIABETES, HEPATITIS, PARKINSONS, ILPD,
  DYSLEXIA / DYSLEXIA_10p, VERTEBRAL_COLUMN.
- **High-dimensional microarray** (thousands of gene-expression features): ADENOCARCINOMA,
  DLBCL, LEUKEMIA1, LEUKEMIA2, PROSTATE6033, PROSTATE_TUMOR, COLON_ALON.

Per-dataset sources, sample sizes, feature counts, and class balances are listed in the
paper (dataset table and its citations).

## How to build the CSVs

1. Obtain the raw dataset files from their original sources (cited in the paper) and place
   them where the conversion scripts expect them (raw `.mat` files under `MAT_Datasets/`,
   or provider-specific formats).
2. From the repository root, run:

   ```bash
   python convert_datasets.py          # builds the processed CSVs into data/
   python add_healthcare_datasets.py   # adds the clinical tabular sets
   ```

Each processed dataset is a CSV whose final column is the integer class label. The
experiment scripts load these CSVs by dataset name.
