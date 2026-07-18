# DCS-VSE: Variable-Length Evolutionary Search for Compact Neural Models

Reference implementation for the paper

> **Enabling Variable-Length Evolutionary Search: Cross-Length Alignment, a Transformation Criterion, and Compact Neural Models for Biomedical Tabular Prediction.**
> Poomin Duankhan, Chitsutha Soomlek, Sirapat Chiewchanwattana, Khamron Sunat.
> Khon Kaen University. Submitted to *Intelligent Systems with Applications*.

DCS-VSE extends **Differentiated Creative Search (DCS)**, a rank-guided evolutionary
optimizer, so that it can search over the *structural size* of a solution and its
continuous parameters at the same time. The enabling primitive is **cross-length
alignment**: every cross-candidate operation is restricted to the shared prefix of the
two candidates, which keeps rank-guided, difference-based, and peer-transfer updates well
defined across candidates of different dimensionality. The same idea is packaged as a
reusable **transformation criterion** that decides which fixed-length optimizers can be
lifted to variable length. The method is instantiated on compact single-hidden-layer
networks (SLNNs) and a dendritic neuron model, and benchmarked on eighteen public
biomedical tabular datasets.

**Code:** [https://github.com/minikku/Variable-Length-Differentiated-Creative-Search](https://github.com/minikku/Variable-Length-Differentiated-Creative-Search)

## Repository layout

```
vse/                      Core library
  algorithms/             Optimizers (registry-based)
    dcs.py                DCS + DCS-VSE + the 14 DKA formations
    gprop.py  vlpso.py    G-Prop, variable-length PSO
    code.py  colshade.py  dba.py   DE-family competitors (aligned)
    extended.py           Size-gene adapter + 6 recent metaheuristics
                          (HOA, SSLO, ALA, MSO, EISA, FGO)
    base.py               AlgoResult / Individual, registry helpers
  network.py              Variable-size SLNN <-> vector encode/decode + operators
  mlp.py                  Forward pass
  metrics.py              Macro-F1, AUC, MCC-based loss
  evaluator.py            Size-regularized fitness
  dataset.py  rng.py      Data container, deterministic seeding

main.py                   Parallel repeated-CV runner (the main experiment driver)
config.py                 Default algorithms / datasets / CV design / budget
convert_datasets.py       Build the CSV datasets (run once)
add_healthcare_datasets.py

b1_*.py .. b11_*.py       Study scripts (baselines, ablations, positioning)
b_combine_stats.py        Friedman + Holm-Wilcoxon aggregation
make_*.py, rebuild_figs_v51.py, studyG_make_figure.py   Figure generators

requirements.txt  LICENSE  CITATION.cff
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/minikku/Variable-Length-Differentiated-Creative-Search.git
cd Variable-Length-Differentiated-Creative-Search
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

The core optimizer needs only `numpy`, `scipy`, `pandas`, and `joblib`. The tree and
transformer baselines additionally use `scikit-learn`, `lightgbm`, `xgboost`, and
`optuna` (all listed in `requirements.txt`). `tabpfn` is optional and only needed for the
TabPFN baseline (`b9_tabpfn.py`).

## Data

The eighteen benchmark datasets are public biomedical tabular datasets (clinical and
gene-expression microarray cohorts; see the paper for per-dataset citations and sources).
Place the raw files as described in `data/README.md`, then build the CSVs once:

```bash
python convert_datasets.py
```

Datasets and generated results are **not** shipped in this repository (see `.gitignore`);
they are regenerated from the scripts above.

## Quick start

Run a small smoke test, then a targeted experiment:

```bash
python main.py --quick                              # tiny sanity run
python main.py --algos DCS_noVSE_d CoDE_d FGO_d \
    --datasets ILPD DLBCL --repeats 5 --folds 10 --jobs 8
```

`DCS_noVSE_d` is the proposed **DCS-VSE**; `DCS_VSE_DKA_opt_0_d` is the structural-operator
ablation (DCS-VSE+SO). Every run is reproducible from `--base-seed`. Results are written as
CSV summaries under a results directory that the analysis and figure scripts read.

## Reproducing the paper

Each study in the paper maps to a script (run from the repository root):

| Script | Study |
|---|---|
| `main.py` (full config) | Main family comparison (13 variable-length optimizers) |
| `b3_plain_dcs_fixed_h.py` | Fixed-`H=1` necessity baseline |
| `b6_dcsvse_aligned_vs_padded.py` | Cross-length alignment vs padding (Study B) |
| `b1_tabular_baselines.py`, `b8_matched_complexity_trees.py` | Tree / matched-complexity references |
| `b9_tabpfn.py`, `b10_logreg.py` | TabPFN and footprint-matched logistic regression |
| `b7_sparse_prior_frontend.py` | Sparse-prior front-end (Study E) |
| `b11_dnm.py` | Dendritic neuron model deployment |
| `b_combine_stats.py` | Friedman + Holm-corrected Wilcoxon aggregation |

Figures are regenerated with the `make_*.py` and `rebuild_figs_v51.py` scripts after the
corresponding results exist.

## Using the optimizer as a library

```python
from vse.algorithms import get_algorithm

run = get_algorithm("DCS_noVSE_d")          # the proposed DCS-VSE
result = run(options, rng)                   # options carries the bounded objective
print(result.best_cost, result.best_net)     # best size-regularized fitness + network
```

New fixed-length optimizers can be made variable-length by registering them through the
size-gene adapter in `vse/algorithms/extended.py`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

Released under the MIT License (see `LICENSE`).
