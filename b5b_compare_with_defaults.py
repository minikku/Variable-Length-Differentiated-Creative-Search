#!/usr/bin/env python3
"""B5b vs default DCS-VSE (no-SO): paired test for the v17 fair-budget closure.

Reads:
    results_B5b_dcsvse_noso_optuna/results.csv     (per-fold tuned DCS-VSE no-SO)
    results_B5b_dcsvse_noso_optuna/summary.csv     (per-dataset means tuned)
    results_B5b_dcsvse_noso_optuna/best_params.csv (winning hyperparameters)
    results_H1to20/results.csv                     (per-fold default, filtered)
    results_H1to20/summary.csv                     (per-dataset means default)

Answers the v17 Study~F question on the proposed (no-SO) method:
    "Does Optuna-tuning the proposed DCS-VSE (no structural operators)
    lift its macro-F1 significantly above the default operating point,
    on the same 50 CV partitions per dataset?"

The DEFAULT row is filtered from results_H1to20 using algo == DCS_noVSE_d,
NOT DCS_VSE_DKA_opt_0_d.  This is the only structural difference from
b5_compare_with_defaults.py.

Outputs
-------
    combined/b5b_vs_default.csv               per-dataset numbers + delta
    combined/b5b_vs_default_summary.tex       LaTeX table for v17 Study F
    combined/b5b_per_fold_pairs.csv           per-(dataset,repeat,fold) deltas

Usage
-----
    python b5b_compare_with_defaults.py
    python b5b_compare_with_defaults.py --b5b results_B5b_dcsvse_noso_optuna \
                                        --h1to20 results_H1to20
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))

# v17 reframing: the *default* (unt-tuned) row is DCS_noVSE_d, not the +SO variant.
DEFAULT_ALGO_KEY = "DCS_noVSE_d"
TUNED_ALGO_KEY = "DCS_VSE_NoSO_OptunaTuned"

# Dimensionality-stratified subgroups, kept identical to the v17 Study F figure.
HIGH_D_MICROARRAY = {
    "ADENOCARCINOMA", "DLBCL", "LEUKEMIA1", "LEUKEMIA2",
    "PROSTATE6033", "PROSTATE_TUMOR",
}
LOW_D_CLINICAL = {"DYSLEXIA", "DYSLEXIA_10p", "ILPD"}


def _read_summary(path: str):
    if not os.path.exists(path):
        print(f"  !! missing summary: {path}", file=sys.stderr)
        return None
    return {(r["algo"], r["dataset"]): r for r in csv.DictReader(open(path))}


def _read_perfold(path: str):
    if not os.path.exists(path):
        print(f"  !! missing per-fold: {path}", file=sys.stderr)
        return None
    return pd.read_csv(path)


def _read_best(path: str):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _subgroup(ds: str) -> str:
    if ds in HIGH_D_MICROARRAY:
        return "High-D microarray"
    if ds in LOW_D_CLINICAL:
        return "Low-D clinical"
    return "Other"


def _wilcox(tuned, default):
    try:
        stat, p = wilcoxon(tuned, default,
                           alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return 1.0
    return float(p)


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--b5b", default=os.path.join(HERE, "results_B5b_dcsvse_noso_optuna"))
    ap.add_argument("--h1to20", default=os.path.join(HERE, "results_H1to20"))
    ap.add_argument("--out", default=os.path.join(HERE, "combined"))
    args = ap.parse_args(argv)

    tuned_summ = _read_summary(os.path.join(args.b5b, "summary.csv"))
    default_summ = _read_summary(os.path.join(args.h1to20, "summary.csv"))
    if tuned_summ is None or default_summ is None:
        print("Cannot run comparison without both summary files.", file=sys.stderr)
        return 1

    tuned_pf = _read_perfold(os.path.join(args.b5b, "results.csv"))
    default_pf = _read_perfold(os.path.join(args.h1to20, "results.csv"))
    best_params = _read_best(os.path.join(args.b5b, "best_params.csv"))

    datasets = sorted({k[1] for k in tuned_summ.keys() if k[0] == TUNED_ALGO_KEY})
    os.makedirs(args.out, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Per-dataset means table                                               #
    # --------------------------------------------------------------------- #
    rows = []
    tuned_arr, default_arr = [], []
    for d in datasets:
        t = tuned_summ.get((TUNED_ALGO_KEY, d))
        u = default_summ.get((DEFAULT_ALGO_KEY, d))
        if t is None or u is None:
            print(f"  !! missing rows for {d}; skipping (default key={DEFAULT_ALGO_KEY})")
            continue
        f1_t = float(t["f1_mean"]); f1_u = float(u["f1_mean"])
        tuned_arr.append(f1_t); default_arr.append(f1_u)
        rows.append({
            "dataset": d,
            "subgroup": _subgroup(d),
            "f1_default": f1_u,
            "f1_tuned": f1_t,
            "delta_tuned_minus_default": f1_t - f1_u,
            "auc_default": float(u["auc_mean"]),
            "auc_tuned": float(t["auc_mean"]),
            "hidden_default": float(u["hidden_mean"]),
            "hidden_tuned": float(t["hidden_mean"]),
            "time_default_s": float(u.get("runtime_mean", "nan")),
            "time_tuned_s": float(t.get("runtime_mean", "nan")),
        })

    if not rows:
        print("No paired rows; nothing to compare.", file=sys.stderr)
        return 1

    out_csv = os.path.join(args.out, "b5b_vs_default.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    tuned_arr = np.array(tuned_arr); default_arr = np.array(default_arr)
    diff = tuned_arr - default_arr

    print()
    print("=" * 78)
    print(f"  B5b (Optuna-tuned {TUNED_ALGO_KEY})  vs  default {DEFAULT_ALGO_KEY}")
    print(f"  v17 framing: proposed method = DCS-VSE (no-SO); +SO variant is ablation.")
    print(f"  Paired across {len(rows)} datasets on dataset-mean macro-F1")
    print("=" * 78)
    print()
    print(f"  default mean F1 = {default_arr.mean():.4f}")
    print(f"  tuned   mean F1 = {tuned_arr.mean():.4f}")
    print(f"  delta (tuned-default)         = {diff.mean():+.4f}")
    print()
    print(f"  {'dataset':<18} {'subgroup':<18} {'default':>9} {'tuned':>9} {'delta':>9}")
    print(f"  {'-'*18} {'-'*18} {'-'*9} {'-'*9} {'-'*9}")
    for r in rows:
        mark = "  *" if abs(r["delta_tuned_minus_default"]) > 0.02 else ""
        print(f"  {r['dataset']:<18} {r['subgroup']:<18} "
              f"{r['f1_default']:>9.4f} {r['f1_tuned']:>9.4f} "
              f"{r['delta_tuned_minus_default']:>+9.4f}{mark}")
    print()
    print("  (* = absolute per-dataset delta > 0.02)")
    print()

    # --------------------------------------------------------------------- #
    # Per-fold paired analysis (full aggregate + per-subgroup)              #
    # --------------------------------------------------------------------- #
    pf_merged = None
    if tuned_pf is not None and default_pf is not None:
        td = tuned_pf[tuned_pf["algo"] == TUNED_ALGO_KEY] \
            [["dataset", "repeat", "fold", "f1_macro"]].rename(
                columns={"f1_macro": "f1_tuned"})
        ud = default_pf[default_pf["algo"] == DEFAULT_ALGO_KEY] \
            [["dataset", "repeat", "fold", "f1_macro"]].rename(
                columns={"f1_macro": "f1_default"})
        pf_merged = td.merge(ud, on=["dataset", "repeat", "fold"])
        pf_merged["delta"] = pf_merged["f1_tuned"] - pf_merged["f1_default"]
        pf_merged.to_csv(os.path.join(args.out, "b5b_per_fold_pairs.csv"), index=False)

    # All-aggregate Wilcoxon (per-fold)
    per_fold_p = None
    if pf_merged is not None and len(pf_merged) >= 5:
        per_fold_p = _wilcox(pf_merged["f1_tuned"], pf_merged["f1_default"])
        print(f"  per-fold paired Wilcoxon (n={len(pf_merged)} matched folds): "
              f"p={per_fold_p:.4g}")
        print(f"  per-fold mean delta = {pf_merged['delta'].mean():+.4f}")
        print()

    # Subgroup-stratified per-fold Wilcoxon
    if pf_merged is not None:
        for sg_label, sg_set in [("High-D microarray", HIGH_D_MICROARRAY),
                                  ("Low-D clinical", LOW_D_CLINICAL)]:
            sub = pf_merged[pf_merged.dataset.isin(sg_set)]
            if len(sub) < 5:
                continue
            sp = _wilcox(sub["f1_tuned"], sub["f1_default"])
            wins = int((sub["delta"] > 0).sum())
            print(f"  [{sg_label:<18}] n_folds={len(sub):3d}  "
                  f"mean delta={sub['delta'].mean():+.4f}  "
                  f"wins={wins}/{len(sub)}  Wilcoxon p={sp:.4g}")
        print()

    # --------------------------------------------------------------------- #
    # Per-dataset-mean Wilcoxon                                             #
    # --------------------------------------------------------------------- #
    per_dataset_p = _wilcox(tuned_arr, default_arr)
    print(f"  per-dataset Wilcoxon ({len(rows)} datasets): p={per_dataset_p:.4f}")
    print()

    # Per-subgroup-of-datasets Wilcoxon (matches abstract framing)
    for sg_label, sg_set in [("High-D microarray", HIGH_D_MICROARRAY),
                              ("Low-D clinical", LOW_D_CLINICAL)]:
        sub = [r for r in rows if r["dataset"] in sg_set]
        if len(sub) < 2:
            continue
        t_arr = np.array([r["f1_tuned"] for r in sub])
        u_arr = np.array([r["f1_default"] for r in sub])
        sp = _wilcox(t_arr, u_arr)
        print(f"  [{sg_label:<18}] n_datasets={len(sub)}  "
              f"mean delta={(t_arr-u_arr).mean():+.4f}  "
              f"Wilcoxon p={sp:.4f}")
    print()

    # --------------------------------------------------------------------- #
    # Decision tree                                                         #
    # --------------------------------------------------------------------- #
    print("=" * 78)
    print("  DECISION TREE  (per-dataset Wilcoxon governs the v17 headline)")
    print("=" * 78)

    if per_dataset_p >= 0.05 and abs(diff.mean()) < 0.02:
        outcome = "TIE_AT_DEFAULT"
        msg = (
            "BEST CASE.  The Optuna sweep does NOT lift DCS-VSE (no-SO)\n"
            "  significantly above the default operating point\n"
            "  (per-dataset p={p:.3f}, mean delta={d:+.4f}).  The default is\n"
            "  near its own ceiling on these datasets.\n\n"
            "  v17 ACTION: Study F closure language reads as 'tuning does\n"
            "  not significantly lift the proposed method', which is the\n"
            "  cleanest possible defense against the asymmetric-tuning\n"
            "  critique.  Use B5b as the Study F dataset throughout the\n"
            "  paper; demote the existing B5 (+SO) results to a single line\n"
            "  in the ablation table.\n"
        ).format(p=per_dataset_p, d=diff.mean())
    elif per_dataset_p >= 0.05:
        outcome = "TIE_NON_SIGNIFICANT_BUT_NOTABLE_DELTA"
        msg = (
            "NOT-SIGNIFICANT at the aggregate but mean delta = {d:+.4f}.\n"
            "  Examine the subgroup-stratified Wilcoxons above for the\n"
            "  honest sub-story.  v17 reads as 'tuning is non-significant\n"
            "  in aggregate, with dataset-class-specific effects'.\n"
        ).format(d=diff.mean())
    elif diff.mean() > 0:
        outcome = "TUNED_BEATS_DEFAULT"
        msg = (
            "TUNED beats default significantly (p={p:.4f}, mean delta=\n"
            "  {d:+.4f}).  The default operating point in the headline\n"
            "  tables is under-tuned for the proposed (no-SO) method.\n\n"
            "  v17 ACTION: Rerun the headline DCS-VSE row of Tables 1-2\n"
            "  with the per-dataset best hyperparameters from\n"
            "  best_params.csv and update Section 5.3.  The Pareto-frontier\n"
            "  claim sharpens by approximately {d:+.3f} F1 points on average.\n"
        ).format(p=per_dataset_p, d=diff.mean())
    else:
        outcome = "TUNED_WORSE_THAN_DEFAULT"
        msg = (
            "TUNED is worse than default (p={p:.4f}, mean delta={d:+.4f}).\n"
            "  Likely Optuna overfitting to inner-CV.  Report transparently\n"
            "  as a negative tuning result -- still defends the asymmetric-\n"
            "  tuning critique -- but inspect optuna_trials.csv for the\n"
            "  pattern before finalising Study F.\n"
        ).format(p=per_dataset_p, d=diff.mean())

    print(f"\n  OUTCOME: {outcome}\n")
    for line in msg.splitlines():
        print(f"  {line}")
    print()

    # --------------------------------------------------------------------- #
    # Best-params table                                                     #
    # --------------------------------------------------------------------- #
    if best_params is not None and not best_params.empty:
        print("  Best hyperparameters per dataset (from optuna_trials):")
        for _, br in best_params.iterrows():
            print(f"    {br['dataset']}: popsize={br['popsize']} "
                  f"max_nfe={br['max_nfe']} lambda={br['lambda_size']:.4f} "
                  f"alpha={br['linnik_alpha']:.3f} scale={br['linnik_scale']:.4f}")
        print()

    # --------------------------------------------------------------------- #
    # LaTeX table for v17 Study F                                           #
    # --------------------------------------------------------------------- #
    tex_lines = [
        r"\begin{table*}[ht]\centering\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & default F1 & tuned F1 & $\Delta$ F1 & default $\bar{H}$ & tuned $\bar{H}$ \\",
        r"\midrule",
    ]
    for r in rows:
        d = r["dataset"].replace("_", r"\_")
        tex_lines.append(
            f"  {d} & {r['f1_default']:.4f} & {r['f1_tuned']:.4f} & "
            f"${r['delta_tuned_minus_default']:+.4f}$ & "
            f"{r['hidden_default']:.2f} & {r['hidden_tuned']:.2f} \\\\"
        )
    tex_lines.append(r"\midrule")
    tex_lines.append(
        f"  \\textbf{{Mean (n={len(rows)})}} & "
        f"\\textbf{{{default_arr.mean():.4f}}} & "
        f"\\textbf{{{tuned_arr.mean():.4f}}} & "
        f"\\textbf{{{diff.mean():+.4f}}} & "
        f"\\textbf{{{np.mean([r['hidden_default'] for r in rows]):.2f}}} & "
        f"\\textbf{{{np.mean([r['hidden_tuned'] for r in rows]):.2f}}} \\\\"
    )
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    cap = (
        r"\caption{Study~F (v17): per-dataset macro-F1 of the proposed "
        r"DCS--VSE (no structural operators) under the default operating point "
        r"(popsize $=30$, max\_nfe $=30{,}000$, $\lambda_{\mathrm{size}}=0.05$, "
        r"Linnik $\alpha=\varphi$, scale $=0.05$) and under an Optuna-tuned "
        r"operating point selected per dataset by an outer 40-trial study over "
        r"(popsize, max\_nfe, $\lambda_{\mathrm{size}}$, $\alpha$, scale) using "
        r"inner 5-fold CV. Final macro-F1 evaluated on the same 50 stratified "
        r"CV partitions per dataset. Paired Wilcoxon "
        f"$p={per_dataset_p:.4f}$ on dataset means (n={len(rows)}); "
        f"per-fold $p={per_fold_p:.4f}$ (n={len(pf_merged) if pf_merged is not None else 'NA'}). "
        r"This is the closure test on the proposed (no-SO) variant; the parallel "
        r"closure on the +SO ablation is reported in the supplementary.}"
    )
    tex_lines.append(cap)
    tex_lines.append(r"\label{tab:studyF_b5b_vs_default}")
    tex_lines.append(r"\end{table*}")

    out_tex = os.path.join(args.out, "b5b_vs_default_summary.tex")
    with open(out_tex, "w") as fh:
        fh.write("\n".join(tex_lines))

    print(f"  Wrote: {out_csv}")
    print(f"  Wrote: {out_tex}")
    if pf_merged is not None:
        print(f"  Wrote: {os.path.join(args.out, 'b5b_per_fold_pairs.csv')}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
