#!/usr/bin/env python3
"""B6 compare: paired Wilcoxon on DCS-VSE aligned vs padded.

Reads:
    results_B6_dcsvse_aligned_vs_padded/results.csv  (per-fold paired results)
    results_B6_dcsvse_aligned_vs_padded/summary.csv  (per-dataset means)

Answers the question Study B (B2) never asked directly:
    "Does max-embedding padding change the macro-F1 of the proposed
    DCS-VSE method itself, on the eight-dataset benchmark?"

Outputs
-------
    combined/b6_aligned_vs_padded.csv           per-dataset deltas
    combined/b6_aligned_vs_padded_summary.tex   LaTeX block for the revised
                                                Study B section
    combined/b6_per_fold_pairs.csv              per-(repeat, fold) deltas

Decision-tree printout
----------------------
The script prints a 4-way decision tree mapping the outcome onto a concrete
manuscript action:

    1. ALIGNED >> PADDED   (Holm-significant in favour of alignment)
       => Strengthen the alignment-as-contribution claim with this evidence.

    2. PADDED >> ALIGNED   (Holm-significant in favour of padding)
       => Same direction as B2 on COLSHADE; further weakens the alignment
          contribution.  Reframe Study B to emphasise alignment as a
          definitional condition only, not as performance-improving.

    3. TIED, mean delta < 0.02
       => Cleanest outcome.  Padding and alignment are empirically
          indistinguishable on the proposed method; the choice is
          definitional / conceptual.

    4. TIED, mean delta in [0.02, 0.05]
       => Direction of point estimate matters even though n=8 is
          under-powered for the dataset-mean test; report per-fold
          Wilcoxon for additional resolution.
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
ALIGNED_KEY = "DCS_noVSE_d:aligned"
PADDED_KEY = "DCS_noVSE_d:padded"

# v22 benchmark (eight datasets; LEUKEMIA1 excluded).
HIGH_D = {"ADENOCARCINOMA", "DLBCL", "LEUKEMIA2", "PROSTATE6033", "PROSTATE_TUMOR"}
LOW_D = {"DYSLEXIA", "DYSLEXIA_10p", "ILPD"}


def _subgroup(ds: str) -> str:
    if ds in HIGH_D:
        return "High-D microarray"
    if ds in LOW_D:
        return "Low-D clinical"
    return "Other"


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


def _wilcox(a, b):
    try:
        return float(wilcoxon(a, b, alternative="two-sided",
                              zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--b6", default=os.path.join(HERE,
                    "results_B6_dcsvse_aligned_vs_padded"))
    ap.add_argument("--out", default=os.path.join(HERE, "combined"))
    args = ap.parse_args(argv)

    summ = _read_summary(os.path.join(args.b6, "summary.csv"))
    pf   = _read_perfold(os.path.join(args.b6, "results.csv"))
    if summ is None or pf is None:
        print("Need both summary.csv and results.csv to compare.", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    datasets = sorted({k[1] for k in summ.keys() if k[0] == ALIGNED_KEY})

    rows = []
    aligned_arr, padded_arr = [], []
    for d in datasets:
        a = summ.get((ALIGNED_KEY, d))
        p = summ.get((PADDED_KEY, d))
        if a is None or p is None:
            print(f"  !! missing rows for {d}; skipping")
            continue
        af1 = float(a["f1_mean"]); pf1 = float(p["f1_mean"])
        aligned_arr.append(af1); padded_arr.append(pf1)
        rows.append({
            "dataset": d,
            "subgroup": _subgroup(d),
            "f1_aligned": af1,
            "f1_padded": pf1,
            "delta_aligned_minus_padded": af1 - pf1,
            "auc_aligned": float(a["auc_mean"]),
            "auc_padded": float(p["auc_mean"]),
            "hidden_aligned": float(a["hidden_mean"]),
            "hidden_padded": float(p["hidden_mean"]),
            "time_aligned": float(a.get("runtime_mean", "nan")),
            "time_padded": float(p.get("runtime_mean", "nan")),
        })

    if not rows:
        print("No paired rows.", file=sys.stderr)
        return 1

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out, "b6_aligned_vs_padded.csv"), index=False
    )

    aligned_arr = np.array(aligned_arr)
    padded_arr  = np.array(padded_arr)
    diff = aligned_arr - padded_arr

    print()
    print("=" * 78)
    print("  B6: proposed DCS-VSE aligned  vs  padded  (per-dataset means)")
    print("=" * 78)
    print(f"  aligned mean F1 = {aligned_arr.mean():.4f}")
    print(f"  padded  mean F1 = {padded_arr.mean():.4f}")
    print(f"  delta (aligned - padded)  = {diff.mean():+.4f}")
    print()
    print(f"  {'dataset':<18} {'subgroup':<18} {'aligned':>9} {'padded':>9} {'delta':>9}")
    print(f"  {'-'*18} {'-'*18} {'-'*9} {'-'*9} {'-'*9}")
    for r in rows:
        mark = "  *" if abs(r["delta_aligned_minus_padded"]) > 0.02 else ""
        print(f"  {r['dataset']:<18} {r['subgroup']:<18} "
              f"{r['f1_aligned']:>9.4f} {r['f1_padded']:>9.4f} "
              f"{r['delta_aligned_minus_padded']:>+9.4f}{mark}")
    print()
    print("  (* = absolute per-dataset delta > 0.02)")
    print()

    # Per-fold pairs (merge on dataset, repeat, fold)
    ad = pf[pf["algo"] == ALIGNED_KEY][["dataset", "repeat", "fold", "f1_macro"]] \
        .rename(columns={"f1_macro": "f1_aligned"})
    pd_ = pf[pf["algo"] == PADDED_KEY][["dataset", "repeat", "fold", "f1_macro"]] \
        .rename(columns={"f1_macro": "f1_padded"})
    merged = ad.merge(pd_, on=["dataset", "repeat", "fold"])
    merged["delta"] = merged["f1_aligned"] - merged["f1_padded"]
    merged.to_csv(os.path.join(args.out, "b6_per_fold_pairs.csv"), index=False)

    pd_p = _wilcox(aligned_arr, padded_arr)
    pf_p = _wilcox(merged["f1_aligned"], merged["f1_padded"]) if len(merged) > 0 else 1.0
    print(f"  per-fold paired Wilcoxon (n={len(merged)} matched folds): "
          f"p={pf_p:.4g}")
    print(f"  per-fold mean delta = {merged['delta'].mean():+.4f}")
    print(f"  per-dataset Wilcoxon ({len(rows)} datasets): p={pd_p:.4f}")
    print()

    # Subgroup
    for label, sg_set in [("High-D microarray", HIGH_D),
                          ("Low-D clinical", LOW_D)]:
        sub = merged[merged.dataset.isin(sg_set)]
        if len(sub) < 5:
            continue
        sp = _wilcox(sub["f1_aligned"], sub["f1_padded"])
        delta_mean = sub["delta"].mean()
        print(f"  [{label:<18}] n_folds={len(sub):3d}  "
              f"mean delta={delta_mean:+.4f}  Wilcoxon p={sp:.4g}")
    print()

    # Decision tree
    print("=" * 78)
    print("  DECISION TREE")
    print("=" * 78)
    if pd_p < 0.05 and diff.mean() > 0:
        outcome = "ALIGNED_BEATS_PADDED"
        msg = (
            "ALIGNED beats PADDED (per-dataset Wilcoxon p={p:.4f},\n"
            "  mean delta={d:+.4f} favouring alignment).  This is the\n"
            "  strongest empirical case for the cross-length alignment\n"
            "  contribution; rewrite Study B to emphasize that alignment\n"
            "  beats padding on the proposed method itself, in contrast\n"
            "  with the small effect detected on the three transfer-target\n"
            "  algorithms in B2 (CoDE p=0.11, COLSHADE p=0.04, DBA p=0.25).\n"
        ).format(p=pd_p, d=diff.mean())
    elif pd_p < 0.05 and diff.mean() < 0:
        outcome = "PADDED_BEATS_ALIGNED"
        msg = (
            "PADDED beats ALIGNED on the proposed method itself\n"
            "  (per-dataset Wilcoxon p={p:.4f}, mean delta={d:+.4f}\n"
            "  favouring padding).  This is consistent with B2 on COLSHADE\n"
            "  (p=0.039) and against CoDE/DBA in direction but not\n"
            "  significance.  Reframe Study B to make explicit that\n"
            "  cross-length alignment is a definitional condition\n"
            "  (parameter-type validity, no padding-distribution\n"
            "  dependence) and is not empirically preferred on macro-F1\n"
            "  in this regime; keep the proposed method's headline\n"
            "  Pareto-corner result, which does not depend on alignment\n"
            "  being F1-better than padding.\n"
        ).format(p=pd_p, d=diff.mean())
    elif abs(diff.mean()) < 0.02:
        outcome = "TIED_CLEAN"
        msg = (
            "TIED on the proposed method itself (per-dataset Wilcoxon\n"
            "  p={p:.4f}, mean delta={d:+.4f}).  Cleanest outcome.\n"
            "  Padding and alignment are empirically indistinguishable on\n"
            "  the proposed method; the choice is definitional /\n"
            "  conceptual.  Update Study B to report aligned=padded on\n"
            "  the proposed method directly; this resolves the\n"
            "  pre-submission reviewer point that B2 ran the\n"
            "  counterfactual on the wrong target.\n"
        ).format(p=pd_p, d=diff.mean())
    else:
        outcome = "TIED_BUT_NOTABLE_DELTA"
        msg = (
            "NOT-SIGNIFICANT at dataset-mean level but mean delta\n"
            "  ={d:+.4f}; the test is under-powered with n=8 datasets.\n"
            "  Examine the per-fold Wilcoxon (p={pf:.4g}) for added\n"
            "  resolution; if it is significant in the same direction,\n"
            "  report Study B with the dataset-mean p as the headline and\n"
            "  the per-fold p as a tie-breaker.\n"
        ).format(d=diff.mean(), pf=pf_p)

    print(f"\n  OUTCOME: {outcome}\n")
    for line in msg.splitlines():
        print(f"  {line}")
    print()

    # LaTeX block for the revised Study B
    tex_lines = [
        r"\begin{table}[ht]\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Comparison & $n_\mathrm{ds}$ & mean F1 aligned & mean F1 padded "
        r"& paired-$p$ \\",
        r"\midrule",
        f"  DCS--VSE (proposed) & {len(rows)} & {aligned_arr.mean():.4f} & "
        f"{padded_arr.mean():.4f} & {pd_p:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{B6: paired Wilcoxon signed-rank test of macro-F1 between "
        r"cross-length alignment and max-embedding padding on the proposed "
        r"DCS-VSE method itself, over the eight-dataset benchmark under the "
        r"identical $5\times 10$ stratified CV pipeline.  The corresponding "
        r"counterfactual on the three transfer-target algorithms (CoDE, "
        r"COLSHADE, DBA) is reported in Table~\ref{tab:b2_counterfactual}.}",
        r"\label{tab:b6_proposed_method_counterfactual}",
        r"\end{table}",
    ]
    out_tex = os.path.join(args.out, "b6_aligned_vs_padded_summary.tex")
    with open(out_tex, "w") as fh:
        fh.write("\n".join(tex_lines))

    print(f"  Wrote: {os.path.join(args.out, 'b6_aligned_vs_padded.csv')}")
    print(f"  Wrote: {out_tex}")
    print(f"  Wrote: {os.path.join(args.out, 'b6_per_fold_pairs.csv')}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
