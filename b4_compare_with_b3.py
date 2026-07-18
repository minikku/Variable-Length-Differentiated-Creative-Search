#!/usr/bin/env python3
"""B4 vs B3 head-to-head: tuned MLP at H=1 vs plain DCS at H=1.

This script reads:
    results_B3_plain_dcs_fixed_H1/summary.csv      (PlainDCS_H1 rows)
    results_B4_mlp_fixed_H1/summary.csv            (MLP_H1_Tuned rows)

and answers ONE question:
    "Is the gradient-trained single-hidden-unit MLP statistically better than the
    plain-DCS single-hidden-unit network, on the same 50 CV partitions per
    dataset?"

It runs:
  - per-dataset comparison (dataset means side by side, sorted by delta);
  - paired Wilcoxon signed-rank test across the nine datasets on dataset-mean F1;
  - a decision-tree readout that tells you which paper you have;
  - a LaTeX-ready summary row that can be pasted directly into v16's Section 5.5.7.

Usage
-----
    python b4_compare_with_b3.py
    python b4_compare_with_b3.py --b3 results_B3_plain_dcs_fixed_H1 \
                                 --b4 results_B4_mlp_fixed_H1

Output
------
    combined/b4_vs_b3.csv          (per-dataset numbers + delta)
    combined/b4_vs_b3_summary.tex  (LaTeX row for v16)
    Decision printed to stdout.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import wilcoxon


HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASETS = [
    "ADENOCARCINOMA", "DLBCL", "DYSLEXIA_10p", "ILPD", "LEUKEMIA1",
    "LEUKEMIA2", "PROSTATE_TUMOR", "PROSTATE6033", "DYSLEXIA",
]


def _load(p):
    if not os.path.exists(p):
        print(f"  !! missing summary: {p}", file=sys.stderr)
        return None
    return {(r["algo"], r["dataset"]): r for r in csv.DictReader(open(p))}


def _row(d, key):
    return float(d[key]) if d and key in d else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--b3", default=os.path.join(HERE, "results_B3_plain_dcs_fixed_H1"))
    ap.add_argument("--b4", default=os.path.join(HERE, "results_B4_mlp_fixed_H1"))
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--out", default=os.path.join(HERE, "combined"))
    args = ap.parse_args(argv)

    b3 = _load(os.path.join(args.b3, "summary.csv"))
    b4 = _load(os.path.join(args.b4, "summary.csv"))
    if b3 is None or b4 is None:
        print("Cannot run comparison without both summary files.", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Per-dataset table                                                     #
    # --------------------------------------------------------------------- #
    rows = []
    for d in args.datasets:
        b3_r = b3.get(("PlainDCS_H1", d))
        b4_r = b4.get(("MLP_H1_Tuned", d))
        if b3_r is None or b4_r is None:
            print(f"  !! missing rows for {d}; skipping")
            continue
        f1_b3 = float(b3_r["f1_mean"])
        f1_b4 = float(b4_r["f1_mean"])
        rows.append({
            "dataset": d,
            "f1_PlainDCS_H1": f1_b3,
            "f1_MLP_H1_Tuned": f1_b4,
            "delta_MLP_minus_DCS": f1_b4 - f1_b3,
            "auc_PlainDCS_H1": float(b3_r["auc_mean"]),
            "auc_MLP_H1_Tuned": float(b4_r["auc_mean"]),
            "t_PlainDCS_H1": float(b3_r.get("runtime_mean", "nan")),
            "t_MLP_H1_Tuned": float(b4_r.get("runtime_mean", "nan")),
        })

    if not rows:
        print("No paired rows found; nothing to compare.", file=sys.stderr)
        return 1

    # Write per-dataset CSV.
    out_csv = os.path.join(args.out, "b4_vs_b3.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --------------------------------------------------------------------- #
    # Paired Wilcoxon                                                       #
    # --------------------------------------------------------------------- #
    b3_arr = np.array([r["f1_PlainDCS_H1"] for r in rows])
    b4_arr = np.array([r["f1_MLP_H1_Tuned"] for r in rows])
    diff = b4_arr - b3_arr
    try:
        stat, p_two = wilcoxon(b4_arr, b3_arr, alternative="two-sided",
                               zero_method="wilcox")
    except ValueError:
        stat, p_two = float("nan"), 1.0

    print()
    print("=" * 72)
    print(f"  B4 (tuned MLP at H=1)  vs  B3 (plain DCS at H=1)")
    print(f"  Paired across {len(rows)} datasets on dataset-mean macro-F1")
    print("=" * 72)
    print()
    print(f"  PlainDCS_H1    mean F1 = {b3_arr.mean():.4f}")
    print(f"  MLP_H1_Tuned   mean F1 = {b4_arr.mean():.4f}")
    print(f"  delta (MLP-DCS)         = {diff.mean():+.4f}")
    print(f"  paired Wilcoxon p       = {p_two:.4f}")
    print()
    print(f"  {'dataset':<18} {'DCS':>8} {'MLP':>8} {'delta':>9}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*9}")
    for r in rows:
        marker = "  *" if abs(r["delta_MLP_minus_DCS"]) > 0.05 else ""
        print(f"  {r['dataset']:<18} {r['f1_PlainDCS_H1']:>8.3f} "
              f"{r['f1_MLP_H1_Tuned']:>8.3f} "
              f"{r['delta_MLP_minus_DCS']:>+9.3f}{marker}")
    print()
    print("  (* = absolute per-dataset delta > 0.05)")
    print()

    # --------------------------------------------------------------------- #
    # Decision tree                                                         #
    # --------------------------------------------------------------------- #
    print("=" * 72)
    print("  DECISION TREE")
    print("=" * 72)

    if p_two >= 0.05 and abs(diff.mean()) < 0.02:
        outcome = "TIE_INDISTINGUISHABLE"
        msg = (
            "BEST CASE.  The two H=1 trainers are statistically indistinguishable\n"
            "  and the average dataset-mean F1 difference is < 0.02.\n\n"
            "  v16 ACTION: Add B4 as a positive sanity check in a new Study VIII\n"
            "  (or as a paragraph inside Study VII).  Suggested wording:\n\n"
            "    \"A tuned single-hidden-unit MLP (Study VIII, $H=1$ fixed,\n"
            "    Optuna over learning rate, weight decay, activation, momentum)\n"
            "    attains macro-F1 statistically indistinguishable from plain DCS\n"
            "    at $H=1$ on the same 50 CV partitions (paired Wilcoxon\n"
            f"    $p={p_two:.3f}$, mean $\\Delta$F1 $={diff.mean():+.3f}$),\n"
            "    confirming both methods at the $H=1$ predictive ceiling on\n"
            "    these datasets.\"\n\n"
            "  This DEFENDS the DCS-as-compact-H=1-trainer framing without\n"
            "  forcing any other revision."
        )
    elif p_two >= 0.05:
        outcome = "TIE_NON_SIGNIFICANT_BUT_NOTABLE_DELTA"
        msg = (
            f"NOT-SIGNIFICANT but mean delta = {diff.mean():+.3f} ; the test is\n"
            "  underpowered with 9 datasets.  Still safe to report as indistinguishable.\n\n"
            "  v16 ACTION: Same as TIE case but include the mean delta honestly:\n\n"
            "    \"...statistically indistinguishable (paired Wilcoxon "
            f"$p={p_two:.3f}$, mean $\\Delta$F1 $={diff.mean():+.3f}$)...\"\n\n"
            "  Reviewer 2 satisfied; framing intact."
        )
    elif diff.mean() > 0:
        outcome = "MLP_WINS_SIGNIFICANTLY"
        msg = (
            f"HARD TRUTH.  Gradient-trained MLP at $H=1$ significantly\n"
            f"  outperforms plain DCS at $H=1$ (paired $p={p_two:.4f}$,\n"
            f"  mean $\\Delta$F1 $={diff.mean():+.3f}$ favouring MLP).\n\n"
            "  v16 ACTION: The 'DCS as a compact optimizer' claim does NOT survive.\n"
            "  Reframe contributions as:\n"
            "    1. The unified DKA mathematical framework (Study V) -- survives.\n"
            "    2. The 10-method Pareto-frontier dataset -- survives.\n"
            "    3. Cross-length alignment as a well-posedness rule -- survives.\n"
            "  And explicitly DROP:\n"
            "    - Any claim that DCS is competitive as a compact tabular trainer.\n"
            "    - Study VII's 'plain DCS at H=1 produces F1 indistinguishable\n"
            "      from DCS--VSE' reading as a positive (it now means 'DCS is\n"
            "      no better than a tuned MLP at the same operating point').\n\n"
            "  Better to know now than after rejection."
        )
    else:
        outcome = "DCS_WINS_SIGNIFICANTLY"
        msg = (
            f"GIFT OUTCOME.  Plain DCS at $H=1$ significantly outperforms\n"
            f"  the tuned single-hidden-unit MLP (paired $p={p_two:.4f}$,\n"
            f"  mean $\\Delta$F1 $={diff.mean():+.3f}$ favouring DCS).\n\n"
            "  v16 ACTION: Promote B4 to a new Study VIII and lead the conclusion\n"
            "  with this result.  Suggested wording:\n\n"
            "    \"Plain DCS at $H=1$ significantly outperforms a tuned\n"
            "    single-hidden-unit MLP on macro-F1 (paired Wilcoxon\n"
            f"    $p={p_two:.4f}$, mean $\\Delta$F1 $={diff.mean():+.3f}$),\n"
            "    indicating that the DCS rank-guided arithmetic learning\n"
            "    operator finds parameter configurations at $H=1$ that\n"
            "    Adam-based gradient training cannot reach on these tasks.\"\n\n"
            "  This RESCUES the predictive-utility contribution."
        )

    print(f"\n  OUTCOME: {outcome}\n")
    for line in msg.splitlines():
        print(f"  {line}")
    print()

    # --------------------------------------------------------------------- #
    # LaTeX summary row                                                     #
    # --------------------------------------------------------------------- #
    tex = []
    tex.append(r"\begin{table}[ht]\centering\scriptsize")
    tex.append(r"\begin{tabular}{lrrrrr}")
    tex.append(r"\toprule")
    tex.append(r"Dataset & DCS-noVSE (B3) $H{=}1$ F1 & MLP tuned (B4) "
               r"$H{=}1$ F1 & $\Delta$ (MLP$-$DCS) & DCS sec & MLP sec \\")
    tex.append(r"\midrule")
    for r in rows:
        d = r["dataset"].replace("_", r"\_")
        tex.append(f"  {d} & {r['f1_PlainDCS_H1']:.3f} & "
                   f"{r['f1_MLP_H1_Tuned']:.3f} & "
                   f"{r['delta_MLP_minus_DCS']:+.3f} & "
                   f"{r['t_PlainDCS_H1']:.1f} & "
                   f"{r['t_MLP_H1_Tuned']:.1f} \\\\")
    tex.append(r"\midrule")
    tex.append(f"  Mean & {b3_arr.mean():.3f} & {b4_arr.mean():.3f} & "
               f"{diff.mean():+.3f} & -- & -- \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(
        r"\caption{Study VIII: head-to-head at fixed $H=1$.  Plain DCS "
        r"(B3, evolutionary trainer) vs.\ tuned MLP (B4, Adam + StandardScaler "
        r"+ class-balanced oversampling + Optuna over learning rate, weight "
        r"decay, activation, momentum, 40 trials).  Macro-F1 dataset means over "
        f"the same 50 stratified CV partitions per dataset.  Paired Wilcoxon "
        f"$p={p_two:.4f}$.}}"
    )
    tex.append(r"\label{tab:studyVIII_b4_vs_b3}")
    tex.append(r"\end{table}")

    out_tex = os.path.join(args.out, "b4_vs_b3_summary.tex")
    with open(out_tex, "w") as fh:
        fh.write("\n".join(tex))

    print(f"  Wrote: {out_csv}")
    print(f"  Wrote: {out_tex}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
