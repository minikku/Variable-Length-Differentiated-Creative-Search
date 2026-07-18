#!/usr/bin/env python3
"""Combine B1+B2+B3 outputs with the existing H1to20 results and emit the
Friedman / Wilcoxon tables in LaTeX form (drop straight into Section 5.3).

Usage
-----
    python b_combine_stats.py

By default it expects these four directories to exist as siblings of this file:
    results_H1to20/                  (existing, 7 SLNN-family methods)
    results_B1_baselines/            (new, LGBM/XGBoost/MLP under same CV)
    results_B2_aligned_vs_padded/    (new, CoDE/COLSHADE/DBA x {aligned,padded})
    results_B3_plain_dcs_fixed_H1/   (new, plain DCS at H=1)

It reads each directory's summary.csv, builds a combined per-(method, dataset)
table of dataset-level means, runs Friedman on F1 / AUC / ACC / hidden size
across the union of methods, runs Holm-corrected pairwise Wilcoxon vs.
DCS_VSE_DKA_opt_0_d, and writes:

    combined/combined_summary.csv
    combined/friedman_table.tex
    combined/wilcoxon_f1.tex
    combined/wilcoxon_auc.tex
    combined/wilcoxon_hidden.tex
    combined/wilcoxon_b2_aligned_vs_padded.tex
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path):
    if not os.path.exists(path):
        print(f"  ! missing {path}; skipping")
        return None
    return pd.read_csv(path)


def _holm(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    out = np.zeros(n)
    running = 0.0
    for k, i in enumerate(order):
        adj = (n - k) * pvals[i]
        running = max(running, adj)
        out[i] = min(1.0, running)
    return out


def _build_combined(src_dirs):
    parts = []
    for tag, path in src_dirs.items():
        df = _load(os.path.join(path, "summary.csv"))
        if df is None:
            continue
        df = df.copy()
        df["source"] = tag
        parts.append(df)
    if not parts:
        raise RuntimeError("no summary.csv found in any source directory")
    return pd.concat(parts, ignore_index=True)


def _friedman(table: pd.DataFrame, metric: str, higher_is_better: bool):
    methods = sorted(table["algo"].unique())
    datasets = sorted(table["dataset"].unique())
    M = np.full((len(datasets), len(methods)), np.nan)
    for j, m in enumerate(methods):
        for i, d in enumerate(datasets):
            row = table[(table["algo"] == m) & (table["dataset"] == d)]
            if not row.empty:
                M[i, j] = row[metric].iloc[0]
    # Datasets where every method has a value:
    keep = ~np.isnan(M).any(axis=1)
    if keep.sum() < 3:
        return None
    Mk = M[keep]
    if not higher_is_better:
        Rk = (Mk).argsort(axis=1).argsort(axis=1) + 1
    else:
        Rk = (-Mk).argsort(axis=1).argsort(axis=1) + 1
    ranks = Rk.mean(axis=0)
    if higher_is_better:
        stat, p = friedmanchisquare(*[Mk[:, j] for j in range(Mk.shape[1])])
    else:
        stat, p = friedmanchisquare(*[(-Mk)[:, j] for j in range(Mk.shape[1])])
    return {
        "methods": methods, "ranks": ranks, "chi2": stat, "p": p,
        "n_datasets": int(keep.sum()),
    }


def _pairwise_wilcoxon(table, metric, target, higher_is_better=True):
    methods = sorted(table["algo"].unique())
    datasets = sorted(table["dataset"].unique())
    M = {}
    for m in methods:
        M[m] = np.array([
            table[(table["algo"] == m) & (table["dataset"] == d)][metric].iloc[0]
            if not table[(table["algo"] == m) & (table["dataset"] == d)].empty
            else np.nan
            for d in datasets
        ])
    if target not in M:
        return None
    t = M[target]
    rows = []
    raw_p, contests = [], []
    for m in methods:
        if m == target:
            continue
        v = M[m]
        keep = ~np.isnan(t) & ~np.isnan(v)
        if keep.sum() < 5:
            continue
        diff = t[keep] - v[keep]
        try:
            stat, p = wilcoxon(t[keep], v[keep], alternative="two-sided",
                               zero_method="wilcox")
        except ValueError:
            stat, p = np.nan, 1.0
        sign = np.sign(diff)
        ranks_pos = np.argsort(np.argsort(np.abs(diff))) + 1
        w_plus = int(ranks_pos[diff > 0].sum())
        w_minus = int(ranks_pos[diff < 0].sum())
        median = float(np.median(diff))
        raw_p.append(p)
        contests.append({
            "baseline": m, "n": int(keep.sum()),
            "W+": w_plus, "W-": w_minus, "median_diff": median, "p": p,
        })
    if not contests:
        return None
    holm = _holm(np.array(raw_p))
    for r, h in zip(contests, holm):
        r["p_Holm"] = float(h)
    return contests


def _latex_friedman(metrics):
    cols = sorted({m for v in metrics.values() for m in v["methods"]})
    lines = []
    lines.append("\\begin{table*}[ht]")
    lines.append("\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l" + "r" * (len(cols) + 2) + "}")
    lines.append("\\toprule")
    header = "Metric & " + " & ".join(cols.replace("_", "\\_") if "_" in cols else cols
                                       for cols in cols) + " & $\\chi^2_F$ & $p$ \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for name, v in metrics.items():
        row = [name]
        ranks = dict(zip(v["methods"], v["ranks"]))
        best = min(v["ranks"])
        for c in cols:
            if c in ranks:
                r = ranks[c]
                cell = f"{r:.3f}"
                if abs(r - best) < 1e-9:
                    cell = "\\textbf{" + cell + "}"
                row.append(cell)
            else:
                row.append("--")
        row.append(f"{v['chi2']:.2f}")
        row.append(f"{v['p']:.2e}")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Combined Friedman test across all methods on the "
                 "nine datasets under the identical $5\\times 10$ stratified "
                 "CV pipeline. Lower rank is better.}")
    lines.append("\\label{tab:ranks_combined}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def _latex_wilcoxon(contests, metric, target):
    lines = []
    lines.append("\\begin{table*}[ht]")
    lines.append("\\centering\\scriptsize")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Baseline & $n$ & $W^{+}$ & $W^{-}$ & $p$ & $p_{\\mathrm{Holm}}$ \\\\")
    lines.append("\\midrule")
    for c in contests:
        sig = "\\textbf{" if c["p_Holm"] < 0.05 else ""
        end = "}" if c["p_Holm"] < 0.05 else ""
        name = c["baseline"].replace("_", "\\_")
        lines.append(
            f"{name} & {c['n']} & {c['W+']} & {c['W-']} & "
            f"{c['p']:.4f} & {sig}{c['p_Holm']:.4f}{end} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        f"\\caption{{Paired Wilcoxon signed-rank tests (two-sided) on "
        f"dataset-level mean {metric} comparing "
        f"{target.replace('_', chr(92)+'_')} against each baseline across the "
        f"nine datasets. Holm correction is applied across pairwise "
        f"comparisons; entries with $p_{{\\mathrm{{Holm}}}} < 0.05$ are bolded.}}"
    )
    lines.append(f"\\label{{tab:wil_{metric}_combined}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1to20", default=os.path.join(HERE, "results_H1to20"))
    ap.add_argument("--b1", default=os.path.join(HERE, "results_B1_baselines"))
    ap.add_argument("--b2", default=os.path.join(HERE, "results_B2_aligned_vs_padded"))
    ap.add_argument("--b3", default=os.path.join(HERE, "results_B3_plain_dcs_fixed_H1"))
    ap.add_argument("--target", default="DCS_VSE_DKA_opt_0_d")
    ap.add_argument("--out", default=os.path.join(HERE, "combined"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    combined = _build_combined({
        "H1to20": args.h1to20,
        "B1": args.b1,
        "B2": args.b2,
        "B3": args.b3,
    })
    combined.to_csv(os.path.join(args.out, "combined_summary.csv"), index=False)

    # Friedman on the union of methods (only datasets where every method
    # appears are kept by _friedman; this is the strict comparison set).
    metrics_table = {}
    for metric, hib in [("f1_mean", True), ("auc_mean", True),
                        ("acc_mean", True), ("hidden_mean", False)]:
        r = _friedman(combined, metric, hib)
        if r is not None:
            label = {"f1_mean": "F1", "auc_mean": "AUC",
                     "acc_mean": "ACC", "hidden_mean": "Hidden"}[metric]
            metrics_table[label] = r
    with open(os.path.join(args.out, "friedman_table.tex"), "w") as fh:
        fh.write(_latex_friedman(metrics_table))

    # Pairwise Wilcoxon vs. DCS_VSE_DKA_opt_0_d on each metric.
    for metric, fname, hib in [
        ("f1_mean", "wilcoxon_f1.tex", True),
        ("auc_mean", "wilcoxon_auc.tex", True),
        ("hidden_mean", "wilcoxon_hidden.tex", False),
    ]:
        contests = _pairwise_wilcoxon(combined, metric, args.target, hib)
        if contests:
            with open(os.path.join(args.out, fname), "w") as fh:
                fh.write(_latex_wilcoxon(contests, metric.replace("_mean", ""),
                                          args.target))

    # B2-specific within-algorithm Wilcoxon (aligned vs padded per algo).
    b2 = combined[combined["source"] == "B2"]
    if not b2.empty:
        rows = []
        for algo_root in ["CoDE_d", "COLSHADE_d", "DBA_d"]:
            a = f"{algo_root}:aligned"; p = f"{algo_root}:padded"
            sub_a = b2[b2["algo"] == a]
            sub_p = b2[b2["algo"] == p]
            ds = sorted(set(sub_a["dataset"]) & set(sub_p["dataset"]))
            if len(ds) < 5:
                continue
            f1a = np.array([sub_a[sub_a["dataset"] == d]["f1_mean"].iloc[0] for d in ds])
            f1p = np.array([sub_p[sub_p["dataset"] == d]["f1_mean"].iloc[0] for d in ds])
            try:
                stat, pv = wilcoxon(f1a, f1p, alternative="two-sided",
                                    zero_method="wilcox")
            except ValueError:
                pv = 1.0
            rows.append({
                "algo": algo_root, "n_datasets": len(ds),
                "mean_aligned": float(f1a.mean()),
                "mean_padded": float(f1p.mean()),
                "delta": float(f1a.mean() - f1p.mean()),
                "p_value": float(pv),
            })
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(args.out, "b2_aligned_vs_padded_summary.csv"),
                      index=False)
            lines = ["\\begin{table}[ht]\\centering\\scriptsize",
                     "\\begin{tabular}{lrrrrr}", "\\toprule",
                     "Algorithm & $n_{\\mathrm{ds}}$ & F1 aligned & F1 padded "
                     "& $\\Delta$ F1 & paired-$p$ \\\\", "\\midrule"]
            for r in rows:
                sig = "\\textbf{" if r["p_value"] < 0.05 else ""
                end = "}" if r["p_value"] < 0.05 else ""
                lines.append(
                    f"{r['algo'].replace('_','\\_')} & {r['n_datasets']} & "
                    f"{r['mean_aligned']:.3f} & {r['mean_padded']:.3f} & "
                    f"{r['delta']:+.3f} & {sig}{r['p_value']:.4f}{end} \\\\"
                )
            lines += ["\\bottomrule", "\\end{tabular}",
                      "\\caption{B2 counterfactual: paired Wilcoxon test of "
                      "macro-F1 between the aligned and max-embedding-padded "
                      "interaction rules, for each of CoDE, COLSHADE, DBA.}",
                      "\\label{tab:b2_counterfactual}", "\\end{table}"]
            with open(os.path.join(args.out, "wilcoxon_b2_aligned_vs_padded.tex"),
                      "w") as fh:
                fh.write("\n".join(lines))

    print("Wrote:")
    for f in sorted(os.listdir(args.out)):
        print("  ", os.path.join(args.out, f))


if __name__ == "__main__":
    raise SystemExit(main())
