#!/usr/bin/env python3
"""Render the two-panel Diagnostic Study G figure for Section 5.5.7.

Inputs:
    results_H1to20/curves/<algo>__LEUKEMIA{1,2}.npz    (already on disk)
    results_studyG_weights/weights.npz                  (from studyG_weight_distribution.py)

Outputs:
    figs/studyG/convergence_microarray.pdf
    figs/studyG/weight_distribution.pdf
    figs/studyG/studyG_diagnostic.pdf  (combined two-panel)
"""
from __future__ import annotations
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(os.path.dirname(HERE), "figs", "studyG")
os.makedirs(OUT, exist_ok=True)

METHODS = [
    ("GProp_d",            "G-Prop",            "#7f7f7f"),
    ("VLPSO_d",            "VLPSO",             "#9467bd"),
    ("CoDE_d",             "CoDE",              "#8c564b"),
    ("DBA_d",              "DBA",               "#e377c2"),
    ("COLSHADE_d",         "COLSHADE",          "#bcbd22"),
    ("DCS_noVSE_d",        "DCS-VSE (proposed)","#d62728"),
    ("DCS_VSE_DKA_opt_0_d","DCS-VSE+SO",        "#1f77b4"),
]


def _load_curves(algo, dataset, root):
    path = os.path.join(root, "results_H1to20", "curves", f"{algo}__{dataset}.npz")
    if not os.path.exists(path): return None
    z = np.load(path)
    arrs = [z[k] for k in sorted(z.keys())]
    L = min(a.shape[0] for a in arrs)
    return np.stack([a[:L] for a in arrs])


def panel_a_convergence(ax, dataset, root):
    """Panel (a): best-so-far fitness median + IQR across 50 folds."""
    for algo, lab, col in METHODS:
        C = _load_curves(algo, dataset, root)
        if C is None: continue
        nfe = np.linspace(0, 30_000, C.shape[1])
        med = np.median(C, axis=0)
        q25 = np.percentile(C, 25, axis=0)
        q75 = np.percentile(C, 75, axis=0)
        ax.plot(nfe, med, label=lab, color=col,
                linewidth=2.0 if algo == "DCS_noVSE_d" else 1.4,
                alpha=1.0 if algo == "DCS_noVSE_d" else 0.85)
        ax.fill_between(nfe, q25, q75, color=col,
                        alpha=0.18 if algo == "DCS_noVSE_d" else 0.10)
    ax.set_xlabel("Number of fitness evaluations (NFE)", fontsize=13)
    ax.set_ylabel("Best-so-far MCC-based fitness (lower is better)", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)


def panel_b_weight_dist(ax, dataset, root):
    """Panel (b): cumulative-importance (Lorenz) curves."""
    wpath = os.path.join(root, "results_studyG_weights", "weights.npz")
    if not os.path.exists(wpath):
        ax.text(0.5, 0.5,
                "Run studyG_weight_distribution.py\nfirst to produce weights.npz",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
        return
    w_data = np.load(wpath)
    keys = sorted(w_data.keys())
    # Filter to this dataset only
    ds_keys = [k for k in keys if k.startswith(f"{dataset}__")]
    if not ds_keys:
        ax.text(0.5, 0.5, f"No weight data for {dataset}",
                ha="center", va="center", transform=ax.transAxes)
        return

    color_map = {algo: col for algo, _, col in METHODS}
    color_map.update({"LGBM": "#ff7f0e", "XGBoost": "#2ca02c"})
    label_map = {algo: lab for algo, lab, _ in METHODS}
    label_map.update({"LGBM": "LGBM+Optuna", "XGBoost": "XGBoost+Optuna"})

    for k in ds_keys:
        algo_key = k.replace(f"{dataset}__", "")
        if algo_key not in label_map: continue
        w = w_data[k]
        mag = np.abs(w)
        if mag.sum() == 0:
            continue
        # Lorenz: rank weights smallest -> largest, then cumulative share
        sorted_mag = np.sort(mag)
        cum = np.cumsum(sorted_mag) / sorted_mag.sum()
        x = np.linspace(0, 1, len(cum))
        ax.plot(x, cum,
                color=color_map.get(algo_key, "0.3"),
                linewidth=2.0 if algo_key == "DCS_noVSE_d" else 1.4,
                label=label_map[algo_key],
                alpha=1.0 if algo_key == "DCS_noVSE_d" else 0.85)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5,
            label="uniform reference")
    ax.set_xlabel("Cumulative fraction of features (ranked by $|w|$)", fontsize=13)
    ax.set_ylabel(r"Cumulative share of $\sum_i |w_i|$", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.95)


def main():
    # Two-panel figure: convergence on LEUKEMIA1 + weight Lorenz on LEUKEMIA1
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    panel_a_convergence(axes[0], "LEUKEMIA1", os.path.dirname(HERE) + "/experiment_py")
    panel_b_weight_dist(axes[1], "LEUKEMIA1", os.path.dirname(HERE) + "/experiment_py")
    axes[0].set_title("(a) LEUKEMIA1: optimization trajectory ($n=72$, $d=5{,}327$)", fontsize=13)
    axes[1].set_title("(b) LEUKEMIA1: weight-magnitude concentration", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "studyG_diagnostic.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "studyG_diagnostic.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}/studyG_diagnostic.pdf")

    # Also LEUKEMIA2 standalone (for appendix)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    panel_a_convergence(axes[0], "LEUKEMIA2", os.path.dirname(HERE) + "/experiment_py")
    panel_b_weight_dist(axes[1], "LEUKEMIA2", os.path.dirname(HERE) + "/experiment_py")
    axes[0].set_title("(a) LEUKEMIA2: optimization trajectory ($n=72$, $d=11{,}225$)", fontsize=13)
    axes[1].set_title("(b) LEUKEMIA2: weight-magnitude concentration", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "studyG_diagnostic_LEUK2.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "studyG_diagnostic_LEUK2.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}/studyG_diagnostic_LEUK2.pdf")


if __name__ == "__main__":
    main()
