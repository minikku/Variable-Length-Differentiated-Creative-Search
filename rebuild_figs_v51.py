"""Rebuild Figures 4 (Pareto), 5 (runtime bars), 6 (B6 forest) from latest data."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Top-tier journal style
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

# Color palette
METHOD_COLORS = {
    'G-Prop':       '#7f7f7f',
    'VLPSO':        '#9467bd',
    'CoDE':         '#8c564b',
    'DBA':          '#e377c2',
    'COLSHADE':     '#bcbd22',
    'DCS-VSE':      '#d62728',
    'DCS-VSE+SO':   '#1f77b4',
    'XGBoost_m':    '#2ca02c',
    'LGBM_m':       '#ff7f0e',
    'XGBoost_full': '#2ca02c',
    'LGBM_full':    '#ff7f0e',
}

def load():
    h1 = pd.read_csv(os.path.join(HERE, 'results_H1to20/summary.csv')).dropna(subset=['algo','dataset'])
    b8 = pd.read_csv(os.path.join(HERE, 'results_B8_matched_complexity/summary.csv')).dropna(subset=['algo','dataset'])
    b1 = pd.read_csv(os.path.join(HERE, 'results_B1_baselines/summary.csv')).dropna(subset=['algo','dataset'])
    b6 = pd.read_csv(os.path.join(HERE, 'results_B6_dcsvse_aligned_vs_padded/results.csv'))
    return h1, b8, b1, b6


# ============================================================
# FIGURE 4: Pareto frontier (Macro-F1 vs architectural complexity)
# ============================================================
def fig4_pareto():
    h1, b8, b1, _ = load()
    ext = pd.concat([
        pd.read_csv(os.path.join(HERE, 'results_B13_extended/summary.csv')),
        pd.read_csv(os.path.join(HERE, 'results_B13_extended2/summary.csv'))],
        ignore_index=True).dropna(subset=['algo', 'dataset'])
    b10 = pd.read_csv(os.path.join(HERE, 'results_B10_logreg_matched/summary.csv')).dropna(subset=['algo', 'dataset'])

    def stat(df, algo, xcol):
        sub = df[df['algo'] == algo]
        return (sub[xcol].mean(), sub['f1_mean'].mean(),
                sub['f1_mean'].std(ddof=1) / np.sqrt(len(sub)))

    fig, ax = plt.subplots(figsize=(7.7, 4.8))
    ax.axvspan(0.9, 5.0, color='salmon', alpha=0.08, zorder=0)
    ax.text(1.03, 0.858, 'Compact-SLNN corner', fontsize=11, color='#a04040', ha='left', va='top')

    # Twelve variable-length competitors as a labeled cloud
    comp = [('GProp_d', h1), ('VLPSO_d', h1), ('CoDE_d', h1), ('DBA_d', h1), ('COLSHADE_d', h1),
            ('HOA_d', ext), ('SSLO_d', ext), ('ALA_d', ext), ('MSO_d', ext), ('EISA_d', ext), ('FGO_d', ext)]
    for i, (algo, df) in enumerate(comp):
        x, y, se = stat(df, algo, 'hidden_mean')
        ax.errorbar(x, y, yerr=1.96 * se, fmt='none', ecolor='#9aa5b1', alpha=0.35,
                    capsize=2, elinewidth=0.7, zorder=2)
        ax.scatter([x], [y], marker='o', s=46, facecolor='#c7ced6', edgecolors='#4b5157',
                   linewidth=0.7, zorder=3, label='Evolutionary competitors (12)' if i == 0 else None)
    # +SO ablation
    x, y, se = stat(h1, 'DCS_VSE_DKA_opt_0_d', 'hidden_mean')
    ax.errorbar(x, y, yerr=1.96 * se, fmt='none', ecolor='#1f77b4', alpha=0.55, capsize=3, elinewidth=0.8, zorder=2)
    ax.scatter([x], [y], marker='D', s=95, color='#1f77b4', edgecolors='#0f3f66', linewidth=0.8, zorder=4, label='DCS--VSE+SO')
    # DCS-VSE (proposed)
    dx, dy, dse = stat(h1, 'DCS_noVSE_d', 'hidden_mean')
    ax.errorbar(dx, dy, yerr=1.96 * dse, fmt='none', ecolor='#d62728', alpha=0.6, capsize=3, elinewidth=0.9, zorder=4)
    ax.scatter([dx], [dy], marker='*', s=260, color='#d62728', edgecolors='black', linewidth=1.4, zorder=6, label='DCS--VSE (proposed)')
    # Footprint-matched deterministic references on the compact axis
    lx, ly, lse = stat(b10, 'LogReg', 'f1_mean')
    ax.scatter([1.0], [b10[b10['algo'] == 'LogReg']['f1_mean'].mean()], marker='P', s=150,
               color='#17a2b8', edgecolors='black', linewidth=1.0, zorder=5, label='Logistic regression')
    ax.scatter([2.0], [b10[b10['algo'] == 'Stump']['f1_mean'].mean()], marker='X', s=110,
               color='#8B4513', edgecolors='black', linewidth=1.0, zorder=5, label='Decision stump')
    # Matched-leaves trees
    for algo, label, color in [('XGBoost_matched', r'XGBoost$_{\leq 20}$', '#2ca02c'),
                               ('LGBM_matched', r'LGBM$_{\leq 20}$', '#ff7f0e')]:
        x, y, se = stat(b8, algo, 'leaves_mean')
        ax.errorbar(x, y, yerr=1.96 * se, fmt='none', ecolor=color, alpha=0.55, capsize=3, elinewidth=0.8, zorder=2)
        ax.scatter([x], [y], marker='s', s=95, color=color, edgecolors=color, linewidth=0.8, zorder=3, label=label)
    # Unconstrained trees
    for algo, label, color in [('XGBoost', 'XGBoost+Optuna', '#2ca02c'),
                               ('LGBM', 'LGBM+Optuna', '#ff7f0e')]:
        x, y, se = stat(b1, algo, 'hidden_mean')
        ax.errorbar(x, y, yerr=1.96 * se, fmt='none', ecolor=color, alpha=0.55, capsize=3, elinewidth=0.8, zorder=2)
        ax.scatter([x], [y], marker='s', s=110, color=color, edgecolors='black', linewidth=1.0, zorder=3, label=label)
    ax.annotate('Unconstrained trees\n($\\sim 10^3$ leaves, $n_{\\mathrm{ds}}=9$)',
                xy=(3500, 0.845), xytext=(70, 0.865), fontsize=11, color='#444', ha='left',
                arrowprops=dict(arrowstyle='->', color='#666', lw=0.7))
    ax.annotate('DCS--VSE (proposed)\nlowest complexity, median $\\bar{H}\\approx 1$',
                xy=(dx * 1.05, dy), xytext=(3.0, 0.585), fontsize=11, color='#a04040', ha='left',
                arrowprops=dict(arrowstyle='->', color='#a04040', lw=0.7))
    # TabPFN accuracy ceiling
    _tf = pd.read_csv(os.path.join(HERE, 'results_B9_tabpfn/summary.csv')).dropna(subset=['dataset'])['f1_mean'].mean()
    ax.axhline(_tf, color='#6a3d9a', ls=(0, (6, 3)), lw=1.3, alpha=0.9, zorder=1)
    ax.text(1.0, _tf + 0.004, 'TabPFN ceiling (%.3f, foundation model)' % _tf,
            color='#6a3d9a', fontsize=11, va='bottom', ha='left')
    ax.set_xscale('log')
    ax.set_xlim(0.9, 1.2e4)
    ax.set_ylim(0.55, 0.90)
    ax.set_xlabel('Architectural complexity (SLNN hidden units / tree leaves, log scale)')
    ax.set_ylabel(r'Mean macro-F1 (error bar = $\pm 1.96\,\mathrm{SE}/\sqrt{n_{\mathrm{ds}}}$)')
    ax.grid(True, which='both', linestyle=':', alpha=0.35, zorder=0)
    # Legend placed outside the axes (right) so it never obstructs the data
    leg = ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                    frameon=True, fancybox=False, edgecolor='#888',
                    handletextpad=0.4, borderpad=0.5, labelspacing=0.45,
                    fontsize=10.5, title='Method', title_fontsize=11)
    leg.get_frame().set_linewidth(0.5)
    out = os.path.join(ROOT, 'figs/v20/pareto_F1_vs_complexity.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved {out}")
    return None


# ============================================================
# FIGURE 5: Per-fold wall-clock bars
# ============================================================
def fig5_runtime():
    h1, b8, _, _ = load()
    SLNN = [
        ('GProp_d', 'G-Prop'),
        ('VLPSO_d', 'VLPSO'),
        ('CoDE_d', 'CoDE'),
        ('DBA_d', 'DBA'),
        ('COLSHADE_d', 'COLSHADE'),
        ('DCS_noVSE_d', 'DCS-VSE (proposed)'),
        ('DCS_VSE_DKA_opt_0_d', 'DCS-VSE+SO'),
    ]
    items = []
    for algo, label in SLNN:
        sub = h1[h1['algo'] == algo]
        if not len(sub): continue
        m = sub['runtime_mean'].mean()
        med = sub['runtime_mean'].median()
        items.append((label, m, med, METHOD_COLORS.get(label.split(' ')[0], '#666')))
    # Override DCS-VSE color and label-color lookup
    items = [(lab, m, med, METHOD_COLORS['DCS-VSE'] if 'DCS-VSE (proposed)' == lab
              else (METHOD_COLORS['DCS-VSE+SO'] if lab == 'DCS-VSE+SO' else c))
             for lab, m, med, c in items]
    for algo, label, color_key in [
        ('XGBoost_matched', r'XGBoost$_{\leq 20}$', 'XGBoost_m'),
        ('LGBM_matched',    r'LGBM$_{\leq 20}$',    'LGBM_m'),
    ]:
        sub = b8[b8['algo'] == algo]
        if not len(sub): continue
        items.append((label, sub['runtime_mean'].mean(), sub['runtime_mean'].median(), METHOD_COLORS[color_key]))
    _ext = __import__('pandas').concat([
        __import__('pandas').read_csv(os.path.join(ROOT,'experiment_py','results_B13_extended','summary.csv')),
        __import__('pandas').read_csv(os.path.join(ROOT,'experiment_py','results_B13_extended2','summary.csv'))],
        ignore_index=True)
    for algo,label in [('HOA_d','HOA'),('SSLO_d','SSLO'),('ALA_d','ALA'),('MSO_d','MSO'),('EISA_d','EISA'),('FGO_d','FGO')]:
        sub=_ext[_ext['algo']==algo]
        if not len(sub): continue
        items.append((label, sub['runtime_mean'].mean(), sub['runtime_mean'].median(), '#8c6bb1'))
    # Sort by mean ascending
    items = sorted(items, key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ypos = np.arange(len(items))
    means = [r[1] for r in items]
    medians = [r[2] for r in items]
    labels = [r[0] for r in items]
    colors = [r[3] for r in items]

    bars = ax.barh(ypos, means, facecolor='white', edgecolor='black', linewidth=0.8, hatch='////')
    # Overlay median ticks
    for y, med in zip(ypos, medians):
        ax.plot([med, med], [y - 0.36, y + 0.36], color='black', lw=1.4, solid_capstyle='butt', zorder=3)
    # Value labels at bar end
    for y, m, med in zip(ypos, means, medians):
        ax.text(m + max(means) * 0.012, y, f'mean {m:.0f} s  (median {med:.1f} s)',
                va='center', fontsize=12, color='black')
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(means) * 1.35)
    ax.set_xlabel('Wall-clock per CV fold (seconds, mean across 50 stratified partitions)')
    # Bold the proposed method
    for i, lab in enumerate(labels):
        if 'DCS-VSE (proposed)' in lab:
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_color(METHOD_COLORS['DCS-VSE'])
    ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.35)
    ax.set_axisbelow(True)
    # Legend explaining the tick
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.4, label='median (per-fold tick)'),
        Patch(facecolor='white', edgecolor='black', hatch='////', label='mean across datasets (bar)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, edgecolor='#888',
              handletextpad=0.5, borderpad=0.4)
    out = os.path.join(ROOT, 'figs/v20/runtime_bars.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved {out}")
    return items


# ============================================================
# FIGURE 6: Study B (B6) per-dataset Δ macro-F1, aligned - padded
# ============================================================
def fig6_b6_forest():
    _, _, _, b6 = load()
    HI_D = ['LEUKEMIA1', 'DLBCL', 'ADENOCARCINOMA', 'PROSTATE6033', 'LEUKEMIA2', 'PROSTATE_TUMOR']
    LO_D = ['DYSLEXIA_10p', 'DYSLEXIA', 'ILPD']

    rows = []
    for ds in HI_D + LO_D:
        al = b6[(b6['algo'].str.endswith('aligned')) & (b6['dataset'] == ds)]['f1_macro'].to_numpy()
        pd_ = b6[(b6['algo'].str.endswith('padded')) & (b6['dataset'] == ds)]['f1_macro'].to_numpy()
        # Pair by (repeat, fold)
        al_df = b6[(b6['algo'].str.endswith('aligned')) & (b6['dataset'] == ds)][['repeat','fold','f1_macro']].rename(columns={'f1_macro':'al'})
        pd_df = b6[(b6['algo'].str.endswith('padded')) & (b6['dataset'] == ds)][['repeat','fold','f1_macro']].rename(columns={'f1_macro':'pd'})
        merged = al_df.merge(pd_df, on=['repeat','fold'])
        d = merged['al'].to_numpy() - merged['pd'].to_numpy()
        mean_d = float(np.mean(d))
        se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
        rows.append((ds, mean_d, se, len(d), 'High-D' if ds in HI_D else 'Low-D'))
    # Subgroup means
    hi_dts = [r[1] for r in rows if r[-1] == 'High-D']
    lo_dts = [r[1] for r in rows if r[-1] == 'Low-D']
    from scipy.stats import wilcoxon
    # Per-fold pooled subgroup test
    hi_pooled = np.concatenate([
        (b6[(b6['algo'].str.endswith('aligned')) & (b6['dataset'] == ds)].sort_values(['repeat','fold'])['f1_macro'].to_numpy() -
         b6[(b6['algo'].str.endswith('padded')) & (b6['dataset'] == ds)].sort_values(['repeat','fold'])['f1_macro'].to_numpy())
        for ds in HI_D])
    lo_pooled = np.concatenate([
        (b6[(b6['algo'].str.endswith('aligned')) & (b6['dataset'] == ds)].sort_values(['repeat','fold'])['f1_macro'].to_numpy() -
         b6[(b6['algo'].str.endswith('padded')) & (b6['dataset'] == ds)].sort_values(['repeat','fold'])['f1_macro'].to_numpy())
        for ds in LO_D])
    try:
        _, hi_p = wilcoxon(hi_pooled, alternative='two-sided', zero_method='wilcox')
    except: hi_p = np.nan
    try:
        _, lo_p = wilcoxon(lo_pooled, alternative='two-sided', zero_method='wilcox')
    except: lo_p = np.nan

    from matplotlib.lines import Line2D as _L2D
    fig, ax = plt.subplots(figsize=(8.9, 5.0))
    ypos = np.arange(len(rows))[::-1]  # first appears on top
    hi_y = [y for y, r in zip(ypos, rows) if r[-1] == 'High-D']
    lo_y = [y for y, r in zip(ypos, rows) if r[-1] == 'Low-D']
    # subtle grayscale band + divider to separate the two groups (no color)
    ax.axhspan(min(hi_y) - 0.5, max(hi_y) + 0.5, color='0.93', zorder=0)
    ax.axhline((min(hi_y) + max(lo_y)) / 2.0, color='0.55', lw=0.8, zorder=1)
    ax.axvline(0, color='black', lw=1.0, linestyle=(0, (5, 4)), alpha=0.7, zorder=1)
    # markers: group encoded by shape and fill, all black (grayscale-safe)
    for y, (ds, m, se, n, grp) in zip(ypos, rows):
        mk = 'o' if grp == 'High-D' else 's'
        fc = '0.25' if grp == 'High-D' else 'white'
        ax.errorbar([m], [y], xerr=1.96 * se, fmt=mk, ecolor='black',
                    markerfacecolor=fc, markeredgecolor='black', markersize=7.5,
                    capsize=4, elinewidth=1.3, mew=1.1, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(r'$\Delta$ macro-F1 (aligned $-$ padded), $\pm 1.96\,\mathrm{SE}$')
    ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    # group-summary boxes placed OUTSIDE the axes, on the right, aligned to each group
    hi_c = (min(hi_y) + max(hi_y)) / 2.0
    lo_c = (min(lo_y) + max(lo_y)) / 2.0
    ax.annotate('High-D microarray, filled circle (n=6)\nmean $\\Delta$=%+.4f\nper-fold $p=%.3f$' % (np.mean(hi_dts), hi_p),
                xy=(1.03, hi_c), xycoords=('axes fraction', 'data'), ha='left', va='center',
                fontsize=11, color='black', annotation_clip=False,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.3', linewidth=0.9))
    ax.annotate('Low-D clinical, open square (n=3)\nmean $\\Delta$=%+.4f\nper-fold $p=%.3f$' % (np.mean(lo_dts), lo_p),
                xy=(1.03, lo_c), xycoords=('axes fraction', 'data'), ha='left', va='center',
                fontsize=11, color='black', annotation_clip=False,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.3', linewidth=0.9))
    out = os.path.join(ROOT, 'figs/studyB/b6_forest.pdf')
    fig.savefig(out, dpi=400, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")
    return rows, hi_p, lo_p


# ===============================================

def _short(ds):
    return {
        'ADENOCARCINOMA': 'ADENO',
        'DLBCL': 'DLBCL',
        'LEUKEMIA1': 'LEUK1',
        'LEUKEMIA2': 'LEUK2',
        'PROSTATE6033': 'PROS6k',
        'PROSTATE_TUMOR': 'PROST',
        'DYSLEXIA': 'DYSL',
        'DYSLEXIA_10p': 'DYSL-10p',
        'ILPD': 'ILPD',
    }.get(ds, ds)


def fig8_studyF_forest():
    h1, _, _, _ = load()
    tuned = pd.read_csv(os.path.join(HERE, 'results_B5b_dcsvse_noso_optuna/summary.csv'))
    raw_def = pd.read_csv(os.path.join(HERE, 'results_H1to20/results.csv')) if os.path.exists(os.path.join(HERE,'results_H1to20/results.csv')) else None
    raw_tu = pd.read_csv(os.path.join(HERE, 'results_B5b_dcsvse_noso_optuna/results.csv'))
    HI = ['ADENOCARCINOMA','DLBCL','LEUKEMIA1','LEUKEMIA2','PROSTATE6033','PROSTATE_TUMOR']
    LO = ['DYSLEXIA','DYSLEXIA_10p','ILPD']
    rows = []
    for ds in HI + LO:
        d_def = h1[(h1['algo']=='DCS_noVSE_d') & (h1['dataset']==ds)]
        d_tu = tuned[tuned['dataset']==ds]
        if not len(d_def) or not len(d_tu): continue
        delta = float(d_tu['f1_mean'].iloc[0]) - float(d_def['f1_mean'].iloc[0])
        if raw_def is not None:
            rd = raw_def[(raw_def['algo']=='DCS_noVSE_d') & (raw_def['dataset']==ds)][['repeat','fold','f1_macro']].rename(columns={'f1_macro':'def_'})
            rt = raw_tu[(raw_tu['algo']=='DCS_VSE_NoSO_OptunaTuned') & (raw_tu['dataset']==ds)][['repeat','fold','f1_macro']].rename(columns={'f1_macro':'tu'})
            m = rd.merge(rt, on=['repeat','fold'])
            d_arr = m['tu'].to_numpy() - m['def_'].to_numpy()
            se = float(np.std(d_arr, ddof=1) / np.sqrt(len(d_arr))) if len(d_arr) > 1 else 0.01
        else:
            se = 0.01
        rows.append((ds, delta, se, 'High-D' if ds in HI else 'Low-D'))
    rows.sort(key=lambda r: (0 if r[-1]=='High-D' else 1, -r[1]))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ypos = np.arange(len(rows))[::-1]
    for y, (ds, d, se, grp) in zip(ypos, rows):
        color = '#3a6ea5' if grp == 'High-D' else '#a04040'
        ax.errorbar([d], [y], xerr=1.96 * se, fmt='o', color=color, markersize=6,
                    capsize=4, elinewidth=1.2, mec='black', mew=0.6)
    ax.axvline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
    hi_y = [y for y, r in zip(ypos, rows) if r[-1]=='High-D']
    lo_y = [y for y, r in zip(ypos, rows) if r[-1]=='Low-D']
    if hi_y:
        ax.axhspan(min(hi_y)-0.5, max(hi_y)+0.5, color='#3a6ea5', alpha=0.05, zorder=0)
    if lo_y:
        ax.axhspan(min(lo_y)-0.5, max(lo_y)+0.5, color='#a04040', alpha=0.05, zorder=0)
    hi_mean = np.mean([r[1] for r in rows if r[-1]=='High-D']) if hi_y else 0
    lo_mean = np.mean([r[1] for r in rows if r[-1]=='Low-D']) if lo_y else 0
    if hi_y:
        ax.text(0.115, max(hi_y), f'High-D microarray (n=6)\nmean $\\Delta$={hi_mean:+.4f}\nper-fold $p=0.0014$',
                fontsize=12, va='top', ha='right', color='#3a6ea5',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#3a6ea5', alpha=0.85))
    if lo_y:
        ax.text(0.115, max(lo_y), f'Low-D clinical (n=3)\nmean $\\Delta$={lo_mean:+.4f}\nper-fold $p=0.083$',
                fontsize=12, va='top', ha='right', color='#a04040',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#a04040', alpha=0.85))
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.10, 0.13)
    ax.set_xlabel(r'$\Delta$ macro-F1 (Optuna-tuned $-$ default), $\pm 1.96\,\mathrm{SE}$')
    ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.35)
    ax.set_axisbelow(True)
    out = os.path.join(ROOT, 'figs/studyF/studyF_forest.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved {out}")
    return rows


def fig9_studyF_mechanism():
    h1, _, _, _ = load()
    tuned = pd.read_csv(os.path.join(HERE, 'results_B5b_dcsvse_noso_optuna/summary.csv'))
    bp = pd.read_csv(os.path.join(HERE, 'results_B5b_dcsvse_noso_optuna/best_params.csv'))
    HI = ['ADENOCARCINOMA','DLBCL','LEUKEMIA1','LEUKEMIA2','PROSTATE6033','PROSTATE_TUMOR']
    LO = ['DYSLEXIA','DYSLEXIA_10p','ILPD']
    pts = []
    for ds in HI + LO:
        d_def = h1[(h1['algo']=='DCS_noVSE_d') & (h1['dataset']==ds)]
        d_tu = tuned[tuned['dataset']==ds]
        b = bp[bp['dataset']==ds]
        if not len(d_def) or not len(d_tu) or not len(b): continue
        delta = float(d_tu['f1_mean'].iloc[0]) - float(d_def['f1_mean'].iloc[0])
        lam = float(b['lambda_size'].iloc[0])
        Hbar = float(d_tu['hidden_mean'].iloc[0])
        pts.append((ds, delta, lam, Hbar, 'High-D' if ds in HI else 'Low-D'))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 4.4))
    from matplotlib.lines import Line2D
    for ax, xkey, xlabel, logx, vline, vlabel in [
        (axL, 'lam',  r'Tuned $\lambda_{\mathrm{size}}$ (log scale)', True, 0.05, r'default $\lambda_{\mathrm{size}}=0.05$'),
        (axR, 'Hbar', r'Tuned mean hidden size $\bar{H}$',           False, 1.0, r'default $\bar{H}=1.0$'),
    ]:
        for ds, d, lam, Hbar, grp in pts:
            color = '#3a6ea5' if grp == 'High-D' else '#a04040'
            x = lam if xkey == 'lam' else Hbar
            ax.scatter([x], [d], color=color, s=70, edgecolors='black', linewidth=0.6, zorder=3)
            ax.annotate(_short(ds), (x, d), xytext=(6, 4), textcoords='offset points',
                        fontsize=11, color='#333')
        ax.axhline(0, color='black', lw=0.6, linestyle=':', alpha=0.6)
        ax.axvline(vline, color='gray', lw=0.6, linestyle='--', alpha=0.7, label=vlabel)
        if logx: ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.08, 0.13)
        ax.grid(True, which='major', linestyle=':', alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=11, frameon=True, edgecolor='#888')
    axL.set_ylabel(r'$\Delta$ macro-F1 (tuned $-$ default)')
    sg_legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#3a6ea5', markeredgecolor='black', markersize=8, label='High-D microarray'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#a04040', markeredgecolor='black', markersize=8, label='Low-D clinical'),
    ]
    axL.legend(handles=sg_legend, loc='upper left', fontsize=11, frameon=True, edgecolor='#888')
    axL.set_title('(a) Size-penalty selection vs. outcome', fontsize=13)
    axR.set_title('(b) Network-size inflation vs. outcome', fontsize=13)
    plt.tight_layout()
    out = os.path.join(ROOT, 'figs/studyF/studyF_mechanism.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved {out}")
    return pts


if __name__ == '__main__':
    print("\n=== Figure 4 ===")
    fig4_pareto()
    print("\n=== Figure 5 ===")
    fig5_runtime()
    print("\n=== Figure 6 ===")
    fig6_b6_forest()
    print("\nAll figures rebuilt.")
