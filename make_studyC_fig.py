import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal

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

df = pd.read_csv('results_H1to20/summary.csv').dropna(subset=['algo','dataset'])
sc = ['ADENOCARCINOMA','DLBCL','DYSLEXIA','DYSLEXIA_10p','ILPD',
      'LEUKEMIA1','LEUKEMIA2','PROSTATE6033','PROSTATE_TUMOR']
opt = df[df.algo.str.startswith('DCS_VSE_DKA_opt_')].copy()
opt['form'] = opt.algo.str.extract(r'opt_(\d+)_d')[0].astype(int)
sub = opt[opt.dataset.isin(sc)]
M = sub.pivot_table(index='dataset', columns='form', values='f1_mean')   # 9 x 14
forms = sorted(M.columns)
means = M.mean(axis=0)
order = means.sort_values().index.tolist()                              # ascending
H, p = kruskal(*[M[f].values for f in forms])

fig, ax = plt.subplots(figsize=(7.4, 5.8))
ypos = np.arange(len(order))
for yi, f in zip(ypos, order):
    is0 = (f == 0)
    color = '#d62728' if is0 else '#9aa0a6'
    # per-dataset points
    ax.scatter(M[f].values, np.full(9, yi), s=16, color=color, alpha=0.45,
               edgecolor='none', zorder=2)
    # mean marker
    ax.scatter(means[f], yi, marker='D', s=46, color=color,
               edgecolor='black', linewidth=0.6, zorder=3)
labels = [('opt %d (proposed)' % f) if f == 0 else ('opt %d' % f) for f in order]
ax.set_yticks(ypos); ax.set_yticklabels(labels)
for t, f in zip(ax.get_yticklabels(), order):
    if f == 0: t.set_color('#d62728'); t.set_fontweight('bold')
gmin = means.min(); gmax = means.max()
ax.axvline(means[0], color='#d62728', ls='--', lw=0.9, alpha=0.7, zorder=1)
ax.set_xlabel('Dataset-averaged macro-F1 (9 microarray/clinical datasets)')
ax.set_xlim(0.40, 0.92)
ax.set_ylim(-0.7, len(order) + 1.7)
ax.set_title('DKA design space: 14 formations are statistically indistinguishable', pad=10)
ax.annotate('Kruskal--Wallis $H=%.2f$, $p=1.00$ (n.s.)\nopt 0 mean = %.3f (highest); spread = %.3f'
            % (H, means[0], gmax - gmin),
            xy=(0.015, 0.985), xycoords='axes fraction', va='top', ha='left', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.35', fc='#f5f5f5', ec='#cccccc'))
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='D', color='w', markerfacecolor='#9aa0a6',
              markeredgecolor='black', markersize=8, label='formation mean'),
       Line2D([0],[0], marker='o', color='w', markerfacecolor='#9aa0a6',
              markersize=6, alpha=0.5, label='per-dataset F1')]
ax.legend(handles=leg, loc='lower right', frameon=False)
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig('../figs/studyC/dka_f1_comparison.%s' % ext)
print('saved; H=%.3f p=%.3f opt0_mean=%.4f min_form=%.4f max_form=%.4f'
      % (H, p, means[0], gmin, gmax))
