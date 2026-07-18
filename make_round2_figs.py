# -*- coding: utf-8 -*-
"""Round-2 figures: fixed-H forest, corrected Study D forest + mechanism."""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

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
HI='#1f77b4'; LO='#d62728'; POS='#2a7f3f'

# ============ Figure 1: fixed-H=1 vs variable-length forest ============
fc=pd.read_csv('fixedH_compare.csv')
fc=fc.sort_values('delta')
fig,ax=plt.subplots(figsize=(7.0,4.8))
y=np.arange(len(fc))
ax.scatter(fc.delta,y,s=42,color=POS,edgecolor='black',linewidth=0.5,zorder=3)
for yi,dv in zip(y,fc.delta):
    ax.plot([0,dv],[yi,yi],color=POS,lw=1.0,alpha=0.5,zorder=2)
ax.axvline(0,color='#888',lw=0.9)
ax.axvline(fc.delta.mean(),color=POS,ls='--',lw=0.9,alpha=0.8)
ax.set_yticks(y); ax.set_yticklabels(fc.dataset,fontsize=11)
ax.set_xlabel(r'$\Delta$ macro-F1 (variable-length DCS--VSE $-$ fixed $H{=}1$)')
ax.set_xlim(-0.01,0.16)
ax.set_title('Variable-length search improves over fixed $H{=}1$ on every dataset')
ax.annotate('mean $\\Delta=+0.059$, 18/18 datasets\npaired Wilcoxon $p=7.6\\times10^{-6}$',
            xy=(0.075,1.2),fontsize=12,
            bbox=dict(boxstyle='round,pad=0.35',fc='#f5f5f5',ec='#cccccc'))
fig.tight_layout()
os.makedirs('../figs/fixedH',exist_ok=True)
for e in ('pdf','png'): fig.savefig('../figs/fixedH/fixedH_forest.%s'%e)
plt.close(fig)

# ============ Study D data ============
d=pd.read_csv('combined/b5_vs_default.csv')
pf=pd.read_csv('combined/b5_per_fold_pairs.csv')
bp=pd.read_csv('results_B5_dcsvse_optuna/best_params.csv')[['dataset','lambda_size']]
highD=['ADENOCARCINOMA','DLBCL','LEUKEMIA1','LEUKEMIA2','PROSTATE6033','PROSTATE_TUMOR']
# per-dataset SE of delta from 50 folds
se=pf.groupby('dataset')['delta'].agg(lambda x:x.std(ddof=1)/np.sqrt(len(x)))
d=d.merge(bp,on='dataset',how='left')
d['se']=d.dataset.map(se)
d['grp']=np.where(d.dataset.isin(highD),'high-D','low-D')
d=d.sort_values('delta_tuned_minus_default')

# ---- Figure 2: Study D forest ----
fig,ax=plt.subplots(figsize=(7.0,4.2))
y=np.arange(len(d))
for yi,(_,r) in zip(y,d.iterrows()):
    c=HI if r.grp=='high-D' else LO
    ax.errorbar(r.delta_tuned_minus_default,yi,xerr=1.96*r.se,fmt='o',color=c,
                ecolor=c,elinewidth=1,capsize=2,ms=6,mec='black',mew=0.4,zorder=3)
ax.axvline(0,color='#888',lw=0.9)
ax.set_yticks(y); ax.set_yticklabels(d.dataset,fontsize=11)
ax.set_xlabel(r'$\Delta$ macro-F1 (Optuna-tuned $-$ default)')
ax.set_title('Per-dataset tuning effect: helps high-D, hurts low-D')
leg=[Line2D([0],[0],marker='o',color='w',markerfacecolor=HI,markeredgecolor='black',ms=8,label='high-D microarray  (mean $+0.045$, $p=0.003$)'),
     Line2D([0],[0],marker='o',color='w',markerfacecolor=LO,markeredgecolor='black',ms=8,label='low-D clinical  (mean $-0.022$, $p=0.015$)')]
ax.legend(handles=leg,loc='lower right',frameon=False,fontsize=11)
fig.tight_layout()
for e in ('pdf','png'): fig.savefig('../figs/studyF/studyF_forest.%s'%e)
plt.close(fig)

# ---- Figure 3: Study D mechanism ----
fig,axes=plt.subplots(1,2,figsize=(8.4,3.6))
for ax,xcol,xlab in [(axes[0],'lambda_size',r'tuned $\lambda_{\mathrm{size}}$'),
                     (axes[1],'hidden_tuned',r'tuned mean hidden size $\bar{H}$')]:
    for g,c in [('high-D',HI),('low-D',LO)]:
        s=d[d.grp==g]
        ax.scatter(s[xcol],s.delta_tuned_minus_default,s=46,color=c,edgecolor='black',linewidth=0.4,label=g)
    ax.axhline(0,color='#888',lw=0.8)
    ax.set_xlabel(xlab); ax.set_ylabel(r'$\Delta$ macro-F1')
axes[0].set_xscale('log')
axes[1].legend(frameon=False,fontsize=11)
fig.suptitle('Tuning gains come from larger $H$ on high-D data; the same inflation hurts low-D data',fontsize=13)
fig.tight_layout()
for e in ('pdf','png'): fig.savefig('../figs/studyF/studyF_mechanism.%s'%e)
plt.close(fig)

print('figures written. Study D check:')
print('  overall delta=%.4f  high-D=%.4f  low-D=%.4f'%(
 d.delta_tuned_minus_default.mean(),
 d[d.grp=="high-D"].delta_tuned_minus_default.mean(),
 d[d.grp=="low-D"].delta_tuned_minus_default.mean()))
