import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
    'mathtext.fontset':'stix','font.size':13,'axes.labelsize':14,'axes.titlesize':14,
    'legend.fontsize':12,'xtick.labelsize':12,'ytick.labelsize':12,'figure.dpi':150,'savefig.dpi':300,
    'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COL={'G-Prop':'#7f7f7f','VLPSO':'#9467bd','CoDE':'#8c564b','DBA':'#e377c2','COLSHADE':'#bcbd22',
     'DCS-VSE':'#d62728','DCS-VSE+SO':'#1f77b4','XGBoost_m':'#2ca02c','LGBM_m':'#ff7f0e',
     'LogReg':'#17a2b8','Stump':'#8B4513','TabPFN':'#6a3d9a'}
h1=pd.read_csv(os.path.join(HERE,'results_H1to20/summary.csv')).dropna(subset=['algo','dataset'])
b8=pd.read_csv(os.path.join(HERE,'results_B8_matched_complexity/summary.csv')).dropna(subset=['algo','dataset'])
b9=pd.read_csv(os.path.join(HERE,'results_B9_tabpfn/summary.csv')).dropna(subset=['dataset'])
b10=pd.read_csv(os.path.join(HERE,'results_B10_logreg_matched/summary.csv')).dropna(subset=['algo','dataset'])
def S(df,a,c): return df[df['algo']==a][c]

# ---------- PARETO ----------
def pareto():
    SLNN=[('GProp_d','G-Prop'),('VLPSO_d','VLPSO'),('CoDE_d','CoDE'),('DBA_d','DBA'),
          ('COLSHADE_d','COLSHADE'),('DCS_noVSE_d','DCS-VSE'),('DCS_VSE_DKA_opt_0_d','DCS-VSE+SO')]
    fig,ax=plt.subplots(figsize=(7.6,4.7)); pts=[]
    for algo,label in SLNN:
        sub=h1[h1['algo']==algo]
        if not len(sub): continue
        pts.append((label,sub['hidden_mean'].mean(),sub['f1_mean'].mean(),
                    sub['f1_mean'].std(ddof=1)/np.sqrt(len(sub)),'SLNN',COL[label]))
    for algo,label,ck in [('XGBoost_matched',r'XGBoost$_{\leq 20}$','XGBoost_m'),
                          ('LGBM_matched',r'LGBM$_{\leq 20}$','LGBM_m')]:
        sub=b8[b8['algo']==algo]
        pts.append((label,sub['leaves_mean'].mean(),sub['f1_mean'].mean(),
                    sub['f1_mean'].std(ddof=1)/np.sqrt(len(sub)),'TreeMatched',COL[ck]))
    # footprint-matched deterministic references (place on the compact axis)
    lr=b10[b10['algo']=='LogReg']; stp=b10[b10['algo']=='Stump']
    pts.append(('Logistic regression',1.0,lr['f1_mean'].mean(),
                lr['f1_mean'].std(ddof=1)/np.sqrt(len(lr)),'Det',COL['LogReg']))
    pts.append(('Decision stump',2.0,stp['f1_mean'].mean(),
                stp['f1_mean'].std(ddof=1)/np.sqrt(len(stp)),'Det',COL['Stump']))
    ax.axvspan(0.9,5.0,color='salmon',alpha=0.08,zorder=0)
    ax.text(1.02,0.86,'Compact corner',fontsize=11,color='#a04040',ha='left',va='top')
    for label,x,y,se,grp,color in pts:
        if grp=='SLNN':
            marker='*' if label=='DCS-VSE' else ('D' if label=='DCS-VSE+SO' else 'o')
            size=240 if label=='DCS-VSE' else (100 if label=='DCS-VSE+SO' else 70)
            edge='black' if label=='DCS-VSE' else color; lw=1.5 if label=='DCS-VSE' else 0.8
        elif grp=='TreeMatched':
            marker='s'; size=100; edge=color; lw=0.8
        else:
            marker='P' if label.startswith('Logistic') else 'X'
            size=150 if label.startswith('Logistic') else 110; edge='black'; lw=1.2
        ax.errorbar(x,y,yerr=1.96*se,fmt='none',ecolor=color,alpha=0.55,capsize=3,capthick=0.7,elinewidth=0.8,zorder=2)
        ax.scatter([x],[y],marker=marker,s=size,color=color,edgecolors=edge,linewidth=lw,zorder=3,label=label)
    dcs=next(p for p in pts if p[0]=='DCS-VSE')
    ax.annotate('DCS--VSE (proposed)\nlowest complexity\nmedian $\\bar{H}\\approx 1$',
                xy=(dcs[1]*1.05,dcs[2]),xytext=(2.7,0.60),fontsize=11,color='#a04040',ha='left',
                arrowprops=dict(arrowstyle='->',color='#a04040',lw=0.7))
    _tf=b9['f1_mean'].mean()
    ax.axhline(_tf,color='#6a3d9a',ls=(0,(6,3)),lw=1.3,alpha=0.9,zorder=1)
    ax.text(30,_tf+0.004,'TabPFN ceiling (%.3f, foundation model)'%_tf,color='#6a3d9a',fontsize=11,va='bottom',ha='left')
    ax.set_xscale('log'); ax.set_xlim(0.9,1.2e4); ax.set_ylim(0.55,0.90)
    ax.set_xlabel('Architectural complexity (SLNN hidden units / tree leaves, log scale)')
    ax.set_ylabel(r'Mean macro-F1 (error bar = $\pm 1.96\,\mathrm{SE}/\sqrt{n_{\mathrm{ds}}}$)')
    ax.grid(True,which='both',linestyle=':',alpha=0.35,zorder=0)
    leg=ax.legend(loc='lower right',bbox_to_anchor=(0.995,0.02),frameon=True,fancybox=False,edgecolor='#888',
                  handletextpad=0.4,borderpad=0.5,labelspacing=0.3,columnspacing=1.0,ncol=2,fontsize=11,
                  title='Method',title_fontsize=11)
    leg.get_frame().set_linewidth(0.5); leg.get_frame().set_alpha(0.95)
    out=os.path.join(ROOT,'figs/v20/pareto_F1_vs_complexity.pdf')
    fig.savefig(out); fig.savefig(out.replace('.pdf','.png')); plt.close(fig); print("saved",out)

# ---------- REGIME SPLIT ----------
def regime():
    micro=['ADENOCARCINOMA','COLON_ALON','DLBCL','LEUKEMIA1','LEUKEMIA2','PROSTATE6033','PROSTATE_TUMOR']
    ds=sorted(h1[h1['algo']=='DCS_noVSE_d']['dataset'].unique())
    lowd=[d for d in ds if d not in micro]
    def mean_over(df,algo,sub):
        s=df[df['algo']==algo].set_index('dataset')['f1_mean']; return s.reindex(sub).mean()
    methods=[('DCS--VSE',h1,'DCS_noVSE_d',COL['DCS-VSE']),
             ('Logistic reg.',b10,'LogReg',COL['LogReg']),
             (r'LGBM$_{\leq 20}$',b8,'LGBM_matched',COL['LGBM_m']),
             ('TabPFN',b9,'TabPFN',COL['TabPFN'])]
    groups=[('Low-dimensional\nclinical (11)',lowd),('High-dimensional\nmicroarray (7)',micro)]
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    nG=len(groups); nM=len(methods); width=0.19; xbase=np.arange(nG)
    for j,(mlabel,df,algo,color) in enumerate(methods):
        vals=[mean_over(df,algo,sub) for _,sub in groups]
        xs=xbase+(j-(nM-1)/2)*width
        bars=ax.bar(xs,vals,width=width,facecolor='white',edgecolor='black',linewidth=0.8,hatch=['////','xxxx','....','++++','||||'][j],label=mlabel)
        for x,v in zip(xs,vals):
            ax.text(x,v+0.006,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(xbase); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0.55,0.95); ax.set_ylabel('Mean macro-F1')
    ax.grid(True,axis='y',linestyle=':',alpha=0.35)
    ax.legend(loc='upper left',ncol=2,frameon=True,fontsize=11,edgecolor='#888')
    out=os.path.join(ROOT,'figs/footprint/regime_split.pdf')
    fig.savefig(out); fig.savefig(out.replace('.pdf','.png')); plt.close(fig); print("saved",out)

pareto(); regime()
