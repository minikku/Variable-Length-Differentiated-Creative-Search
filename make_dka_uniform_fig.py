import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':13,'axes.labelsize':14,'axes.titlesize':13.5,'legend.fontsize':12,
 'xtick.labelsize':12,'ytick.labelsize':12,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
main=pd.read_csv(os.path.join(HERE,'results_H1to20/summary.csv')); b12=pd.read_csv(os.path.join(HERE,'results_B12_dka_uniform/summary.csv'))
def S(df,a,c='f1_mean'): return df[df['algo']==a].set_index('dataset')[c]
ds=sorted(S(main,'DCS_noVSE_d').index)
micro=['ADENOCARCINOMA','COLON_ALON','DLBCL','LEUKEMIA1','LEUKEMIA2','PROSTATE6033','PROSTATE_TUMOR']
lowd=[d for d in ds if d not in micro]
dka=S(main,'DCS_noVSE_d'); uni=S(b12,'DCS_VSE_uniformDKA_d')
fig,ax=plt.subplots(figsize=(6.6,4.0))
groups=[('Low-dimensional\nclinical (11)',lowd,'0.0068'),('High-dimensional\nmicroarray (7)',micro,'0.047')]
x=np.arange(len(groups)); w=0.34
for j,(lab,col,color) in enumerate([('Rank-conditioned DKA',dka,'#0B6E6E'),('Rank-flat control',uni,'#C08457')]):
    vals=[col.reindex(sub).mean() for _,sub,_ in groups]
    xs=x+(j-0.5)*w
    ax.bar(xs,vals,width=w,facecolor='white',edgecolor='black',linewidth=0.9,hatch=['////','....'][j],label=lab)
    for xi,v in zip(xs,vals): ax.text(xi,v+0.006,f'{v:.3f}',ha='center',va='bottom',fontsize=10)
for i,(lab,sub,pv) in enumerate(groups):
    ax.text(i,0.905,f'$p={pv}$',ha='center',va='bottom',fontsize=11,color='#B22222')
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0.60,0.94)
ax.set_ylabel('Mean macro-F1'); ax.legend(loc='upper center',ncol=2,fontsize=11,frameon=True,edgecolor='#888',bbox_to_anchor=(0.5,-0.12))
out=os.path.join(ROOT,'figs/dka/dka_vs_uniform.pdf'); plt.savefig(out); plt.savefig(out.replace('.pdf','.png')); print("saved",out)
print("low-D DKA %.4f uni %.4f | high-D DKA %.4f uni %.4f | all DKA %.4f uni %.4f"%(
 dka.reindex(lowd).mean(),uni.reindex(lowd).mean(),dka.reindex(micro).mean(),uni.reindex(micro).mean(),dka.reindex(ds).mean(),uni.reindex(ds).mean()))
