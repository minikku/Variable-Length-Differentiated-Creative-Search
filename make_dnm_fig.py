import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':13,'axes.labelsize':14,'axes.titlesize':13.5,'legend.fontsize':12,
 'xtick.labelsize':12,'ytick.labelsize':12,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
main=pd.read_csv(os.path.join(HERE,'results_H1to20/summary.csv')); b11=pd.read_csv(os.path.join(HERE,'results_B11_dnm/summary.csv'))
def S(df,a,c='f1_mean'): return df[df['algo']==a].set_index('dataset')[c]
ds=sorted(S(main,'DCS_noVSE_d').index)
micro=['ADENOCARCINOMA','COLON_ALON','DLBCL','LEUKEMIA1','LEUKEMIA2','PROSTATE6033','PROSTATE_TUMOR']
lowd=[d for d in ds if d not in micro]
dnm=S(b11,'DNM_VSE'); dnm1=S(b11,'DNM_fixed1'); slnn=S(main,'DCS_noVSE_d'); Msel=S(b11,'DNM_VSE','dendrites_mean')
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.6,3.9),gridspec_kw={'width_ratios':[1,1.25]})
# panel a: regime bars
groups=[('Low-dimensional\nclinical (11)',lowd),('High-dimensional\nmicroarray (7)',micro)]
methods=[('DNM (variable $M$)',dnm,'#2c7fb8'),('DNM (fixed $M{=}1$)',dnm1,'#7fcdbb'),('SLNN DCS--VSE',slnn,'#d62728')]
x=np.arange(len(groups)); w=0.26
for j,(lab,s,col) in enumerate(methods):
    vals=[s.reindex(sub).mean() for _,sub in groups]
    xs=x+(j-1)*w
    ax1.bar(xs,vals,width=w,facecolor='white',edgecolor='black',linewidth=0.8,hatch=['////','xxxx','....'][j],label=lab)
    for xi,v in zip(xs,vals): ax1.text(xi,v+0.008,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
ax1.set_xticks(x); ax1.set_xticklabels([g[0] for g in groups]); ax1.set_ylim(0.20,0.90)
ax1.set_ylabel('Mean macro-F1'); ax1.legend(loc='upper right',fontsize=10.5,frameon=True,edgecolor='#888')
ax1.set_title('(a) DNM deployment by data regime')
# panel b: self-selected M on low-D
order=sorted(lowd,key=lambda d:Msel[d])
vals=[Msel[d] for d in order]
ax2.barh(np.arange(len(order)),vals,facecolor='white',edgecolor='black',linewidth=0.8,height=0.62,hatch='////')
ax2.set_yticks(np.arange(len(order))); ax2.set_yticklabels([d.replace('_',' ') for d in order],fontsize=10)
ax2.set_xlabel('Self-selected dendrite count $\\bar{M}$'); ax2.set_xlim(0,6)
for i,v in enumerate(vals): ax2.text(v+0.08,i,f'{v:.1f}',va='center',fontsize=9)
ax2.axvline(1,color='#888',ls=':',lw=1); ax2.set_title('(b) Structure self-selected on low-D clinical')
plt.tight_layout()
out=os.path.join(ROOT,'figs/dnm/dnm_deployment.pdf'); plt.savefig(out); plt.savefig(out.replace('.pdf','.png')); print("saved",out)
# print numbers for the text
print("low-D: DNM %.3f DNM1 %.3f SLNN %.3f | high-D DNM %.3f | M low-D mean %.2f range %.1f-%.1f"%(
 dnm.reindex(lowd).mean(),dnm1.reindex(lowd).mean(),slnn.reindex(lowd).mean(),dnm.reindex(micro).mean(),
 Msel.reindex(lowd).mean(),min(Msel.reindex(lowd)),max(Msel.reindex(lowd))))
