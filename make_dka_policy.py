import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': True, 'axes.spines.right': True,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

NPOP = 30
NMC  = 400000
rng  = np.random.default_rng(20260704)

def rnd(x):                      # round-half-up nearest integer (matches MATLAB round for (0,1))
    return np.floor(x + 0.5)

def U():  return rng.random(NMC)
def Nrm(): return rng.standard_normal(NMC)

def qi(rho):  # rank-shaped self-improvement coefficient, Eq (qcr)
    return 0.25 + 0.55*np.sqrt(rho)

# Each formation returns q_CB samples (length NMC) for a given rho scalar.
def formations(rho):
    q = qi(rho)
    F = {}
    # Family A
    F[0]  = 0.5*( rnd(U()*rho) + (U()<=q) )
    F[2]  = 0.5*( rnd(Nrm()*(1-rho)) + (Nrm()*(1-rho)<=q) )
    # Family B
    F[1]  = 0.5*( rnd(U()*(1-rho)) + (U()*(1-rho)<=q) )
    F[3]  = 0.5*( rnd(U()*rho) + (U()*(1-rho)<=q) + ((1-rho)<=0.5)*rnd(np.full(NMC,rho)) )
    F[4]  = 0.5*( rnd(U()*rho) + (U()<=q) + ((1-rho)<=0.5)*rnd(np.full(NMC,rho)) )
    F[5]  = 0.5*( rnd(U()*rho) + (U()<=q) + ((1-rho)<=0.5)*rnd(U()*(1-rho)) )
    F[6]  = (1/3)*( rnd(U()*rho) + (U()<=q) + ((1-rho)<=q)*rnd(U()*(1-rho)) + (rho>=q) )
    F[7]  = (1/3)*( rnd(U()*rho) + (U()<=q) + ((1-rho)<=q)*rnd(U()*rho) + (rho>=q) )
    F[12] = 0.5*( rnd(U()*rho) + (U()<=q) + (rho<=0.5)*rnd(np.full(NMC,rho)) + (rho>0.5)*rnd(U()) )
    r1x   = rnd(U())
    b1    = 0.5*( rnd(U()*rho) + (U()<=q) + ((1-rho)<=0.5)*rnd(np.full(NMC,rho)) )
    b2    = (1/3)*( rnd(U()*rho) + (U()<=q) + ((1-rho)<=q)*rnd(U()*(1-rho)) + (rho>=q) )
    F[13] = r1x*b1 + (1-r1x)*b2
    # Family C
    F[11] = (1/3)*( rnd(U()*rho) + (U()<=q) + ((1-rho)<=0.5)*rnd(np.full(NMC,rho)) )
    # Family D
    u1=U(); F[8]  = 0.5*( rnd(u1*rho) + ((1-u1)>=q) + ((1-rho)<=0.5)*rnd(np.full(NMC,rho)) )
    u1=U(); u2=U(); F[9]  = 0.5*( rnd(u2*rho) + ((1-u1)>=q) + ((1-rho)<=0.5)*rnd(np.full(NMC,rho)) )
    u1=U(); u2=U(); F[10] = 0.5*( rnd(u1*rho) + ((1-u1)<=q) + ((1-u2)<=(1-rho))*rnd(np.full(NMC,rho)) )
    return F

# probabilities of the three regimes by nearest of {0,0.5,1}
def regimes(qcb):
    qcb = np.clip(qcb, 0, 1)
    blue   = np.mean(qcb < 0.25)
    orange = np.mean((qcb >= 0.25) & (qcb < 0.75))
    yellow = np.mean(qcb >= 0.75)
    return blue, orange, yellow

# sanity print for opt0
for i in [1,15,20,30]:
    rho=i/NPOP
    b,o,y=regimes(formations(rho)[0])
    print(f"opt0 rank {i:2d}: blue={b:.3f} orange={o:.3f} yellow={y:.3f}")

# ---- MATLAB default colors to match the original panels ----
BLUE   = (0.000, 0.447, 0.741)
ORANGE = (0.850, 0.325, 0.098)
YELLOW = (0.929, 0.694, 0.125)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figs', 'dka')

ranks = np.arange(1, NPOP + 1)
rhos  = ranks / NPOP

def build_panel(fid, legend=True):
    blue = np.zeros(NPOP); orange = np.zeros(NPOP); yellow = np.zeros(NPOP)
    for j, rho in enumerate(rhos):
        b, o, y = regimes(formations(rho)[fid])
        blue[j], orange[j], yellow[j] = b, o, y
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(ranks, blue,   width=1.0, color=BLUE,   edgecolor='black', linewidth=0.4, label='Change only one dimension')
    ax.bar(ranks, orange, width=1.0, bottom=blue, color=ORANGE, edgecolor='black', linewidth=0.4, label='Change 50% of all dimensions')
    ax.bar(ranks, yellow, width=1.0, bottom=blue+orange, color=YELLOW, edgecolor='black', linewidth=0.4, label='Change all dimensions')
    ax.set_xlim(0.5, NPOP + 0.5); ax.set_ylim(0, 1)
    ax.set_xlabel('Rank of individual'); ax.set_ylabel('Probability')
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    if legend:
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l[::-1], loc='upper right', framealpha=0.95,
                  edgecolor='#888', fontsize=11, handlelength=1.4, borderpad=0.4)
    fig.savefig(os.path.join(OUT, f'DKA{fid}-policy.pdf'))
    fig.savefig(os.path.join(OUT, f'DKA{fid}-policy.png'))
    plt.close(fig)

for fid in range(14):
    build_panel(fid, legend=True)
    print('built DKA%d-policy' % fid)
