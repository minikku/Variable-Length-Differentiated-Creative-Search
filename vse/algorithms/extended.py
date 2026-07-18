"""Recent evolutionary competitors adapted to the variable-length SLNN task.

Each optimizer here is a general population-based continuous optimizer, ported
faithfully from its reference MATLAB implementation. It is made a variable-length
SLNN optimizer through a size-gene encoding, which applies the paper's
cross-length alignment principle uniformly: a candidate is a real vector of
length 1 + D(H_max), where the leading gene selects the hidden size H and the
network is decoded from the length-D(H) prefix of the remaining coordinates. The
size gene searches width, prefix decoding realizes alignment, and each
optimizer's native update rule is used without modification. The size-regularized
MCC objective, population size, and evaluation budget match the rest of the
SLNN family.

Interface for a ported optimizer:
    opt(NP, max_nfe, lb, ub, dim, fobj, rng) -> (best_cost, best_vec)
where fobj(vec) -> scalar fitness (lower is better).
"""
from __future__ import annotations

import math
import numpy as np

from ..network import Network, Layer
from .base import AlgoResult


# --------------------------------------------------------------------------- #
# Size-gene objective: vector <-> variable-length SLNN with prefix decoding    #
# --------------------------------------------------------------------------- #
def _dim_at(H, inp, outp):
    return H * (inp + 1) + outp * (H + 1)


class SizeGeneObjective:
    """Wrap the network Evaluator as a fixed-length continuous objective.

    A candidate ``z`` has length ``1 + D(H_max)``. ``z[0]`` maps to the hidden
    size ``H``; the network weights are read from ``z[1:1+D(H)]``. Coordinates
    beyond the active prefix are inert, which is exactly the cross-length
    alignment rule applied at decode time.
    """

    def __init__(self, evaluator, inp, outp, h_min, h_max, lb, ub):
        self.ev = evaluator
        self.inp = int(inp)
        self.outp = int(outp)
        self.h_min = int(h_min)
        self.h_max = int(h_max)
        self.lb = float(lb)
        self.ub = float(ub)
        self.dim = 1 + _dim_at(self.h_max, self.inp, self.outp)
        self.best_cost = np.inf
        self.best_net = None

    def _decode(self, z):
        span = self.ub - self.lb
        frac = 0.0 if span == 0 else (float(z[0]) - self.lb) / span
        frac = min(1.0, max(0.0, frac))
        H = int(round(self.h_min + (self.h_max - self.h_min) * frac))
        H = min(self.h_max, max(self.h_min, H))
        d = _dim_at(H, self.inp, self.outp)
        vec = np.asarray(z[1:1 + d], dtype=float)
        n_hidden = H * (self.inp + 1)
        hb = vec[:n_hidden].reshape(H, self.inp + 1)
        ob = vec[n_hidden:n_hidden + self.outp * (H + 1)].reshape(self.outp, H + 1)
        net = Network(Layer(hb[:, :self.inp].copy(), hb[:, self.inp:self.inp + 1].copy()),
                      Layer(ob[:, :H].copy(), ob[:, H:H + 1].copy()))
        return net

    def __call__(self, z):
        net = self._decode(z)
        fit, _, _ = self.ev(net)
        if fit < self.best_cost:
            self.best_cost = fit
            self.best_net = net
        return float(fit)


def _clip(x, lb, ub):
    return np.minimum(ub, np.maximum(lb, x))


def make(opt_fn):
    """Return a registry callable ``(options, rng) -> AlgoResult`` for ``opt_fn``."""
    def run(options, rng):
        obj = SizeGeneObjective(options["fobj"], options["inp"], options["outp"],
                                options["min_hidden_size"], options["max_hidden_size"],
                                options["lb"], options["ub"])
        best_cost, _ = opt_fn(int(options["popsize"]), int(options["max_nfe"]),
                              float(options["lb"]), float(options["ub"]),
                              obj.dim, obj, rng)
        best_net = obj.best_net if obj.best_net is not None else obj._decode(
            np.full(obj.dim, 0.5 * (options["lb"] + options["ub"])))
        return AlgoResult(obj.best_cost, best_net, [])
    return run


# --------------------------------------------------------------------------- #
# Ported optimizers                                                            #
# --------------------------------------------------------------------------- #
def hoa(NP, max_nfe, lb, ub, dim, fobj, rng):
    """Hiking Optimization Algorithm (Oladejo, Ekwe & Mirjalili, 2024, KBS 296)."""
    Pop = lb + (ub - lb) * rng.random((NP, dim))
    fit = np.array([fobj(Pop[q]) for q in range(NP)])
    nfe = NP
    best_i = int(np.argmin(fit))
    best_cost = float(fit[best_i]); best_x = Pop[best_i].copy()
    while nfe < max_nfe:
        Xbest = Pop[int(np.argmin(fit))].copy()
        for j in range(NP):
            Xini = Pop[j]
            theta = rng.integers(0, 51)
            s = math.tan(theta)
            SF = rng.integers(1, 3)
            Vel = 6.0 * math.exp(-3.5 * abs(s + 0.05))
            newVel = Vel + rng.random(dim) * (Xbest - SF * Xini)
            newPop = _clip(Pop[j] + newVel, lb, ub)
            fnew = fobj(newPop); nfe += 1
            if fnew < fit[j]:
                Pop[j] = newPop; fit[j] = fnew
            if fnew < best_cost:
                best_cost = float(fnew); best_x = newPop.copy()
            if nfe >= max_nfe:
                break
    return best_cost, best_x


def sslo(NP, max_nfe, lb, ub, dim, fobj, rng):
    """Stochastic Social Learning Optimization (Ye, Sunat & Chiewchanwattana, 2026, KBS 341)."""
    X = lb + (ub - lb) * rng.random((NP, dim))
    fitness = np.array([fobj(X[i]) for i in range(NP)])
    nfe = NP
    Range = ub - lb
    beta = math.e
    unsuccess = np.zeros(NP)
    CR = 0.9 * np.ones(NP)
    it = 0
    MaxIter = max(1, round(max_nfe / NP))
    best_i = int(np.argmin(fitness))
    best_cost = float(fitness[best_i]); best_x = X[best_i].copy()
    newX = np.zeros((NP, dim))
    while nfe < max_nfe:
        it += 1
        CROld = CR.copy()
        xId = np.argsort(fitness)
        for i in range(NP):
            if i == xId[-1] and rng.random() < 0.25:
                w = np.exp(-(beta * np.maximum(0, it - unsuccess[i] + rng.standard_normal(dim) * 10) / MaxIter) ** beta)
                if it < 0.3 * MaxIter:
                    newX[i] = X[i] + w * np.sign(rng.random(dim) - 0.5) * np.abs(X[i])
                else:
                    if unsuccess[i] <= 5:
                        newX[i] = X[xId[0]] + w * rng.random() * np.sign(rng.random(dim) - 0.5) * Range
                    else:
                        unsuccess[i] = 0
                        newX[i] = X[xId[0]] + np.sign(rng.random(dim) - 0.5) * rng.random(dim) * Range
            else:
                jx = int(rng.integers(0, NP))
                while jx == i:
                    jx = int(rng.integers(0, NP))
                Stp = X[i] - X[jx]
                if fitness[jx] < fitness[i]:
                    Stp = -Stp
                newX[i] = X[i] + 1.4588 * np.log(1.0 / rng.random(dim)) * Stp
            below = newX[i] < lb
            newX[i][below] = (X[i][below] + lb) / 2.0
            above = newX[i] > ub
            newX[i][above] = (X[i][above] + ub) / 2.0
            if rng.random() < 0.3:
                CR[i] = rng.random()
            rndqCR = CR[i] - 0.1 * rng.random(dim)
            XCr = (rng.random(dim) <= rndqCR)
            XCr[int(rng.random() * dim)] = True
            newX[i] = np.where(XCr, newX[i], X[i])
        for i in range(NP):
            nf = fobj(newX[i]); nfe += 1
            if nf <= fitness[i]:
                fitness[i] = nf; X[i] = newX[i].copy(); unsuccess[i] = 0
            else:
                CR[i] = CROld[i]; unsuccess[i] += 1
            if nf < best_cost:
                best_cost = float(nf); best_x = newX[i].copy()
            if nfe >= max_nfe:
                break
    return best_cost, best_x


def _levy(dim, rng, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal(dim) * sigma
    v = rng.standard_normal(dim)
    return u / np.abs(v) ** (1 / beta)


def ala(NP, max_nfe, lb, ub, dim, fobj, rng):
    """Artificial Lemming Algorithm (Brownian/Levy explore-exploit)."""
    X = lb + (ub - lb) * rng.random((NP, dim))
    fit = np.array([fobj(X[i]) for i in range(NP)])
    nfe = NP
    bi = int(np.argmin(fit)); Score = float(fit[bi]); Pos = X[bi].copy()
    vec_flag = (1.0, -1.0)
    while nfe < max_nfe:
        RB = rng.standard_normal((NP, dim))
        F = vec_flag[int(math.floor(2 * rng.random()))]
        theta = 2 * math.atan(1 - nfe / max_nfe)
        Xnew = np.empty((NP, dim))
        for i in range(NP):
            E = 2 * math.log(1 / rng.random()) * theta
            if E > 1:
                if rng.random() < 0.3:
                    r1 = 2 * rng.random(dim) - 1
                    Xnew[i] = Pos + F * RB[i] * (r1 * (Pos - X[i]) + (1 - r1) * (X[i] - X[rng.integers(NP)]))
                else:
                    r2 = rng.random() * (1 + math.sin(0.5 * nfe))
                    Xnew[i] = X[i] + F * r2 * (Pos - X[rng.integers(NP)])
            else:
                if rng.random() < 0.5:
                    radius = math.sqrt(np.sum((Pos - X[i]) ** 2))
                    r3 = rng.random()
                    spiral = radius * (math.sin(2 * math.pi * r3) + math.cos(2 * math.pi * r3))
                    Xnew[i] = Pos + F * X[i] * spiral * rng.random()
                else:
                    G = 2 * (1 if rng.random() - 0.5 > 0 else -1) * (1 - nfe / max_nfe)
                    Xnew[i] = Pos + F * G * _levy(dim, rng) * (Pos - X[i])
        for i in range(NP):
            Xnew[i] = _clip(Xnew[i], lb, ub)
            nf = fobj(Xnew[i]); nfe += 1
            if nf < fit[i]:
                X[i] = Xnew[i].copy(); fit[i] = nf
            if fit[i] < Score:
                Score = float(fit[i]); Pos = X[i].copy()
            if nf < Score:
                Score = float(nf); Pos = Xnew[i].copy()
            if nfe >= max_nfe:
                break
    return Score, Pos


def cco(NP, max_nfe, lb, ub, dim, fobj, rng):
    """Centered Collision Optimizer (PCA-guided collisions)."""
    x = lb + (ub - lb) * rng.random((NP, dim))
    fit = np.array([fobj(x[i]) for i in range(NP)])
    nfe = NP
    order = np.argsort(fit); x = x[order]; fit = fit[order]
    fbest = float(fit[0]); sbest = x[0].copy()
    half = max(2, NP // 2)
    K = 0.5
    KP = set(rng.choice(NP, half, replace=False).tolist())
    it = 0; iters = max(1, round(max_nfe / NP))
    while nfe < max_nfe:
        it += 1
        xmean = x.mean(axis=0); xc = x - xmean
        try:
            C = np.cov(xc[half:].T)
            _, V = np.linalg.eigh(C if getattr(C, "ndim", 0) == 2 else np.eye(dim))
        except Exception:
            V = np.eye(dim)
        xpca = x @ V
        a1 = 1.0; a2 = 1.0
        alpha = (math.cos(it * math.pi / iters) + 1) / 3 + 0.2
        for j in range(NP):
            U = (rng.random(dim) > alpha).astype(float)
            r = rng.random() if rng.random() < 1 else rng.standard_normal(dim)
            r1 = alpha
            rj = rng.integers(0, j + 1); rp = rng.integers(0, NP)
            rj2 = rng.integers(0, j + 1); rp2 = rng.integers(0, NP)
            if j in KP:
                base = xpca[rj] * r1 + xpca[rj2] * (1 - r1) + r * (xpca[rj2] - xpca[rp]) + (1 - r) * (xpca[rj2] - xpca[rp2])
                cand = (U * xpca[rng.integers(0, j + 1)] + (1 - U) * base) @ V.T
            else:
                base = x[rj] * r1 + x[rj2] * (1 - r1) + r * (x[rj2] - x[rp]) + (1 - r) * (x[rj2] - x[rp2])
                cand = U * x[rng.integers(0, j + 1)] + (1 - U) * base
            out = (cand < lb) | (cand > ub)
            if out.any():
                cand[out] = lb + rng.random(int(out.sum())) * (ub - lb)
            nf = fobj(cand); nfe += 1
            if nf < fit[j]:
                x[j] = cand; fit[j] = nf
                if j in KP: a1 += 1
                else: a2 += 1
            if nfe >= max_nfe:
                break
        K = (a1 / K) / (a1 / K + a2 / (1 - K)); K = min(0.7, max(0.3, K))
        KP = set(rng.choice(NP, max(1, round(K * NP)), replace=False).tolist())
        order = np.argsort(fit); x = x[order]; fit = fit[order]
        if fit[0] < fbest:
            fbest = float(fit[0]); sbest = x[0].copy()
    return fbest, sbest


def mso(NP, max_nfe, lb, ub, dim, fobj, rng):
    """Mirage Search Optimization (degree-trigonometric refraction updates)."""
    def sind(x): return np.sin(np.deg2rad(x))
    def cosd(x): return np.cos(np.deg2rad(x))
    def tand(x): return np.tan(np.deg2rad(x))
    def atand(x): return np.degrees(np.arctan(x))
    def asind(x): return np.degrees(np.arcsin(np.clip(x, -1, 1)))
    pos = lb + (ub - lb) * rng.random((NP, dim))
    pops = np.array([fobj(pos[i]) for i in range(NP)])
    nfes = NP
    bi = int(np.argmin(pops)); gbests = float(pops[bi]); gbest = pos[bi].copy()
    MaxIter = max(1, round(max_nfe / NP))
    while nfes < max_nfe:
        ac = rng.permutation(np.arange(1, NP)) + 0   # indices 1..NP-1 (0-based here means 1..NP-1)
        cv = int(np.ceil((NP * (2.0 / 3.0)) * ((MaxIter - nfes + 1) / MaxIter)))
        cv = max(0, min(cv, len(ac)))
        newpos = []; newpops = []
        cmax = 1.0
        bound = 5 * np.arctanh(np.clip(-(nfes / max_nfe) + 1, -0.999999, 0.999999)) + cmax
        for j in ac[:cv]:
            cosx = pos[j].copy()
            for k in range(dim):
                h = (gbest[k] - pos[j, k]) * rng.random()
                if h > bound: h = bound
                if h < cmax: h = cmax
                zf = rng.integers(1, 3) * 2 - 3
                a = rng.random() * 20; b = rng.random() * (45 - a / 2); z = rng.integers(1, 3)
                if z == 1:
                    C = b + 90; Dg = 180 - C - a; B = 180 - 2 * Dg; A = 180 - B + a - 90
                    dx = (sind(B) * h * sind(C)) / (sind(Dg) * sind(A) + 1e-12) * zf
                elif z == 2 and a < b:
                    C = 90 - b; Dg = 90 + a - b; B = 180 - 2 * Dg; A = 180 - B - a - 90
                    dx = (sind(B) * h * sind(C)) / (sind(Dg) * sind(A) + 1e-12) * zf
                elif z == 2 and a > b:
                    C = 90 - b; Dg = 180 - C - a; B = 180 - 2 * Dg; A = 180 - B - 90 + a
                    dx = (sind(B) * h * sind(C)) / (sind(Dg) * sind(A) + 1e-12) * zf
                else:
                    dx = 0.0
                cosx[k] = pos[j, k] + dx
            cosx = _clip(cosx, lb, ub)
            c = fobj(cosx); nfes += 1
            if c < gbests: gbests = float(c); gbest = cosx.copy()
            newpos.append(cosx); newpops.append(c)
            if nfes >= max_nfe: break
        if newpos:
            pos = np.vstack([pos] + [np.array(newpos)]); pops = np.concatenate([pops, np.array(newpops)])
            idx = np.argsort(pops)[:NP]; pos = pos[idx]; pops = pops[idx]
        if nfes >= max_nfe: break
        newpos = []; newpops = []
        for j in range(NP):
            hh = gbest - pos[j]
            if np.allclose(hh, 0): hh = np.ones(dim) * 0.05 * (rng.integers(1, 3) * 2 - 3)
            zf = np.sign(hh); hh = np.abs(hh * rng.random(dim))
            gama = rng.random(dim) * 90.0 * ((MaxIter - nfes * 0.99) / MaxIter) + 1e-6
            amax = atand(1.0 / (2 * tand(gama) + 1e-12)); amin = atand((sind(gama) * cosd(gama)) / (1 + sind(gama) ** 2))
            fai = (amax - amin) * rng.random() + amin
            omg = asind(rng.random() * sind(fai + gama))
            x = (hh / (tand(gama) + 1e-12)) - ((((hh / (sind(gama) + 1e-12)) - (hh * sind(fai)) / (cosd(fai + gama) + 1e-12)) * cosd(omg)) / (cosd(omg - gama) + 1e-12))
            cosx = _clip(pos[j] + x * zf, lb, ub)
            c = fobj(cosx); nfes += 1
            if c < gbests: gbests = float(c); gbest = cosx.copy()
            newpos.append(cosx); newpops.append(c)
            if nfes >= max_nfe: break
        if newpos:
            pos = np.vstack([pos] + [np.array(newpos)]); pops = np.concatenate([pops, np.array(newpops)])
            idx = np.argsort(pops)[:NP]; pos = pos[idx]; pops = pops[idx]
    return gbests, gbest


def eisa(NP, max_nfe, lb, ub, dim, fobj, rng):
    """Enhanced Intelligence Swarm / hierarchical-group PSO variant."""
    nGP = 10; nGroup = max(1, NP // nGP)
    c1 = c2 = 1.0; c3 = 0.75
    VelMax = 0.1 * (ub - lb); VelMin = -VelMax
    Pos = lb + (ub - lb) * rng.random((NP, dim))
    Vel = np.zeros((NP, dim))
    Cost = np.array([fobj(Pos[i]) for i in range(NP)]); nfe = NP
    gi = int(np.argmin(Cost)); GBcost = float(Cost[gi]); GBpos = Pos[gi].copy()
    PBcost = Cost.copy(); PBpos = Pos.copy()
    GrBcost = np.empty(nGroup); GrBpos = np.empty((nGroup, dim))
    for g in range(nGroup):
        s = g * nGP; e = min((g + 1) * nGP, NP)
        k = s + int(np.argmin(Cost[s:e])); GrBcost[g] = Cost[k]; GrBpos[g] = Pos[k].copy()
    max_iter = max(1, round(max_nfe / NP)); it = 0
    rival_idx = 0
    while nfe < max_nfe:
        it += 1; w = 1 - 0.5 * (it / max_iter); progress = it / max_iter
        r1 = np.log10(1 + rng.random((NP, dim)) * 9)
        r2 = np.log10(1 + rng.random((NP, dim)) * 9)
        r3 = np.log10(1 + rng.random((NP, dim)) * 9)
        for i in range(NP):
            G = min(i // nGP, nGroup - 1)
            if np.all(Pos[i] == GrBpos[G]):
                Vel[i] = w * Vel[i] + c1 * r1[i] * (PBpos[i] - Pos[i]) + c2 * r2[i] * (GrBpos.mean(axis=0) - Pos[i])
            else:
                Vel[i] = w * Vel[i] + c1 * r1[i] * (PBpos[i] - Pos[i]) + c2 * r2[i] * (GrBpos[G] - Pos[i])
            if i % nGP == 0:
                rg = int(rng.integers(0, nGroup))
                while rg == G and nGroup > 1:
                    rg = int(rng.integers(0, nGroup))
                rival_idx = min(rg * nGP + int(rng.integers(0, nGP)), NP - 1)
            if Cost[i] > Cost[rival_idx]:
                if rng.random() < math.exp(-2 * progress):
                    Vel[i] = -Vel[i]
                else:
                    Vel[i] = Vel[i] + c3 * r3[i] * (Pos[rival_idx] - Pos[i])
            Vel[i] = np.maximum(np.minimum(Vel[i], VelMax), VelMin)
            Pos[i] = _clip(Pos[i] + Vel[i], lb, ub)
            Cost[i] = fobj(Pos[i]); nfe += 1
            if Cost[i] < PBcost[i]:
                PBcost[i] = Cost[i]; PBpos[i] = Pos[i].copy()
                if Cost[i] < GrBcost[G]:
                    GrBcost[G] = Cost[i]; GrBpos[G] = Pos[i].copy()
                    if Cost[i] < GBcost:
                        GBcost = float(Cost[i]); GBpos = Pos[i].copy()
            if nfe >= max_nfe:
                break
    return GBcost, GBpos


def fgo(NP, max_nfe, lb, ub, dim, fobj, rng):
    """Fungal Growth Optimizer (hyphal growth, branching, germination)."""
    M=0.6; Ep=0.7; R=0.9; Tmax=max_nfe
    S=lb+(ub-lb)*rng.random((NP,dim))
    fit=np.array([fobj(S[i]) for i in range(NP)]); t=NP
    gi=int(np.argmin(fit)); Gb_Fit=float(fit[gi]); Gb_Sol=S[gi].copy(); Sp=S.copy()
    def distinct(i):
        while True:
            a,b,c=rng.integers(0,NP),rng.integers(0,NP),rng.integers(0,NP)
            if len({a,b,c,i})==4: return a,b,c
    def repair(v): 
        o=(v>ub)|(v<lb); 
        if o.any(): v[o]=lb+rng.random(int(o.sum()))*(ub-lb)
        return v
    while t<Tmax:
        if t<=Tmax/2: nutrients=rng.random(NP)
        else: nutrients=fit.copy()
        nutrients=nutrients/(nutrients.sum()+1e-12)+2*rng.random()
        fmin,fmax=fit.min(),fit.max()
        if rng.random()<rng.random():
            for i in range(NP):
                a,b,c=distinct(i)
                p=(fit[i]-fmin)/(fmax-fmin+1e-12); Er=M+(1-t/Tmax)*(1-M)
                if p<Er:
                    F=(fit[i]/(fit.sum()+1e-12))*rng.random()*(1-t/Tmax)**(1-t/Tmax); E=math.exp(F)
                    U1=(rng.random(dim)<rng.random()).astype(float)
                    Sn=U1*S[i]+(1-U1)*(S[i]+E*(S[a]-S[b]))
                else:
                    Ec=(rng.random(dim)-0.5)*rng.random()*(S[a]-S[b])
                    if rng.random()<rng.random():
                        De2=rng.random(dim)*(S[i]-Gb_Sol)*(rng.random(dim)>rng.random())
                        Sn=S[i]+De2*nutrients[i]+Ec*(rng.random()>rng.random())
                    else:
                        De=rng.random()*(S[a]-S[i])+rng.random(dim)*((rng.random()>rng.random()*2-1)*Gb_Sol-S[i])*(rng.random()>R)
                        Sn=S[i]+De*nutrients[i]+Ec*(rng.random()>Ep)
                Sn=repair(Sn); nF=fobj(Sn); t+=1
                if fit[i]<nF: pass
                else:
                    S[i]=Sn; Sp[i]=Sn.copy(); fit[i]=nF
                    if fit[i]<=Gb_Fit: Gb_Fit=float(fit[i]); Gb_Sol=Sn.copy()
                if t>Tmax: break
        else:
            r5=rng.random()
            for i in range(NP):
                a,b,c=distinct(i); Sn=S[i].copy()
                if rng.random()<0.5:
                    EL=1+math.exp(fit[i]/(fit.sum()+1e-12))*(rng.random()>rng.random())
                    Dep1=S[b]-S[c]; Dep2=S[a]-Gb_Sol
                    U1=(rng.random(dim)<rng.random()).astype(float)
                    Sn=S[i]*U1+(S[i]+r5*Dep1*EL+(1-r5)*Dep2*EL)*(1-U1)
                else:
                    sig=1 if rng.random()>rng.random()*2-1 else -1
                    F=(fit[i]/(fit.sum()+1e-12))*rng.random()*(1-t/Tmax)**(1-t/Tmax); E=math.exp(F)
                    for j in range(dim):
                        mu=sig*rng.random()*E
                        if rng.random()>rng.random():
                            Sn[j]=(((t/Tmax)*Gb_Sol[j]+(1-t/Tmax)*S[a,j])+S[b,j])/2.0+mu*abs((S[c,j]+S[a,j]+S[b,j])/3.0-S[i,j])
                Sn=repair(Sn); nF=fobj(Sn); t+=1
                if fit[i]<nF: pass
                else:
                    S[i]=Sn; Sp[i]=Sn.copy(); fit[i]=nF
                    if fit[i]<=Gb_Fit: Gb_Fit=float(fit[i]); Gb_Sol=Sn.copy()
                if t>Tmax: break
    return Gb_Fit, Gb_Sol


# CCO uses a per-iteration covariance and eigendecomposition, which is O(D^2)
# and not viable at the full encoding dimension on the high-dimensional
# microarray datasets; it is defined above but not registered for the full run.
EXTENDED = {
    "HOA_d": make(hoa),
    "SSLO_d": make(sslo),
    "ALA_d": make(ala),
    "MSO_d": make(mso),
    "EISA_d": make(eisa),
    "FGO_d": make(fgo),
}
