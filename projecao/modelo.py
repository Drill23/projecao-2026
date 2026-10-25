"""Modelo de projeção do 2º turno a partir das seções já apuradas.

Ideia (em uma frase): para cada município, compara o que já foi apurado no 2º turno com o
que AS MESMAS SEÇÕES deram no 1º turno; aprende como os votos dos candidatos eliminados se
dividiram e quanto cada região/UF/município se moveu; aplica isso às seções que faltam.

Notação
-------
G[i, k]  votos do 1º turno na seção i para o grupo k (k=0 é o candidato A, k=1 o B,
         k>=2 os demais grupos, cujos votos são "transferidos" com fração q_k para A).
C1[i]    comparecimento da seção no 1º turno.  E[i]  eleitores aptos.
Observado por município m (parte apurada): votos de A e B, comparecimento,
         eleitorado apurado (est) e total (te).

Para o agregado das seções apuradas do município m:
    base_m(q)  = log((A1 + sum_k q_k O_k + .5) / (B1 + sum_k (1-q_k) O_k + .5))
    y_m        = log((A2 + .5) / (B2 + .5))
    y_m        = base_m(q) + X_m beta + u_uf + v_m + ruído
com X = (interceptos por região, covariáveis de seção agregadas, porte do município),
u_uf e v_m efeitos aleatórios com encolhimento empírico-bayesiano.
As seções pendentes recebem logit p_i = base_i(q) + X_i beta + u_uf + rho * v_m.
Incerteza: bootstrap de Poisson dos municípios + sorteio dos efeitos não observados.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

EPS = 0.5


def _bincount(idx, w, n):
    return np.bincount(idx, weights=w, minlength=n)


class Modelo:
    def __init__(self, G, C1, E, mun, uf, reg, X_sec, q_prior, q_sd, n_mun, n_uf, n_reg,
                 nomes_cov=(), rho=0.85, prior_turn=1.0, prior_valid=0.95, sd_reg=None):
        self.G = np.asarray(G, float)              # (n, K)
        self.K = self.G.shape[1]
        self.C1 = np.asarray(C1, float)
        self.E = np.asarray(E, float)
        self.mun = np.asarray(mun, np.int64)
        self.uf = np.asarray(uf, np.int64)
        self.reg = np.asarray(reg, np.int64)
        self.X_sec = np.asarray(X_sec, float).reshape(len(self.E), -1)   # covariáveis por seção
        self.q_prior = logit(np.clip(np.asarray(q_prior, float), 0.02, 0.98))
        self.q_sd = np.asarray(q_sd, float)
        self.n_mun, self.n_uf, self.n_reg = n_mun, n_uf, n_reg
        self.nomes_cov = list(nomes_cov)
        self.rho = rho
        self.prior_turn = prior_turn
        self.prior_valid = prior_valid
        # desvio de cada região em relação ao Brasil (o exterior, índice 5, é bem diferente)
        if sd_reg is None:
            sd_reg = np.full(n_reg, 0.08)
            if n_reg > 5:
                sd_reg[5] = 0.35
        self.sd_reg = np.asarray(sd_reg, float)
        # UF e região de cada município
        self.uf_mun = np.zeros(n_mun, np.int64)
        self.uf_mun[self.mun] = self.uf
        self.reg_mun = np.zeros(n_mun, np.int64)
        self.reg_mun[self.mun] = self.reg
        Em = _bincount(self.mun, self.E, n_mun)
        self.logE_mun = np.log(np.maximum(Em, 50.0))
        self.logE_mun -= np.average(self.logE_mun, weights=np.maximum(Em, 1))
        self.n_cov = self.X_sec.shape[1]
        self._x0 = None

    # ------------------------------------------------------------------ util
    def _agg_counted(self, cont):
        """Somas das seções apuradas por município (base do 1º turno)."""
        n = self.n_mun
        w = cont.astype(float)
        Gc = np.stack([_bincount(self.mun, self.G[:, k] * w, n) for k in range(self.K)], 1)
        C1c = _bincount(self.mun, self.C1 * w, n)
        Ec = _bincount(self.mun, self.E * w, n)
        # covariáveis agregadas ponderadas pelos votos válidos do 1º turno
        Vb = self.G.sum(1) * w
        Vbm = _bincount(self.mun, Vb, n)
        Xc = np.stack([_bincount(self.mun, self.X_sec[:, j] * Vb, n) / np.maximum(Vbm, 1e-9)
                       for j in range(self.n_cov)], 1) if self.n_cov else np.zeros((n, 0))
        return Gc, C1c, Ec, Xc

    @staticmethod
    def _base(Ga, q):
        """log-razão A/B do 1º turno com transferências q (Ga: (m, K))."""
        A = Ga[:, 0] + Ga[:, 2:] @ q + EPS
        B = Ga[:, 1] + Ga[:, 2:] @ (1 - q) + EPS
        return np.log(A) - np.log(B)

    def _unpack(self, th):
        """th = [logit q (K-2), efeito Brasil, desvios das regiões, covariáveis, porte]."""
        K2 = self.K - 2
        q = expit(th[:K2])
        bnat = th[K2]
        breg = bnat + th[K2 + 1:K2 + 1 + self.n_reg]
        bcov = th[K2 + 1 + self.n_reg:K2 + 1 + self.n_reg + self.n_cov]
        bsize = th[-1]
        return q, breg, bcov, bsize

    def _pred_mun(self, th, Gc, Xc, idx):
        q, breg, bcov, bsize = self._unpack(th)
        p = self._base(Gc[idx], q) + breg[self.reg_mun[idx]] + bsize * self.logE_mun[idx]
        if self.n_cov:
            p = p + Xc[idx] @ bcov
        return p

    # ------------------------------------------------------------------ ajuste
    def _ajustar(self, Gc, Xc, idx, y, w, x0, prior_reg=0.0):
        K2 = self.K - 2
        mu = np.concatenate([self.q_prior, [prior_reg], np.zeros(self.n_reg), np.zeros(self.n_cov), [0.0]])
        sd = np.concatenate([self.q_sd, [0.6], self.sd_reg, np.full(self.n_cov, 1.5), [0.3]])
        sw = np.sqrt(w)

        def res(th):
            r = sw * (y - self._pred_mun(th, Gc, Xc, idx))
            return np.concatenate([r, (th - mu) / sd])
        sol = least_squares(res, x0 if x0 is not None else mu.copy(), method="trf",
                            x_scale="jac", max_nfev=200)
        return sol.x

    def projetar(self, cont, obs, n_boot=120, seed=0, prior_reg=0.0, full=False):
        """cont: bool por seção (apurada segundo o arquivo de seções).
        obs: dict de arrays por município: A, B, comp, est, te (parte apurada / total)."""
        rng = np.random.default_rng(seed)
        n = self.n_mun
        A = np.asarray(obs["A"], float)
        Bv = np.asarray(obs["B"], float)
        comp = np.asarray(obs["comp"], float)
        est = np.asarray(obs["est"], float)
        te = np.asarray(obs["te"], float)
        cont = np.asarray(cont, bool)

        Gc, C1c, Ec, Xc = self._agg_counted(cont)
        Etot = _bincount(self.mun, self.E, n)
        # --- conciliação proporcional (arquivo de seções x resultado do município)
        fc = np.where(Ec > 0, est / np.maximum(Ec, 1), 0.0)
        sem_cs = (Ec <= 0) & (est > 0)            # resultado chegou, arquivo de seções ainda não
        if sem_cs.any():
            frac = np.where(Etot > 0, est / np.maximum(Etot, 1), 0)
            Gall = np.stack([_bincount(self.mun, self.G[:, k], n) for k in range(self.K)], 1)
            C1all = _bincount(self.mun, self.C1, n)
            Gc[sem_cs] = Gall[sem_cs] * frac[sem_cs, None]
            C1c[sem_cs] = C1all[sem_cs] * frac[sem_cs]
            Ec[sem_cs] = est[sem_cs]
            fc[sem_cs] = 1.0
        Gc = Gc * fc[:, None]
        C1c = C1c * fc
        Ependb = Etot - _bincount(self.mun, self.E * cont, n)
        Ependb = np.where(sem_cs, Etot * (1 - np.where(Etot > 0, est / np.maximum(Etot, 1), 0)), Ependb)
        fp = np.where(Ependb > 0, np.maximum(te - est, 0) / np.maximum(Ependb, 1), 0.0)
        fp = np.clip(fp, 0, 3)
        if sem_cs.any():
            # seções desse município: nenhuma está marcada como apurada; a fração pendente é
            # aplicada sobre todas elas
            pass

        V = A + Bv
        idx = np.where((V >= 1) & (Gc.sum(1) >= 1))[0]
        out = {"apurado_A": float(A.sum()), "apurado_B": float(Bv.sum()),
               "apurado_comp": float(comp.sum()),
               "pct_eleitorado": float(est.sum() / max(te.sum(), 1)),
               "n_mun_obs": int(len(idx))}
        if len(idx) < 5:
            out["ok"] = False
            return out

        y = np.log((A[idx] + EPS) / (Bv[idx] + EPS))
        p_obs = np.clip((A[idx] + EPS) / (V[idx] + 2 * EPS), 0.02, 0.98)
        noise = 1.0 / (V[idx] * p_obs * (1 - p_obs))
        noise_full = np.zeros(n)
        noise_full[idx] = noise
        sig2 = 0.01
        th = self._x0
        for _ in range(3):
            w = 1.0 / (sig2 + noise)
            th = self._ajustar(Gc, Xc, idx, y, w, th, prior_reg)
            r = y - self._pred_mun(th, Gc, Xc, idx)
            sig2 = float(np.clip(np.sum(V[idx] * (r ** 2 - noise)) / np.sum(V[idx]), 2e-4, 0.25))
        self._x0 = th

        # --- comparecimento e taxa de votos válidos (A+B)/comp, por UF com encolhimento
        uf_i = self.uf_mun[idx]
        C1o = _bincount(uf_i, C1c[idx], self.n_uf)
        compo = _bincount(uf_i, comp[idx], self.n_uf)
        Vo = _bincount(uf_i, V[idx], self.n_uf)
        k_nat = 2e5
        turn_nat = (compo.sum() + k_nat * self.prior_turn) / (C1o.sum() + k_nat)
        val_nat = (Vo.sum() + k_nat * self.prior_valid) / (compo.sum() + k_nat)
        k_uf = 3e4
        turn_uf = (compo + k_uf * turn_nat) / (C1o + k_uf)
        val_uf = (Vo + k_uf * val_nat) / (compo + k_uf)

        pend = ~cont
        # seções do município sem arquivo de seções mas com resultado: pendente = fração restante
        sec_fp = fp[self.mun]
        sec_fp = np.where(sem_cs[self.mun], np.where(Etot[self.mun] > 0,
                          np.maximum(te - est, 0)[self.mun] / np.maximum(Etot[self.mun], 1), 0), sec_fp)
        pend = pend | sem_cs[self.mun]
        Vp = self.C1 * turn_uf[self.uf] * val_uf[self.uf] * sec_fp * pend   # votos válidos A+B previstos

        Xs = self.X_sec

        def efeitos(th_, w_, y_, idx_):
            r_ = y_ - self._pred_mun(th_, Gc, Xc, idx_)
            ufi = self.uf_mun[idx_]
            sw_ = _bincount(ufi, w_, self.n_uf)
            swr = _bincount(ufi, w_ * r_, self.n_uf)
            # tau2 (variância entre UFs) por momentos
            ruf = np.where(sw_ > 0, swr / np.maximum(sw_, 1e-12), 0)
            tem = sw_ > 0
            tau2 = float(np.clip(np.average(ruf[tem] ** 2, weights=np.sqrt(sw_[tem])) if tem.sum() > 3 else 0.01,
                                 1e-4, 0.05))
            u = swr / (sw_ + 1.0 / tau2)
            rm = r_ - u[ufi]
            vm = np.zeros(n)
            vm[idx_] = rm * sig2 / (sig2 + noise_full[idx_])
            return u, vm, tau2, sw_

        def projeta(th_, u, vm, extra_mun=None, extra_uf=None):
            q, breg, bcov, bsize = self._unpack(th_)
            eta = self._base(self.G, q) + breg[self.reg] + bsize * self.logE_mun[self.mun] + u[self.uf] \
                + self.rho * vm[self.mun]
            if self.n_cov:
                eta = eta + Xs @ bcov
            if extra_mun is not None:
                eta = eta + extra_mun[self.mun]
            if extra_uf is not None:
                eta = eta + extra_uf[self.uf]
            pa = expit(eta)
            Ap = Vp * pa
            return Ap, Vp - Ap

        w = 1.0 / (sig2 + noise)
        u, vm, tau2, swuf = efeitos(th, w, y, idx)
        Ap, Bp = projeta(th, u, vm)
        totA = A.sum() + Ap.sum()
        totB = Bv.sum() + Bp.sum()
        q, breg, bcov, bsize = self._unpack(th)
        out.update({"ok": True, "A": float(totA), "B": float(totB),
                    "pA": float(totA / (totA + totB)), "sig2": sig2, "tau2": tau2,
                    "q": q.tolist(), "breg": breg.tolist(), "bcov": bcov.tolist(), "bsize": float(bsize),
                    "turn_nat": float(turn_nat), "val_nat": float(val_nat)})
        if full:
            out["Ap_sec"], out["Bp_sec"], out["u"], out["vm"] = Ap, Bp, u, vm
        # por UF (projeção)
        ufA = _bincount(self.uf_mun, A, self.n_uf) + _bincount(self.uf, Ap, self.n_uf)
        ufB = _bincount(self.uf_mun, Bv, self.n_uf) + _bincount(self.uf, Bp, self.n_uf)
        out["uf_pA"] = (ufA / np.maximum(ufA + ufB, 1)).tolist()
        out["uf_V"] = (ufA + ufB).tolist()
        out["uf_apur"] = (_bincount(self.uf_mun, est, self.n_uf) / np.maximum(_bincount(self.uf_mun, te, self.n_uf), 1)).tolist()
        out["uf_obs_pA"] = (_bincount(self.uf_mun, A, self.n_uf) /
                            np.maximum(_bincount(self.uf_mun, A + Bv, self.n_uf), 1)).tolist()

        # --- bootstrap
        sims = []
        obs_mun = np.zeros(n, bool)
        obs_mun[idx] = True
        for b in range(n_boot):
            pw = rng.poisson(1.0, len(idx)).astype(float)
            ok = pw > 0
            if ok.sum() < 5:
                continue
            wb = w[ok] * pw[ok]
            thb = self._ajustar(Gc, Xc, idx[ok], y[ok], wb, th, prior_reg)
            ub, vmb, _, swb = efeitos(thb, wb, y[ok], idx[ok])
            # efeitos não observados: UFs sem dados e municípios sem dados
            ex_uf = rng.normal(0, np.sqrt(tau2), self.n_uf) * (1.0 / (1.0 + swb * tau2))
            ex_mun = rng.normal(0, np.sqrt(sig2), n)
            ex_mun = np.where(obs_mun, ex_mun * np.sqrt(max(1 - self.rho ** 2, 0)) * 0.5, ex_mun)
            Apb, Bpb = projeta(thb, ub, vmb, ex_mun, ex_uf)
            # incerteza no comparecimento/válidos pendentes (pequena)
            s = 1 + rng.normal(0, 0.004)
            sims.append((A.sum() + Apb.sum() * s, Bv.sum() + Bpb.sum() * s))
        if sims:
            S = np.array(sims)
            pa = S[:, 0] / S.sum(1)
            out["pA_sims"] = pa.tolist() if full else None
            out["pA_lo"], out["pA_hi"] = float(np.quantile(pa, 0.05)), float(np.quantile(pa, 0.95))
            out["pA_sd"] = float(pa.std())
            out["prob_A"] = float((pa > 0.5).mean())
        return out
