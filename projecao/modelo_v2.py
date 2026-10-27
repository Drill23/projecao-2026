"""Modelo de projeção v2 para qualquer 2º turno (presidente ou governador).

O que muda em relação ao v1 (projecao/modelo.py):

1. Mobilização. Além das transferências dos candidatos eliminados, o modelo tem um termo
   para os votos válidos "novos" do 2º turno (quem se absteve ou votou branco/nulo no 1º turno
   e agora votou em A ou B). Em cada município apurado, N = max(válidos 2º turno - válidos
   1º turno, 0) entra como mais um grupo, com fração q_N indo para A.
2. Hierarquia geográfica genérica em duas camadas (nível 1 e nível 2). Para presidente:
   região → UF. Para governador: grupos de municípios formados pela posição geográfica
   (vizinhos parecidos), em vez de um único nível estadual.
3. Variância entre unidades com prior aprendido do histórico (estável no começo da apuração,
   quando poucas unidades têm dados).
4. Prior opcional de pesquisa para o efeito nacional.
5. Incerteza sensível à cobertura: além dos sorteios dos parâmetros e das unidades não vistas,
   um termo cresce quando unidades grandes ainda estão sem apuração (índice U). A dispersão
   final é sqrt((a·sd)² + (b·pendente)² + (c·p(1-p)·U)²), com a, b, c calibrados no histórico.

Notação igual ao v1: G (n, K) votos do 1º turno por seção, colunas [A, B, outros...].
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit
from scipy.stats import norm

EPS = 0.5


def _bc(idx, w, n):
    return np.bincount(idx, weights=w, minlength=n)


class ModeloV2:
    def __init__(self, G, C1, E, mun, l2, l1, n_mun, n_l2, n_l1, q_prior=None, q_sd=None,
                 mobiliza=True, qN_prior=0.5, qN_sd=1.0, prior_nat=(0.0, 0.6), sd_l1=0.08,
                 sd_l1_unidade=None, tau_prior=0.035, k_tau=4.0, rho=0.85,
                 calib=(1.3, 0.004, 1.0), prior_turn=1.0, prior_valid=0.95):
        self.G = np.asarray(G, float)
        self.K = self.G.shape[1]
        self.V1 = self.G.sum(1)
        self.C1 = np.asarray(C1, float)
        self.E = np.asarray(E, float)
        self.mun = np.asarray(mun, np.int64)
        self.l2 = np.asarray(l2, np.int64)
        self.l1 = np.asarray(l1, np.int64)
        self.n_mun, self.n_l2, self.n_l1 = n_mun, n_l2, n_l1
        K2 = self.K - 2
        qp = np.full(K2, 0.5) if q_prior is None else np.asarray(q_prior, float)
        self.q_prior = logit(np.clip(qp, 0.02, 0.98))
        self.q_sd = np.full(K2, 0.5) if q_sd is None else np.asarray(q_sd, float)
        self.mobiliza = mobiliza
        self.qN_prior, self.qN_sd = float(logit(np.clip(qN_prior, 0.02, 0.98))), qN_sd
        self.prior_nat = prior_nat
        self.sd_l1 = np.full(n_l1, sd_l1) if sd_l1_unidade is None else np.asarray(sd_l1_unidade, float)
        self.tau_prior, self.k_tau = tau_prior, k_tau
        self.rho = rho
        self.calib = calib
        self.prior_turn, self.prior_valid = prior_turn, prior_valid
        self.l2_mun = np.zeros(n_mun, np.int64)
        self.l2_mun[self.mun] = self.l2
        self.l1_mun = np.zeros(n_mun, np.int64)
        self.l1_mun[self.mun] = self.l1
        Em = _bc(self.mun, self.E, n_mun)
        self.logE_mun = np.log(np.maximum(Em, 50.0))
        self.logE_mun -= np.average(self.logE_mun, weights=np.maximum(Em, 1))
        self.E_l2 = _bc(self.l2, self.E, n_l2)
        self._x0 = None

    # ----------------------------------------------------------------- parâmetros
    def _n_par(self):
        return (self.K - 2) + (1 if self.mobiliza else 0) + 1 + self.n_l1 + 1

    def _unpack(self, th):
        K2 = self.K - 2
        q = expit(th[:K2])
        j = K2
        qN = 0.5
        if self.mobiliza:
            qN = expit(th[j])
            j += 1
        bnat = th[j]
        bl1 = bnat + th[j + 1:j + 1 + self.n_l1]
        bsize = th[-1]
        return q, qN, bnat, bl1, bsize

    @staticmethod
    def _base(Ga, q, qN, N):
        A = Ga[:, 0] + Ga[:, 2:] @ q + qN * N + EPS
        B = Ga[:, 1] + Ga[:, 2:] @ (1 - q) + (1 - qN) * N + EPS
        return np.log(A) - np.log(B)

    def _pred(self, th, Gc, N, idx):
        q, qN, bnat, bl1, bsize = self._unpack(th)
        return self._base(Gc[idx], q, qN, N[idx]) + bl1[self.l1_mun[idx]] + bsize * self.logE_mun[idx]

    def _ajustar(self, Gc, N, idx, y, w, x0, cov=False):
        K2 = self.K - 2
        mu = list(self.q_prior)
        sd = list(self.q_sd)
        if self.mobiliza:
            mu.append(self.qN_prior)
            sd.append(self.qN_sd)
        mu += [self.prior_nat[0]] + [0.0] * self.n_l1 + [0.0]
        sd += [self.prior_nat[1]] + list(self.sd_l1) + [0.3]
        mu, sd = np.array(mu), np.array(sd)
        sw = np.sqrt(w)

        def res(th):
            return np.concatenate([sw * (y - self._pred(th, Gc, N, idx)), (th - mu) / sd])
        sol = least_squares(res, x0 if x0 is not None else mu.copy(), method="trf", x_scale="jac",
                            max_nfev=200)
        if cov:
            J = sol.jac
            return sol.x, np.linalg.pinv(J.T @ J)
        return sol.x

    # ----------------------------------------------------------------- projeção
    def projetar(self, cont, obs, n_draws=120, seed=0):
        rng = np.random.default_rng(seed)
        n = self.n_mun
        A = np.asarray(obs["A"], float)
        Bv = np.asarray(obs["B"], float)
        comp = np.asarray(obs["comp"], float)
        est = np.asarray(obs["est"], float)
        te = np.asarray(obs["te"], float)
        cont = np.asarray(cont, bool)
        w = cont.astype(float)
        Gc = np.stack([_bc(self.mun, self.G[:, k] * w, n) for k in range(self.K)], 1)
        C1c = _bc(self.mun, self.C1 * w, n)
        Ec = _bc(self.mun, self.E * w, n)
        Etot = _bc(self.mun, self.E, n)
        # conciliação arquivo de seções x resultado do município
        fc = np.where(Ec > 0, est / np.maximum(Ec, 1), 0.0)
        sem_cs = (Ec <= 0) & (est > 0)
        if sem_cs.any():
            frac = np.where(Etot > 0, est / np.maximum(Etot, 1), 0)
            Gall = np.stack([_bc(self.mun, self.G[:, k], n) for k in range(self.K)], 1)
            Gc[sem_cs] = Gall[sem_cs] * frac[sem_cs, None]
            C1c[sem_cs] = _bc(self.mun, self.C1, n)[sem_cs] * frac[sem_cs]
            Ec[sem_cs] = est[sem_cs]
            fc[sem_cs] = 1.0
        fc = np.clip(fc, 0, 3)
        resto_c = np.where(sem_cs, 0.0, np.clip(1 - fc, 0, 1))
        Gc = Gc * fc[:, None]
        C1c = C1c * fc
        Econt = _bc(self.mun, self.E * w, n)
        Ependb = Etot - Econt + Econt * resto_c
        Ependb = np.where(sem_cs, Etot * (1 - np.where(Etot > 0, est / np.maximum(Etot, 1), 0)), Ependb)
        fp = np.clip(np.where(Ependb > 0, np.maximum(te - est, 0) / np.maximum(Ependb, 1), 0.0), 0, 3)

        V = A + Bv
        Nc = np.maximum(V - Gc.sum(1), 0) if self.mobiliza else np.zeros(n)
        idx = np.where((V >= 1) & (Gc.sum(1) >= 1))[0]
        out = {"pct": float(est.sum() / max(te.sum(), 1)), "n_mun_obs": int(len(idx))}
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
            wt = 1.0 / (sig2 + noise)
            th = self._ajustar(Gc, Nc, idx, y, wt, th)
            r = y - self._pred(th, Gc, Nc, idx)
            sig2 = float(np.clip(np.sum(V[idx] * (r ** 2 - noise)) / np.sum(V[idx]), 2e-4, 0.25))
        wt = 1.0 / (sig2 + noise)
        th, Sigma = self._ajustar(Gc, Nc, idx, y, wt, th, cov=True)
        self._x0 = th

        # comparecimento e válidos por unidade de nível 2, encolhidos ao total
        l2i = self.l2_mun[idx]
        C1o = _bc(l2i, C1c[idx], self.n_l2)
        compo = _bc(l2i, comp[idx], self.n_l2)
        Vo = _bc(l2i, V[idx], self.n_l2)
        k_nat, k_u = 2e5, 3e4
        turn_nat = (compo.sum() + k_nat * self.prior_turn) / (C1o.sum() + k_nat)
        val_nat = (Vo.sum() + k_nat * self.prior_valid) / (compo.sum() + k_nat)
        turn_u = (compo + k_u * turn_nat) / (C1o + k_u)
        val_u = (Vo + k_u * val_nat) / (compo + k_u)

        sec_fp = fp[self.mun]
        sec_fp = np.where(sem_cs[self.mun], np.where(Etot[self.mun] > 0,
                          np.maximum(te - est, 0)[self.mun] / np.maximum(Etot[self.mun], 1), 0), sec_fp)
        peso = np.where(cont & ~sem_cs[self.mun], resto_c[self.mun], 1.0) * sec_fp
        Vp = self.C1 * turn_u[self.l2] * val_u[self.l2] * peso
        Np = np.maximum(Vp - self.V1 * peso, 0) if self.mobiliza else np.zeros(len(Vp))
        Gp = self.G * peso[:, None]

        def efeitos(th_):
            r_ = y - self._pred(th_, Gc, Nc, idx)
            sw_ = _bc(l2i, wt, self.n_l2)
            swr = _bc(l2i, wt * r_, self.n_l2)
            tem = sw_ > 0
            ru = np.where(tem, swr / np.maximum(sw_, 1e-12), 0)
            n_u = int(tem.sum())
            t_obs = float(np.average(ru[tem] ** 2, weights=np.sqrt(sw_[tem]))) if n_u > 1 else self.tau_prior ** 2
            tau2 = float(np.clip((n_u * t_obs + self.k_tau * self.tau_prior ** 2) / (n_u + self.k_tau), 1e-4, 0.05))
            u = swr / (sw_ + 1.0 / tau2)
            vm = np.zeros(n)
            vm[idx] = (r_ - u[l2i]) * sig2 / (sig2 + noise)
            return u, vm, tau2, sw_

        def projeta(th_, u, vm, ex_mun=None, ex_u=None):
            q, qN, bnat, bl1, bsize = self._unpack(th_)
            eta = self._base(Gp, q, qN, Np) + bl1[self.l1] + bsize * self.logE_mun[self.mun] + u[self.l2] \
                + self.rho * vm[self.mun]
            if ex_mun is not None:
                eta = eta + ex_mun[self.mun]
            if ex_u is not None:
                eta = eta + ex_u[self.l2]
            pa = expit(eta)
            return Vp * pa, Vp * (1 - pa)

        u, vm, tau2, swu = efeitos(th)
        Ap, Bp = projeta(th, u, vm)
        tA, tB = A.sum() + Ap.sum(), Bv.sum() + Bp.sum()
        pA = float(tA / (tA + tB))
        q, qN, bnat, bl1, bsize = self._unpack(th)
        out.update({"ok": True, "pA": pA, "A": float(tA), "B": float(tB), "q": q.tolist(), "qN": float(qN),
                    "sig2": sig2, "tau2": tau2, "turn": float(turn_nat), "val": float(val_nat)})
        # por unidade de nível 2
        uA = _bc(self.l2_mun, A, self.n_l2) + _bc(self.l2, Ap, self.n_l2)
        uB = _bc(self.l2_mun, Bv, self.n_l2) + _bc(self.l2, Bp, self.n_l2)
        out["u_pA"] = (uA / np.maximum(uA + uB, 1)).tolist()
        out["u_V"] = (uA + uB).tolist()
        cov_u = _bc(self.l2_mun, est, self.n_l2) / np.maximum(_bc(self.l2_mun, te, self.n_l2), 1)
        out["u_apur"] = cov_u.tolist()
        out["u_bruto"] = (_bc(self.l2_mun, A, self.n_l2) / np.maximum(_bc(self.l2_mun, V, self.n_l2), 1)).tolist()
        # índice de cobertura: peso dos votos pendentes em unidades quase sem apuração
        Pu = _bc(self.l2, Vp, self.n_l2)
        T = tA + tB
        U = float(np.sqrt(np.sum((Pu / max(T, 1)) ** 2 * np.exp(-cov_u / 0.05))))
        out["U"] = U
        out["pend"] = float(Vp.sum() / max(T, 1))
        # sorteios
        sims = []
        try:
            Lc = np.linalg.cholesky(Sigma + 1e-10 * np.eye(len(th)))
        except np.linalg.LinAlgError:
            ev, evec = np.linalg.eigh(Sigma)
            Lc = evec * np.sqrt(np.clip(ev, 0, None))
        info_u = C1o / (C1o + k_u)
        obs_mun = np.zeros(n, bool)
        obs_mun[idx] = True
        for _ in range(n_draws):
            thb = th + Lc @ rng.standard_normal(len(th))
            ub, vmb, _, swb = efeitos(thb)
            ex_u = rng.normal(0, np.sqrt(tau2), self.n_l2) / np.sqrt(1.0 + swb * tau2)
            ex_m = rng.normal(0, np.sqrt(sig2), n)
            ex_m = np.where(obs_mun, ex_m * np.sqrt(max(1 - self.rho ** 2, 0.05)), ex_m)
            Apb, Bpb = projeta(thb, ub, vmb, ex_m, ex_u)
            s_u = 1 + rng.normal(0, 0.012, self.n_l2) * (1 - info_u)
            sims.append((A.sum() + (Apb * s_u[self.l2]).sum(), Bv.sum() + (Bpb * s_u[self.l2]).sum()))
        S = np.array(sims)
        ps = S[:, 0] / S.sum(1)
        sd_raw = float(ps.std())
        a, b, c = self.calib
        sd = float(np.sqrt((a * sd_raw) ** 2 + (b * out["pend"]) ** 2 + (c * pA * (1 - pA) * U) ** 2))
        out.update({"sd_raw": sd_raw, "sd": sd, "lo": pA - 1.645 * sd, "hi": pA + 1.645 * sd,
                    "prob_A": float(norm.cdf((pA - 0.5) / max(sd, 1e-9)))})
        return out
