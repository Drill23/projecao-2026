"""Carrega a base por seção e monta as matrizes do modelo para o 2º turno de 2026."""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.special import logit

from .modelo import Modelo, _bincount

REG_IDX = {"N": 0, "NE": 1, "CO": 2, "SE": 3, "S": 4, "EX": 5}
GRUPOS = ["L", "F", "O1", "O2", "O3", "NE", "ND"]
# priores para a fração dos votos de cada grupo que vai para Lula (A) no 2º turno.
# Cury, Caiado, Zema e Barão declararam apoio a Flávio; UP, PSTU, PCB e PCO a Lula.
PRIOR_Q = {"O1": (0.30, 0.5), "O2": (0.25, 0.5), "O3": (0.30, 0.5), "NE": (0.80, 0.5), "ND": (0.25, 0.5)}
# covariáveis de 2022 por seção: testadas na reprise (com 2018 no papel de 2022) e não melhoraram a
# projeção; ficam desligadas.
USAR_COVARIAVEIS = False


def _media_imputada(x, ok, peso, grupos_list):
    """Preenche x onde ok=False com a média ponderada do primeiro grupo que tiver dados."""
    x = x.copy()
    falta = ~ok
    for g in grupos_list:
        if not falta.any():
            break
        n = g.max() + 1
        s = _bincount(g, np.where(ok, x * peso, 0), n)
        w = _bincount(g, np.where(ok, peso, 0), n)
        m = np.where(w > 0, s / np.maximum(w, 1e-9), np.nan)
        cand = m[g]
        usar = falta & ~np.isnan(cand)
        x[usar] = cand[usar]
        falta = falta & ~usar
    x[falta] = 0.0
    return x


class Base2026:
    def __init__(self, pasta="base"):
        z = np.load(os.path.join(pasta, "base.npz"))
        b = {k: z[k] for k in z.files}
        self.meta = json.load(open(os.path.join(pasta, "meta.json"), encoding="utf-8"))
        sel = b["no_t2"]
        self.b = {k: v[sel] for k, v in b.items()}
        b = self.b
        self.n = int(sel.sum())
        self.ufs = self.meta["ufs"]
        self.mun_keys = [m["k"] for m in self.meta["municipios"]]
        self.n_mun = len(self.mun_keys)
        self.mun_idx = {k: i for i, k in enumerate(self.mun_keys)}
        self.uf = b["uf"].astype(np.int64)
        self.mun = b["mun"].astype(np.int64)
        self.reg = np.array([REG_IDX[self.meta["regiao"][u]] for u in self.ufs])[self.uf]
        self.G = np.stack([b[g] for g in GRUPOS], 1).astype(float)
        self.C1 = b["QT_COMPARECIMENTO"].astype(float)
        self.E = b["QT_APTOS"].astype(float)
        # chave de seção -> índice (para marcar as apuradas a partir do arquivo de seções)
        self.chave = {}
        for i, (u, m, z_, s) in enumerate(zip(self.uf, self.mun, b["zona"], b["secao"])):
            self.chave[(self.mun_keys[m], int(z_), int(s))] = i
        # covariáveis de 2022 (seção com a mesma chave)
        ok = b["tem22"] & (b["x22_L2"] + b["x22_B2"] > 20) & (b["x22_L1"] + b["x22_B1"] > 20)
        L1, B1 = b["x22_L1"].astype(float), b["x22_B1"].astype(float)
        L2, B2 = b["x22_L2"].astype(float), b["x22_B2"].astype(float)
        d22 = np.log((L2 + .5) / (B2 + .5)) - np.log((L1 + .5) / (B1 + .5))
        lean22 = logit(np.clip((L2 + .5) / (L2 + B2 + 1), 0.01, 0.99))
        peso = np.maximum(self.G.sum(1), 1)
        d22 = _media_imputada(np.where(ok, d22, 0), ok, peso, [self.mun, self.uf])
        lean22 = _media_imputada(np.where(ok, lean22, 0), ok, peso, [self.mun, self.uf])
        lean22 = lean22 - np.average(lean22, weights=peso)
        d22 = d22 - np.average(d22, weights=peso)
        if USAR_COVARIAVEIS:
            self.X = np.stack([d22, lean22], 1)
            self.nomes_cov = ["swing 1º→2º turno em 2022", "inclinação 2022"]
        else:
            self.X = np.zeros((self.n, 0))
            self.nomes_cov = []

    def modelo(self, rho=0.85):
        q = [PRIOR_Q[g][0] for g in GRUPOS[2:]]
        s = [PRIOR_Q[g][1] for g in GRUPOS[2:]]
        m = Modelo(self.G, self.C1, self.E, self.mun, self.uf, self.reg, self.X, q, s,
                   n_mun=self.n_mun, n_uf=len(self.ufs), n_reg=6, nomes_cov=self.nomes_cov,
                   rho=rho, prior_turn=1.0, prior_valid=0.95)
        # calibração da faixa (reprise de 2022 com ordem real de chegada), com folga para 2026
        m.infla, m.piso = 1.5, 0.001
        return m
