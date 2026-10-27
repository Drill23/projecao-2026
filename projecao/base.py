"""Base por seção de uma eleição (formato v2) e montagem do modelo v2 a partir dela.

A base é gerada por scripts/preparar_base.py a partir de um arquivo de configuração em
eleicoes/*.json e fica em base/{id}.npz + base/{id}.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .modelo_v2 import ModeloV2

UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB",
       "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO", "ZZ"]
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul", "Exterior"]
REG = {"AC": 0, "AM": 0, "AP": 0, "PA": 0, "RO": 0, "RR": 0, "TO": 0, "AL": 1, "BA": 1, "CE": 1,
       "MA": 1, "PB": 1, "PE": 1, "PI": 1, "RN": 1, "SE": 1, "DF": 2, "GO": 2, "MS": 2, "MT": 2,
       "ES": 3, "MG": 3, "RJ": 3, "SP": 3, "PR": 4, "RS": 4, "SC": 4, "ZZ": 5}
GRUPOS = ["A1", "B1", "O1", "O2", "O3", "OR"]


def carregar_config(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def calibracao_padrao():
    """Coeficientes da incerteza calibrados no histórico (resultados/calibracao.json)."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resultados", "calibracao.json")
    if os.path.exists(p):
        c = json.load(open(p))
        if "alfa" in c:
            return {k: c.get(k, 0.0) for k in ("alfa", "beta", "gama", "b", "c", "d")}
    return {"alfa": 0.1, "beta": -0.8, "gama": 0.3, "b": 0.003, "c": 0.0, "d": 0.0}


def variante_padrao(cargo):
    """Variante do v2 escolhida na avaliação para cada cargo (resultados/calibracao.json)."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resultados", "calibracao.json")
    if os.path.exists(p):
        return json.load(open(p)).get("variante", {}).get(cargo, {})
    return {}


class BaseEleicao:
    def __init__(self, cfg, pasta="base"):
        self.cfg = cfg
        z = np.load(os.path.join(pasta, f"{cfg['id']}.npz"))
        self.meta = json.load(open(os.path.join(pasta, f"{cfg['id']}.json"), encoding="utf-8"))
        b = {k: z[k] for k in z.files}
        self.b = b
        self.n = len(b["mun"])
        self.mun_keys = [m["k"] for m in self.meta["municipios"]]
        self.n_mun = len(self.mun_keys)
        self.mun_idx = {k: i for i, k in enumerate(self.mun_keys)}
        self.mun = b["mun"].astype(np.int64)
        self.G = np.stack([b[g] for g in GRUPOS], 1).astype(float)
        self.C1 = b["COMP1"].astype(float)
        self.E = b["APT"].astype(float)
        l1m = np.array([m["l1"] for m in self.meta["municipios"]], np.int64)
        l2m = np.array([m["l2"] for m in self.meta["municipios"]], np.int64)
        self.l1, self.l2 = l1m[self.mun], l2m[self.mun]
        self.n_l1, self.n_l2 = len(self.meta["l1_nomes"]), len(self.meta["l2_nomes"])
        self.chave = {(self.mun_keys[m], int(zz), int(s)): i
                      for i, (m, zz, s) in enumerate(zip(self.mun, b["zona"], b["secao"]))}
        self.ufs = sorted({k[:2] for k in self.mun_keys})

    def modelo(self):
        cfg = self.cfg
        outros = self.meta["outros"]
        pri = cfg.get("outros_prior") or {}
        q = [pri.get(str(o["n"]), 0.5) for o in outros] + [pri.get("resto", 0.5)]
        q_sd = [0.5 if str(o["n"]) in pri else 0.7 for o in outros] + [0.7]
        sd_l1 = None
        if self.meta["cargo"] == "presidente":
            sd_l1 = np.full(self.n_l1, 0.08)
            if "Exterior" in self.meta["l1_nomes"]:
                sd_l1[self.meta["l1_nomes"].index("Exterior")] = 0.35
        prior_nat = (0.0, 0.6)
        pesq = cfg.get("pesquisa")
        if pesq:
            # converte a pesquisa (A em % dos válidos) num prior para o efeito nacional,
            # medido a partir do 1º turno com as transferências do prior
            G = self.G
            A0 = G[:, 0].sum() + (G[:, 2:] * np.array(q)).sum()
            B0 = G[:, 1].sum() + (G[:, 2:] * (1 - np.array(q))).sum()
            from scipy.special import logit
            prior_nat = (float(logit(pesq["pA"]) - np.log(A0 / B0)), float(pesq.get("sd", 0.025) * 4.0))
        return ModeloV2(self.G, self.C1, self.E, self.mun, self.l2, self.l1, self.n_mun, self.n_l2, self.n_l1,
                        q_prior=q, q_sd=q_sd, sd_l1_unidade=sd_l1, prior_nat=prior_nat,
                        calib=calibracao_padrao(), presidente=self.meta["cargo"] == "presidente",
                        **variante_padrao(self.meta["cargo"]))
