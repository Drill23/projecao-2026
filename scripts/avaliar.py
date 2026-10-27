"""Reprise de 2º turnos passados: compara o modelo v1 (o da noite de 25/10/2026) com o v2.

Para cada eleição (presidente ou governador) e cada semente:
  - ordena as seções por um horário de chegada realista (horários reais de 2026);
  - libera as seções em fatias (0,5% ... 90% do eleitorado) e roda os dois modelos;
  - grava projeção, faixa, chance e os ingredientes da incerteza, mais o resultado final.

Uso: python scripts/avaliar.py --eleicoes pres_2022,gov_2022_SP --seeds 2 --saida resultados/x.jsonl
     python scripts/avaliar.py --shard 0/8 ...  (divide a lista de eleições entre máquinas)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.cluster.vq import kmeans2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projecao.modelo import Modelo  # noqa: E402
from projecao.modelo_v2 import ModeloV2, _bc  # noqa: E402

UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB",
       "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO", "ZZ"]
REG = {"AC": 0, "AM": 0, "AP": 0, "PA": 0, "RO": 0, "RR": 0, "TO": 0, "AL": 1, "BA": 1, "CE": 1,
       "MA": 1, "PB": 1, "PE": 1, "PI": 1, "RN": 1, "SE": 1, "DF": 2, "GO": 2, "MS": 2, "MT": 2,
       "ES": 3, "MG": 3, "RJ": 3, "SP": 3, "PR": 4, "RS": 4, "SC": 4, "ZZ": 5}
CHECK = [0.5, 1, 2, 3, 5, 10, 20, 30, 50, 70, 90]
GRUPOS = ["A1", "B1", "O1", "O2", "O3", "OR"]


# ------------------------------------------------------------------ horários de chegada
class Chegada:
    """Sorteia horários de chegada a partir dos horários reais medidos em 2026."""

    def __init__(self, arquivo_amostra):
        z = np.load("base/base.npz")
        meta = json.load(open("base/meta.json"))
        mk = [m["k"].upper() for m in meta["municipios"]]
        am = np.load(arquivo_amostra)
        idx, mins = am["idx"], am["minutos"]
        sel = z["no_t2"]
        # índices da amostra referem-se às seções do 2º turno (base filtrada por no_t2)
        pos = np.where(sel)[0]
        mun = z["mun"][pos][idx]
        zon = z["zona"][pos][idx]
        sec = z["secao"][pos][idx]
        E = z["QT_APTOS"][pos]
        Em = np.bincount(z["mun"][pos], weights=E, minlength=len(mk))
        self.chave = {}
        self.por_mun, self.por_uf_porte, self.por_uf = {}, {}, {}
        for m, zz, s, t in zip(mun, zon, sec, mins):
            k = mk[m]
            self.chave[(k, int(zz), int(s))] = float(t)
            self.por_mun.setdefault(k, []).append(float(t))
            b = self.porte(Em[m])
            self.por_uf_porte.setdefault((k[:2], b), []).append(float(t))
            self.por_uf.setdefault(k[:2], []).append(float(t))
        for d in (self.por_mun, self.por_uf_porte, self.por_uf):
            for k in d:
                d[k] = np.array(d[k])
        self.todos = np.array(mins, float)

    @staticmethod
    def porte(e):
        return int(np.digitize(np.log10(max(e, 1)), [3.7, 4.2, 4.7, 5.2, 5.7]))

    def horarios(self, chaves_mun, zona, secao, E_mun_sec, rng, ruido):
        n = len(zona)
        t = np.full(n, np.nan)
        for i in range(n):
            k = chaves_mun[i]
            v = self.chave.get((k, int(zona[i]), int(secao[i])))
            if v is not None:
                t[i] = v + (rng.normal(0, ruido) if ruido else 0)
                continue
            pool = self.por_mun.get(k)
            if pool is None or len(pool) < 3:
                pool = self.por_uf_porte.get((k[:2], self.porte(E_mun_sec[i])))
            if pool is None or len(pool) < 3:
                pool = self.por_uf.get(k[:2], self.todos)
            t[i] = rng.choice(pool) + rng.normal(0, 4)
        return t


# ------------------------------------------------------------------ uma eleição
def carregar(eid, geo):
    z = np.load(f"historico/{eid}.npz")
    meta = json.load(open(f"historico/{eid}.json"))
    ok = (z["APT2"] > 0) & (z["uf"] >= 0)
    d = {k: z[k][ok] for k in z.files}
    ufs = np.array(UFS)[d["uf"]]
    chaves = np.char.add(ufs, np.char.zfill(d["mun"].astype(str), 5))
    umun, mun = np.unique(chaves, return_inverse=True)
    n_mun = len(umun)
    E = d["APT2"].astype(float)
    Em = _bc(mun, E, n_mun)
    G = np.stack([d[g] for g in GRUPOS], 1).astype(float)
    uf_mun = np.array([UFS.index(k[:2]) for k in umun])
    reg_mun = np.array([REG[k[:2]] for k in umun])
    pres = meta["cargo"] == "presidente"
    # hierarquia v2
    if pres:
        l1_mun, l2_mun, n_l1, n_l2 = reg_mun, uf_mun, 6, 28
    else:
        xy = np.array([geo.get(k, [np.nan, np.nan]) for k in umun], float)
        falta = np.isnan(xy).any(1)
        if falta.all():
            xy[:] = 0
        else:
            xy[falta] = np.nanmean(xy[~falta], 0)
        xy = xy + np.random.default_rng(0).normal(0, 1e-4, xy.shape)
        k1, k2 = max(1, min(4, n_mun // 10)), max(1, min(12, n_mun // 5))
        _, c1 = kmeans2(xy, k1, seed=1, minit="++") if k1 > 1 else (None, np.zeros(n_mun, int))
        _, c2 = kmeans2(xy, k2, seed=2, minit="++") if k2 > 1 else (None, np.zeros(n_mun, int))
        l1_mun, l2_mun = c1.astype(int), c2.astype(int)
        n_l1, n_l2 = int(l1_mun.max()) + 1, int(l2_mun.max()) + 1
    return {"meta": meta, "d": d, "mun": mun, "umun": umun, "n_mun": n_mun, "E": E, "Em": Em, "G": G,
            "uf_mun": uf_mun, "reg_mun": reg_mun, "l1_mun": l1_mun, "l2_mun": l2_mun, "n_l1": n_l1,
            "n_l2": n_l2, "pres": pres}


VARIANTES = {
    "v2": {},
    "v2_semN": {"mobiliza": False},
    "v2_semtau": {"k_tau": 0.0},
    "v2_base": {"mobiliza": False, "k_tau": 0.0},
}


def modelo_v2(e, **kw):
    d, mun = e["d"], e["mun"]
    sd_l1 = None
    if e["pres"]:
        sd_l1 = np.full(6, 0.08)
        sd_l1[5] = 0.35
    return ModeloV2(e["G"], d["COMP1"], e["E"], mun, e["l2_mun"][mun], e["l1_mun"][mun], e["n_mun"],
                    e["n_l2"], e["n_l1"], sd_l1_unidade=sd_l1, calib=(1.0, 0.0, 0.0), **kw)


def modelos(e):
    d, mun = e["d"], e["mun"]
    K2 = e["G"].shape[1] - 2
    m1 = Modelo(e["G"], d["COMP1"], e["E"], mun, e["uf_mun"][mun], e["reg_mun"][mun], np.zeros((len(mun), 0)),
                q_prior=[0.5] * K2, q_sd=[0.5] * K2, n_mun=e["n_mun"], n_uf=28, n_reg=6)
    m1.infla, m1.piso = 1.5, 0.001       # como rodou ao vivo
    sd_l1 = None
    if e["pres"]:
        sd_l1 = np.full(6, 0.08)
        sd_l1[5] = 0.35
    m2 = ModeloV2(e["G"], d["COMP1"], e["E"], mun, e["l2_mun"][mun], e["l1_mun"][mun], e["n_mun"],
                  e["n_l2"], e["n_l1"], sd_l1_unidade=sd_l1, calib=(1.0, 0.0, 0.0))
    return m1, m2


def reprise(eid, seeds, geo, chegadas, n_draws, variantes=("v2",), com_v1=True):
    e = carregar(eid, geo)
    d, mun, n_mun = e["d"], e["mun"], e["n_mun"]
    final = float(d["A2"].sum() / (d["A2"].sum() + d["B2"].sum()))
    ch = chegadas["2t"] if e["meta"]["ano"] == 2026 else chegadas["1t"]
    te = _bc(mun, e["E"], n_mun)
    linhas = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        t = ch.horarios(e["umun"][mun], d["zona"], d["secao"], e["Em"][mun], rng, 0 if seed == 0 else 1.5)
        ordem = np.argsort(t, kind="stable")
        cumE = np.cumsum(e["E"][ordem]) / e["E"].sum()
        m1, _ = modelos(e)
        m2s = {v: modelo_v2(e, **VARIANTES[v]) for v in variantes}
        for cp in CHECK:
            k = int(np.searchsorted(cumE, cp / 100)) + 1
            cont = np.zeros(len(mun), bool)
            cont[ordem[:k]] = True
            obs = {"A": _bc(mun, d["A2"] * cont, n_mun), "B": _bc(mun, d["B2"] * cont, n_mun),
                   "comp": _bc(mun, d["COMP2"] * cont, n_mun), "est": _bc(mun, e["E"] * cont, n_mun), "te": te}
            bruto = float(obs["A"].sum() / max(obs["A"].sum() + obs["B"].sum(), 1))
            lin = {"id": eid, "cargo": e["meta"]["cargo"], "ano": e["meta"]["ano"], "uf": e["meta"]["uf"],
                   "seed": seed, "cp": cp, "final": final, "bruto": bruto}
            t0 = time.time()
            if com_v1:
                try:
                    o1 = m1.projetar(cont, obs, n_boot=n_draws, seed=seed)
                except Exception as ex:  # noqa: BLE001
                    o1 = {"ok": False, "erro": repr(ex)[:100]}
                if o1.get("ok"):
                    lin["v1"] = {"pA": o1["pA"], "lo": o1["pA_lo"], "hi": o1["pA_hi"], "sd": o1["pA_sd"],
                                 "prob": o1["prob_A"]}
            t1 = time.time()
            for v, m2 in m2s.items():
                try:
                    o2 = m2.projetar(cont, obs, n_draws=n_draws, seed=seed)
                except Exception as ex:  # noqa: BLE001
                    o2 = {"ok": False, "erro": repr(ex)[:100]}
                if o2.get("ok"):
                    lin[v] = {k: o2[k] for k in ("pA", "sd_raw", "U", "pend", "qN", "tau2")}
            lin["s"] = [round(t1 - t0, 2), round(time.time() - t1, 2)]
            linhas.append(lin)
            v1 = lin.get("v1", {}).get("pA")
            v2 = lin.get("v2", {}).get("pA")
            print(f"{eid} s{seed} {cp:5.1f}% final {100*final:.2f} bruto {100*bruto:.2f} "
                  f"v1 {None if v1 is None else round(100*v1, 2)} v2 {None if v2 is None else round(100*v2, 2)} "
                  f"({lin['s']})", flush=True)
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eleicoes", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--draws", type=int, default=80)
    ap.add_argument("--saida", default="resultados/reprise.jsonl")
    ap.add_argument("--variantes", default="v2")
    ap.add_argument("--sem-v1", action="store_true")
    a = ap.parse_args()
    ids = a.eleicoes.split(",") if a.eleicoes else sorted(os.path.basename(p)[:-4] for p in glob.glob("historico/*_*.npz")
                                                          if not os.path.basename(p).startswith("chegada"))
    i, n = map(int, a.shard.split("/"))
    # distribui pelo tamanho (presidente pesa mais)
    ids = sorted(ids, key=lambda x: (not x.startswith("pres"), x))
    ids = ids[i::n]
    geo = json.load(open("historico/municipios_geo.json")) if os.path.exists("historico/municipios_geo.json") else {}
    geo = {k.upper(): v for k, v in geo.items()}
    chegadas = {"1t": Chegada("base/chegada_amostra.npz")}
    chegadas["2t"] = Chegada("historico/chegada_2t_2026.npz") if os.path.exists("historico/chegada_2t_2026.npz") else chegadas["1t"]
    os.makedirs(os.path.dirname(a.saida) or ".", exist_ok=True)
    with open(a.saida, "w") as f:
        for eid in ids:
            t0 = time.time()
            for lin in reprise(eid, range(a.seeds), geo, chegadas, a.draws, a.variantes.split(","), not a.sem_v1):
                f.write(json.dumps(lin) + "\n")
            f.flush()
            print(f"== {eid} em {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
