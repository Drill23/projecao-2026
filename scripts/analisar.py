"""Calibra a incerteza do v2 no histórico e calcula as estatísticas v1 x v2.

Entrada: resultados/*.jsonl (gerados por scripts/avaliar.py)
Saída:   resultados/resumo.json (tabelas) e resultados/calibracao.json (a, b, c do v2)

Regras para não "colar na prova":
  - a calibração do v2 usa só 2010-2022; cada eleição desse período é avaliada com
    parâmetros ajustados SEM ela (validação deixando uma eleição de fora);
  - 2026 (presidente e governadores) é teste fora da amostra: usa a calibração de 2010-2022.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

Z90 = 1.6449


def carregar():
    L = []
    for p in sorted(glob.glob("resultados/reprise*.jsonl")):
        for linha in open(p):
            r = json.loads(linha)
            if "v1" in r and "v2" in r:
                L.append(r)
    return L


def sd_v2(r, a, b, c):
    v = r["v2"]
    p = v["pA"]
    return np.sqrt((a * v["sd_raw"]) ** 2 + (b * v["pend"]) ** 2 + (c * p * (1 - p) * v["U"]) ** 2)


def _arrays(L):
    p = np.array([r["v2"]["pA"] for r in L])
    return (p, np.array([r["v2"]["sd_raw"] for r in L]), np.array([r["v2"]["pend"] for r in L]),
            np.array([r["v2"]["U"] for r in L]) * p * (1 - p), np.array([r["final"] for r in L]))


def nll(theta, arr):
    a, b, c = np.exp(theta)
    p, s0, pend, u, f = arr
    s = np.maximum(np.sqrt((a * s0) ** 2 + (b * pend) ** 2 + (c * u) ** 2), 1e-5)
    return float(np.mean(-norm.logpdf(f, p, s)))


def calibrar(L):
    arr = _arrays(L)
    best = None
    for x0 in ([0.0, np.log(0.003), 0.0], [0.5, np.log(0.01), np.log(0.5)], [0.3, np.log(0.001), np.log(2.0)]):
        res = minimize(nll, x0, args=(arr,), method="Nelder-Mead", options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    return tuple(float(x) for x in np.exp(best.x))


def aplicar(L, calib_por_id):
    for r in L:
        a, b, c = calib_por_id[r["id"]]
        s = sd_v2(r, a, b, c)
        p = r["v2"]["pA"]
        r["v2"].update({"sd": s, "lo": p - Z90 * s, "hi": p + Z90 * s, "prob": float(norm.cdf((p - 0.5) / s))})


def metricas(L, modelo):
    if not L:
        return None
    if modelo == "bruto":
        e = np.array([100 * (r["bruto"] - r["final"]) for r in L])
        return {"n": len(L), "erro_medio_abs": float(np.abs(e).mean()), "erro_mediano_abs": float(np.median(np.abs(e))),
                "erro_p90_abs": float(np.percentile(np.abs(e), 90)), "acerto_vencedor": float(np.mean([(r["bruto"] > 0.5) for r in L]))}
    e = np.array([100 * (r[modelo]["pA"] - r["final"]) for r in L])
    dentro = np.array([r[modelo]["lo"] <= r["final"] <= r[modelo]["hi"] for r in L])
    prob = np.array([r[modelo]["prob"] for r in L])
    return {"n": len(L),
            "erro_medio_abs": float(np.abs(e).mean()),
            "erro_mediano_abs": float(np.median(np.abs(e))),
            "erro_p90_abs": float(np.percentile(np.abs(e), 90)),
            "erro_max_abs": float(np.abs(e).max()),
            "vies": float(e.mean()),
            "cobertura_90": float(dentro.mean()),
            "acerto_vencedor": float(np.mean([r[modelo]["pA"] > 0.5 for r in L])),
            "brier": float(np.mean((1 - prob) ** 2)),
            "confianca_errada": int(np.sum(prob < 0.05)),      # >95% de certeza no candidato errado
            "largura_media_faixa": float(np.mean([100 * (r[modelo]["hi"] - r[modelo]["lo"]) for r in L]))}


def cravadas(L, modelo, limiar=0.95):
    """Primeiro % apurado a partir do qual a chance fica >= limiar até o fim (por eleição/semente)."""
    por = defaultdict(list)
    for r in L:
        por[(r["id"], r["seed"])].append(r)
    res = []
    for (eid, s), rs in por.items():
        rs.sort(key=lambda r: r["cp"])
        cp_cravada = None
        for i, r in enumerate(rs):
            if all(x[modelo]["prob"] >= limiar for x in rs[i:]):
                cp_cravada = r["cp"]
                break
        errada = any(x[modelo]["prob"] <= 1 - limiar for x in rs)
        res.append({"id": eid, "seed": s, "cp": cp_cravada, "errada": errada,
                    "margem": abs(rs[0]["final"] - 0.5) * 200})
    return res


def main():
    L = carregar()
    ids = sorted({r["id"] for r in L})
    hist = [i for i in ids if "_2026" not in i]
    print(f"{len(L)} pontos, {len(ids)} eleições ({len(hist)} de 2010-2022)")
    # calibração: LOO no histórico; 2026 com tudo de 2010-2022
    calib = {}
    Lh = [r for r in L if r["id"] in hist]
    geral = calibrar(Lh)
    for eid in ids:
        if eid in hist:
            calib[eid] = calibrar([r for r in Lh if r["id"] != eid])
        else:
            calib[eid] = geral
        print(eid, np.round(calib[eid], 4), flush=True)
    aplicar(L, calib)
    json.dump({"a": geral[0], "b": geral[1], "c": geral[2], "base": "2010-2022, todas as eleições"},
              open("resultados/calibracao.json", "w"), indent=1)
    resumo = {"calibracao": geral, "eleicoes": ids, "por_checkpoint": {}, "geral": {}, "cravadas": {}}
    grupos = {
        "todas": lambda r: True,
        "historico_2010_2022": lambda r: r["ano"] < 2026,
        "teste_2026": lambda r: r["ano"] == 2026,
        "presidente": lambda r: r["cargo"] == "presidente",
        "governador": lambda r: r["cargo"] == "governador",
        "apertadas_ate_3pts": lambda r: abs(r["final"] - 0.5) * 200 <= 3,
    }
    for gname, f in grupos.items():
        Lg = [r for r in L if f(r)]
        resumo["geral"][gname] = {m: metricas(Lg, m) for m in ("bruto", "v1", "v2")}
        resumo["por_checkpoint"][gname] = {}
        for cp in sorted({r["cp"] for r in Lg}):
            Lc = [r for r in Lg if r["cp"] == cp]
            resumo["por_checkpoint"][gname][str(cp)] = {m: metricas(Lc, m) for m in ("bruto", "v1", "v2")}
        resumo["cravadas"][gname] = {m: cravadas(Lg, m) for m in ("v1", "v2")}
    # trajetórias de exemplo
    resumo["trajetorias"] = {}
    for eid in ids:
        rs = sorted([r for r in L if r["id"] == eid and r["seed"] == 0], key=lambda r: r["cp"])
        resumo["trajetorias"][eid] = [{"cp": r["cp"], "final": r["final"], "bruto": r["bruto"],
                                       "v1": {k: r["v1"][k] for k in ("pA", "lo", "hi", "prob")},
                                       "v2": {k: r["v2"][k] for k in ("pA", "lo", "hi", "prob")}} for r in rs]
    os.makedirs("resultados", exist_ok=True)
    json.dump(resumo, open("resultados/resumo.json", "w"), indent=1, ensure_ascii=False)
    # impressão rápida
    for gname in grupos:
        print("\n==", gname)
        for cp, d in resumo["por_checkpoint"][gname].items():
            if d["v1"] is None:
                continue
            print(f"{cp:>5}%  bruto {d['bruto']['erro_medio_abs']:.2f} | v1 erro {d['v1']['erro_medio_abs']:.2f} "
                  f"cob {d['v1']['cobertura_90']:.2f} conf.err {d['v1']['confianca_errada']} | "
                  f"v2 erro {d['v2']['erro_medio_abs']:.2f} cob {d['v2']['cobertura_90']:.2f} "
                  f"conf.err {d['v2']['confianca_errada']}  (n={d['v2']['n']})")


if __name__ == "__main__":
    main()
