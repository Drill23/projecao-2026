"""Escolhe a variante do v2, calibra a incerteza no histórico e calcula as estatísticas v1 x v2.

Entrada: resultados/reprise-*.jsonl (v1 e v2) e, se existirem, resultados/ablacao-*.jsonl
         (variantes do v2: sem o termo de votos novos, sem o prior de variância...)
Saída:   resultados/resumo.json, resultados/calibracao.json, resultados/ablacao.json

Regras para não "colar na prova":
  - a variante e a calibração do v2 são escolhidas só com 2010-2022; cada eleição desse
    período é avaliada com coeficientes ajustados SEM ela (validação deixando uma de fora);
  - 2026 (presidente e governadores) é teste fora da amostra.
  - O DF tem um município só (o TSE divulga um resultado só), então não dá para projetar por
    município; fica fora.
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
VARS = ["v2", "v2_semN", "v2_semtau", "v2_base"]


def carregar():
    base = {}
    for p in sorted(glob.glob("resultados/reprise*.jsonl")):
        for linha in open(p):
            r = json.loads(linha)
            base[(r["id"], r["seed"], r["cp"])] = r
    for p in sorted(glob.glob("resultados/ablacao*.jsonl")):
        for linha in open(p):
            r = json.loads(linha)
            k = (r["id"], r["seed"], r["cp"])
            if k in base:
                for v in VARS:
                    if v in r:
                        base[k][v] = r[v]
    return [r for r in base.values() if "v1" in r and "v2" in r]


# ------------------------------------------------------------------ incerteza
def _arrays(L, var):
    p = np.array([r[var]["pA"] for r in L])
    return {"p": p, "s0": np.array([r[var]["sd_raw"] for r in L]), "pend": np.array([r[var]["pend"] for r in L]),
            "u": np.array([r[var]["U"] for r in L]) * p * (1 - p), "lpct": np.log10(np.array([r["cp"] for r in L])),
            "pres": np.array([r["cargo"] == "presidente" for r in L], float), "f": np.array([r["final"] for r in L]),
            "w": _pesos(L)}


def _pesos(L):
    """Cada fatia de % apurado pesa o mesmo (senão as fatias finais, mais numerosas, dominam)."""
    cnt = defaultdict(int)
    for r in L:
        cnt[r["cp"]] += 1
    w = np.array([1.0 / cnt[r["cp"]] for r in L])
    return w / w.mean()


def sd_de(k, a):
    m = np.exp(k[0] + k[1] * a["lpct"] + k[2] * a["pres"])
    return np.maximum(np.sqrt((m * a["s0"]) ** 2 + (k[3] * a["pend"]) ** 2 + (k[4] * a["u"]) ** 2 + k[5] ** 2), 1e-5)


def nll(theta, a):
    k = [theta[0], theta[1], theta[2], np.exp(theta[3]), np.exp(theta[4]), np.exp(theta[5])]
    return float(np.sum(a["w"] * -norm.logpdf(a["f"], a["p"], sd_de(k, a))) / a["w"].sum())


def calibrar(L, var):
    a = _arrays(L, var)
    best = None
    for x0 in ([0.5, -0.3, 0.0, np.log(0.003), np.log(0.5), np.log(2e-4)],
               [1.0, -0.5, 0.0, np.log(0.01), np.log(2.0), np.log(5e-4)],
               [0.0, -0.8, 0.3, np.log(0.003), np.log(0.01), np.log(1e-4)]):
        res = minimize(nll, x0, args=(a,), method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    t = best.x
    return {"alfa": float(t[0]), "beta": float(t[1]), "gama": float(t[2]), "b": float(np.exp(t[3])),
            "c": float(np.exp(t[4])), "d": float(np.exp(t[5]))}


def aplicar(L, var, calib_por_id, destino="v2"):
    for r in L:
        k = calib_por_id[r["id"]]
        a = _arrays([r], var)
        s = float(sd_de([k["alfa"], k["beta"], k["gama"], k["b"], k["c"], k["d"]], a)[0])
        p = r[var]["pA"]
        r[destino] = dict(r[var], sd=s, lo=p - Z90 * s, hi=p + Z90 * s, prob=float(norm.cdf((p - 0.5) / s)))


# ------------------------------------------------------------------ métricas
def metricas(L, m):
    if not L:
        return None
    if m == "bruto":
        e = np.array([100 * (r["bruto"] - r["final"]) for r in L])
        return {"n": len(L), "erro_medio_abs": float(np.abs(e).mean()), "erro_mediano_abs": float(np.median(np.abs(e))),
                "erro_p90_abs": float(np.percentile(np.abs(e), 90)),
                "acerto_vencedor": float(np.mean([(r["bruto"] > 0.5) == (r["final"] > 0.5) for r in L]))}
    e = np.array([100 * (r[m]["pA"] - r["final"]) for r in L])
    dentro = np.array([r[m]["lo"] <= r["final"] <= r[m]["hi"] for r in L])
    prob = np.array([r[m]["prob"] for r in L])
    venc = np.array([r["final"] > 0.5 for r in L], float)
    return {"n": len(L), "erro_medio_abs": float(np.abs(e).mean()), "erro_mediano_abs": float(np.median(np.abs(e))),
            "erro_p90_abs": float(np.percentile(np.abs(e), 90)), "erro_max_abs": float(np.abs(e).max()),
            "vies": float(e.mean()), "cobertura_90": float(dentro.mean()),
            "acerto_vencedor": float(np.mean((np.array([r[m]["pA"] for r in L]) > 0.5) == (venc > 0.5))),
            "brier": float(np.mean((prob - venc) ** 2)),
            "confianca_errada": int(np.sum(np.abs(prob - venc) > 0.95)),
            "largura_media_faixa": float(np.mean([100 * (r[m]["hi"] - r[m]["lo"]) for r in L]))}


def cravadas(L, m, limiar=0.95):
    por = defaultdict(list)
    for r in L:
        por[(r["id"], r["seed"])].append(r)
    res = []
    for (eid, s), rs in por.items():
        rs.sort(key=lambda r: r["cp"])
        venc_A = rs[0]["final"] > 0.5
        pv = [x[m]["prob"] if venc_A else 1 - x[m]["prob"] for x in rs]
        cp = None
        for i in range(len(rs)):
            if all(p >= limiar for p in pv[i:]):
                cp = rs[i]["cp"]
                break
        res.append({"id": eid, "seed": s, "cp": cp, "errada": any(p <= 1 - limiar for p in pv),
                    "margem": abs(rs[0]["final"] - 0.5) * 200})
    return res


def resumo_cravadas(cr):
    cps = [c["cp"] for c in cr if c["cp"] is not None]
    return {"n": len(cr), "cravou": len(cps), "mediana_cp": float(np.median(cps)) if cps else None,
            "erradas": int(sum(c["errada"] for c in cr))}


def main():
    L = carregar()
    ids = sorted({r["id"] for r in L})
    hist = [i for i in ids if "_2026" not in i]
    Lh = [r for r in L if r["id"] in hist]
    print(f"{len(L)} pontos, {len(ids)} eleições ({len(hist)} de 2010-2022)")
    # 1) escolha da variante por cargo (erro médio no histórico, fatias com peso igual)
    disponiveis = [v for v in VARS if all(v in r for r in L)]
    abl = {}
    escolha = {}
    for cargo in ("presidente", "governador"):
        Lc = [r for r in Lh if r["cargo"] == cargo]
        w = _pesos(Lc)
        abl[cargo] = {}
        for v in disponiveis:
            e = np.array([abs(r[v]["pA"] - r["final"]) * 100 for r in Lc])
            abl[cargo][v] = {"erro_medio_ponderado": float(np.sum(w * e) / w.sum()),
                             "por_cp": {str(cp): float(np.mean([abs(r[v]["pA"] - r["final"]) * 100 for r in Lc if r["cp"] == cp]))
                                        for cp in sorted({r["cp"] for r in Lc})}}
        escolha[cargo] = min(abl[cargo], key=lambda v: abl[cargo][v]["erro_medio_ponderado"])
        print(cargo, {v: round(x["erro_medio_ponderado"], 3) for v, x in abl[cargo].items()}, "->", escolha[cargo])
    for r in L:
        r["v2e"] = r[escolha[r["cargo"]]]
    # 2) calibração (LOO no histórico; 2026 com tudo de 2010-2022)
    geral = calibrar(Lh, "v2e")
    calib = {}
    for eid in ids:
        calib[eid] = calibrar([r for r in Lh if r["id"] != eid], "v2e") if eid in hist else geral
    aplicar(L, "v2e", calib, "v2")
    print("calibração geral:", {k: round(v, 4) for k, v in geral.items()})
    opcoes = {"v2": {}, "v2_semN": {"mobiliza": False}, "v2_semtau": {"k_tau": 0.0},
              "v2_base": {"mobiliza": False, "k_tau": 0.0}}
    json.dump({**geral, "variante": {c: opcoes[v] for c, v in escolha.items()}, "variante_nome": escolha,
               "base": "2010-2022 (sem o DF)"}, open("resultados/calibracao.json", "w"), indent=1)
    json.dump(abl, open("resultados/ablacao.json", "w"), indent=1)
    # 3) estatísticas
    grupos = {
        "todas": lambda r: True,
        "historico_2010_2022": lambda r: r["ano"] < 2026,
        "teste_2026": lambda r: r["ano"] == 2026,
        "presidente": lambda r: r["cargo"] == "presidente",
        "governador": lambda r: r["cargo"] == "governador",
        "apertadas_ate_3pts": lambda r: abs(r["final"] - 0.5) * 200 <= 3,
    }
    res = {"calibracao": geral, "variante": escolha, "eleicoes": ids, "n_eleicoes": len(ids),
           "por_checkpoint": {}, "geral": {}, "cravadas": {}, "cravadas_resumo": {}}
    for g, f in grupos.items():
        Lg = [r for r in L if f(r)]
        res["geral"][g] = {m: metricas(Lg, m) for m in ("bruto", "v1", "v2")}
        res["por_checkpoint"][g] = {str(cp): {m: metricas([r for r in Lg if r["cp"] == cp], m) for m in ("bruto", "v1", "v2")}
                                    for cp in sorted({r["cp"] for r in Lg})}
        res["cravadas"][g] = {m: cravadas(Lg, m) for m in ("v1", "v2")}
        res["cravadas_resumo"][g] = {m: resumo_cravadas(res["cravadas"][g][m]) for m in ("v1", "v2")}
        res["n_eleicoes_" + g] = len({r["id"] for r in Lg})
    res["trajetorias"] = {}
    for eid in ids:
        rs = sorted([r for r in L if r["id"] == eid and r["seed"] == 0], key=lambda r: r["cp"])
        res["trajetorias"][eid] = [{"cp": r["cp"], "final": r["final"], "bruto": r["bruto"],
                                    "v1": {k: r["v1"][k] for k in ("pA", "lo", "hi", "prob")},
                                    "v2": {k: r["v2"][k] for k in ("pA", "lo", "hi", "prob")}} for r in rs]
    json.dump(res, open("resultados/resumo.json", "w"), indent=1, ensure_ascii=False)
    for g in grupos:
        print(f"\n== {g} ({res['n_eleicoes_' + g]} eleições) cravadas v1 {res['cravadas_resumo'][g]['v1']} "
              f"v2 {res['cravadas_resumo'][g]['v2']}")
        for cp, d in res["por_checkpoint"][g].items():
            print(f"{cp:>5}%  bruto {d['bruto']['erro_medio_abs']:.2f} | v1 erro {d['v1']['erro_medio_abs']:.2f} "
                  f"cob {d['v1']['cobertura_90']:.2f} brier {d['v1']['brier']:.3f} | v2 erro {d['v2']['erro_medio_abs']:.2f} "
                  f"cob {d['v2']['cobertura_90']:.2f} brier {d['v2']['brier']:.3f} (n={d['v2']['n']})")


if __name__ == "__main__":
    main()
