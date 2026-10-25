"""Reprise de 2022: prevê o 2º turno de 2022 usando o 1º turno de 2022 como base,
liberando as seções na ordem real de chegada do 1º turno de 2026 (mesma chave de seção).

Mede, a cada fatia de % apurado, o erro da projeção vs. o resultado final, e a cobertura
da faixa de 90%.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from projecao.modelo import Modelo, _bincount  # noqa: E402

QSD = float(__import__("os").environ.get("QSD", "0.5"))
REG_IDX = {"N": 0, "NE": 1, "CO": 2, "SE": 3, "S": 4, "EX": 5}


def carregar():
    z = np.load("base/base.npz")
    meta = json.load(open("base/meta.json"))
    return {k: z[k] for k in z.files}, meta


def covs18(b, m, mun, peso):
    from scipy.special import logit
    from projecao.dados import _media_imputada
    ok = b["tem18"][m] & (b["x18_L2"][m] + b["x18_B2"][m] > 20) & (b["x18_L1"][m] + b["x18_B1"][m] > 20)
    L1, B1 = b["x18_L1"][m].astype(float), b["x18_B1"][m].astype(float)
    L2, B2 = b["x18_L2"][m].astype(float), b["x18_B2"][m].astype(float)
    d = np.log((L2 + .5) / (B2 + .5)) - np.log((L1 + .5) / (B1 + .5))
    lean = logit(np.clip((L2 + .5) / (L2 + B2 + 1), 0.01, 0.99))
    uf = b["uf"][m].astype(np.int64)
    d = _media_imputada(np.where(ok, d, 0), ok, peso, [mun, uf])
    lean = _media_imputada(np.where(ok, lean, 0), ok, peso, [mun, uf])
    d -= np.average(d, weights=peso)
    lean -= np.average(lean, weights=peso)
    return np.stack([d, lean], 1)


def main(n_boot=60, seeds=(0,), checkpoints=(0.5, 1, 2, 3, 5, 8, 12, 20, 35, 50, 70, 90),
         usar_cov=False, rho=0.85, saida="reprise_2022.json", verbose=True):
    b, meta = carregar()
    m = b["tem22"] & (b["APT22"] > 0)
    print("seções na reprise:", int(m.sum()))
    regiao = np.array([REG_IDX[meta["regiao"][u]] for u in meta["ufs"]])
    uf = b["uf"][m].astype(np.int64)
    mun_old = b["mun"][m].astype(np.int64)
    # reindexa municípios
    umun, mun = np.unique(mun_old, return_inverse=True)
    n_mun = len(umun)
    G = np.stack([b["x22_L1"][m], b["x22_B1"][m], b["x22_T1"][m], b["x22_C1"][m], b["x22_O1"][m]], 1)
    C1 = b["COMP22a"][m].astype(float)
    E = b["APT22"][m].astype(float)
    L2 = b["x22_L2"][m].astype(float)
    B2 = b["x22_B2"][m].astype(float)
    comp2 = b["COMP22b"][m].astype(float)
    reg = regiao[uf]
    final = L2.sum() / (L2.sum() + B2.sum())
    print(f"resultado final (Lula, % válidos A+B): {100*final:.3f}")

    # horários reais de chegada (amostra dos arquivos auxiliares do 1º turno de 2026)
    am = np.load("base/chegada_amostra.npz")
    pos = np.full(len(b["uf"]), -1.0)
    pos[am["idx"]] = am["minutos"]
    cheg = pos[m]
    Emun = _bincount(mun, E, n_mun)
    faixa = np.digitize(np.log10(np.maximum(Emun, 1)), [3.7, 4.2, 4.7, 5.2, 5.7])[mun]   # porte do município
    resultados = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ch = cheg.copy()
        tem = ch >= 0
        # 1) sorteia entre os horários amostrados do mesmo município (se houver 2 ou mais)
        cnt = _bincount(mun, tem * 1.0, n_mun)
        ordm = np.argsort(mun[tem], kind="stable")
        tempos_m = ch[tem][ordm]
        inicio = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        falta = ~tem & (cnt[mun] >= 2)
        r = rng.random(falta.sum())
        j = inicio[mun[falta]] + (r * cnt[mun[falta]]).astype(np.int64)
        ch[falta] = tempos_m[j] + rng.normal(0, 4, falta.sum())
        # 2) senão, entre os da mesma UF e porte de município
        falta = ch < 0
        for u in np.unique(uf[falta]):
            for f in np.unique(faixa[falta & (uf == u)]):
                sel = falta & (uf == u) & (faixa == f)
                pool = cheg[tem & (uf == u) & (faixa == f)]
                if len(pool) == 0:
                    pool = cheg[tem & (uf == u)]
                ch[sel] = rng.choice(pool, sel.sum()) + rng.normal(0, 4, sel.sum())
        # 3) nas seções com horário real, pequeno ruído diferente por semente
        ch[tem] = ch[tem] + rng.normal(0, 1.5 if seed else 0.0, tem.sum())
        ordem = np.argsort(ch)
        cumE = np.cumsum(E[ordem]) / E.sum()
        X = covs18(b, m, mun, np.maximum(G.sum(1), 1)) if usar_cov else np.zeros((len(E), 0))
        mod = Modelo(G, C1, E, mun, uf, reg, X, rho=rho,
                     q_prior=[0.55, 0.6, 0.35], q_sd=[QSD, QSD, QSD],
                     n_mun=n_mun, n_uf=len(meta["ufs"]), n_reg=6, prior_turn=1.0, prior_valid=0.95)
        mod.infla = float(__import__("os").environ.get("INFLA", "1"))
        mod.piso = float(__import__("os").environ.get("PISO", "0"))
        te = _bincount(mun, E, n_mun)
        for cp in checkpoints:
            k = int(np.searchsorted(cumE, cp / 100))
            cont = np.zeros(len(E), bool)
            cont[ordem[:k]] = True
            obs = {"A": _bincount(mun, L2 * cont, n_mun), "B": _bincount(mun, B2 * cont, n_mun),
                   "comp": _bincount(mun, comp2 * cont, n_mun),
                   "est": _bincount(mun, E * cont, n_mun), "te": te}
            t0 = time.time()
            o = mod.projetar(cont, obs, n_boot=n_boot, seed=seed)
            bruto = obs["A"].sum() / max(obs["A"].sum() + obs["B"].sum(), 1)
            if not o.get("ok"):
                print(f"{cp:5.1f}% sem projeção")
                continue
            dentro = o["pA_lo"] <= final <= o["pA_hi"]
            resultados.append({"seed": seed, "pct": cp, "proj": o["pA"], "lo": o["pA_lo"], "hi": o["pA_hi"],
                               "bruto": bruto, "erro_pp": 100 * (o["pA"] - final), "dentro": bool(dentro),
                               "q": o["q"], "sd": o["pA_sd"]})
            if verbose: print(f"seed {seed} {cp:5.1f}% apurado | bruto {100*bruto:6.2f} | proj {100*o['pA']:6.2f} "
                  f"[{100*o['pA_lo']:.2f}, {100*o['pA_hi']:.2f}] erro {100*(o['pA']-final):+.2f} pp "
                  f"{'ok' if dentro else 'FORA'} | q={np.round(o['q'],2)} sig={np.sqrt(o['sig2']):.3f} "
                  f"tau={np.sqrt(o['tau2']):.3f} ({time.time()-t0:.1f}s)", flush=True)
    json.dump({"final": final, "res": resultados}, open(saida, "w"), indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in resultados:
        agg[r["pct"]].append(r)
    print(f"--- cov={usar_cov} rho={rho}")
    for cp, rs in sorted(agg.items()):
        e = np.array([r["erro_pp"] for r in rs])
        w = np.array([100 * (r["hi"] - r["lo"]) / 2 for r in rs])
        print(f"{cp:5.1f}%  erro médio {e.mean():+.2f}  |erro| {np.abs(e).mean():.2f}  max {np.abs(e).max():.2f}"
              f"  meia-faixa {w.mean():.2f}  cobertura {np.mean([r['dentro'] for r in rs]):.2f}")
    return resultados


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", action="store_true")
    ap.add_argument("--rho", type=float, default=0.85)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--boot", type=int, default=60)
    ap.add_argument("-q", action="store_true")
    a = ap.parse_args()
    main(n_boot=a.boot, seeds=tuple(range(a.seeds)), usar_cov=a.cov, rho=a.rho, verbose=not a.q)
