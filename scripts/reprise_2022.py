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

REG_IDX = {"N": 0, "NE": 1, "CO": 2, "SE": 3, "S": 4, "EX": 5}


def carregar():
    z = np.load("base/base.npz")
    meta = json.load(open("base/meta.json"))
    return {k: z[k] for k in z.files}, meta


def main(n_boot=60, seeds=(0,), checkpoints=(0.5, 1, 2, 3, 5, 8, 12, 20, 35, 50, 70, 90)):
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

    cheg = b["cheg1"][m].astype(float)
    resultados = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ch = cheg.copy()
        falt = ch < 0
        # sem horário: sorteia dentro da distribuição da UF
        for u in np.unique(uf[falt]):
            pool = ch[(uf == u) & ~falt]
            sel = falt & (uf == u)
            ch[sel] = rng.choice(pool, sel.sum()) if len(pool) else rng.uniform(0, 300, sel.sum())
        # pequeno ruído para variar a ordem entre sementes
        ch = ch + rng.normal(0, 3 if seed else 0.01, len(ch))
        ordem = np.argsort(ch)
        cumE = np.cumsum(E[ordem]) / E.sum()
        mod = Modelo(G, C1, E, mun, uf, reg, np.zeros((len(E), 0)),
                     q_prior=[0.6, 0.55, 0.45], q_sd=[1.0, 1.0, 1.0],
                     n_mun=n_mun, n_uf=len(meta["ufs"]), n_reg=6, prior_turn=1.0, prior_valid=0.95)
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
                               "q": o["q"]})
            print(f"seed {seed} {cp:5.1f}% apurado | bruto {100*bruto:6.2f} | proj {100*o['pA']:6.2f} "
                  f"[{100*o['pA_lo']:.2f}, {100*o['pA_hi']:.2f}] erro {100*(o['pA']-final):+.2f} pp "
                  f"{'ok' if dentro else 'FORA'} | q={np.round(o['q'],2)} sig={np.sqrt(o['sig2']):.3f} "
                  f"tau={np.sqrt(o['tau2']):.3f} ({time.time()-t0:.1f}s)", flush=True)
    json.dump({"final": final, "res": resultados}, open("reprise_2022.json", "w"), indent=1)


if __name__ == "__main__":
    main()
