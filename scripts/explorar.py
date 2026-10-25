"""Diagnóstico: roda 2 ciclos do robô (sem publicar) e mostra onde a leitura falha."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ao_vivo as av
from projecao.dados import Base2026
os.makedirs("exploracao", exist_ok=True)
base = Base2026(); est = av.Estado(base)
L = []
import time
for c in range(4):
    t0 = time.time(); out = est.ciclo(n_boot=20); dt_ = time.time() - t0
    falt = np.where((est.st_ab > 0) & (est.est <= 0))[0]
    L.append({"ciclo": c, "pct": out["apurado"]["pct_eleitorado"] if out else None,
              "mun_st_ab>0": int((est.st_ab > 0).sum()), "mun_st_res>0": int((est.st_res > 0).sum()),
              "est_sum": float(est.est.sum()), "est_ab_sum": float(est.est_ab.sum()),
              "faltando": [(base.mun_keys[i], est.st_ab[i], est.st_res[i]) for i in falt[:15]],
              "n_falt": len(falt), "erros": dict(av.ERROS), "orc": est.orcamento,
              "oficial": {k: est.oficial.get(k) for k in ("A", "B", "est", "pest")},
              "lidos": est.n_lidos, "gap_ufs": est.gap_ufs, "seg": round(dt_, 1),
              "uf_res_vs_mun": {u: (est.est_uf_res.get(u), float(est.est[est.uf_de_mun == u].sum())) for u in base.ufs if est.est_uf_res.get(u)},
              "proj": None if not (out and out["projecao"]) else {k: out["projecao"][k] for k in ("pA", "lo", "hi", "prob_A")},
              "bruto": out["apurado"]["pA"] if out else None})
    time.sleep(20)
# por UF: est_ab vs est
L.append({"por_uf": [(u, float(est.est_ab[[i for i, k in enumerate(base.mun_keys) if k[:2] == u]].sum()),
                      float(est.est[[i for i, k in enumerate(base.mun_keys) if k[:2] == u]].sum())) for u in base.ufs]})
# teste direto de um município do exterior
zz = [k for k in base.mun_keys if k[:2] == "zz"][:3]
for k in zz:
    d = av.get_json(f"{av.DIV}/{av.ELE}/dados/zz/{k}-c0001-e{av.E6}-u.json")
    L.append({"zz_teste": k, "ok": bool(d), "est": d["e"]["est"] if d else None})
json.dump(L, open("exploracao/diag2.json", "w"), indent=1, ensure_ascii=False, default=str)
for x in L: print(json.dumps(x, ensure_ascii=False, default=str)[:900])
