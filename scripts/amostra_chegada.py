"""Amostra o horário real de recebimento dos boletins de urna no 1º turno de 2026.

O arquivo de configuração de seções guarda só horários de regeração em lote; o horário
verdadeiro de chegada de cada urna está no arquivo auxiliar da seção (campo hr do hash).
Busca uma amostra aleatória de seções para dar uma ordem de chegada realista à reprise.
"""
import concurrent.futures as cf
import json
import os
import random
import sys
import time
import urllib.request

import numpy as np

PLEITO = os.environ.get("PLEITO", "3220")
DIV = f"https://resultados.tse.jus.br/oficial/ele2026/arquivo-urna/{PLEITO}/dados"
FRAC = float(os.environ.get("FRAC", "0.12"))


def get_json(url):
    for t in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 projecao-2026"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1 + t)
        except Exception:  # noqa: BLE001
            time.sleep(1 + t)
    return None


def main():
    z = np.load("base/base.npz")
    meta = json.load(open("base/meta.json"))
    ok = z["no_t2"] & z["tem26"] & (z["QT_APTOS"] > 0)
    idx = np.where(ok)[0]
    random.seed(7)
    amostra = sorted(random.sample(list(idx), int(len(idx) * FRAC)))
    print("seções na amostra:", len(amostra), flush=True)
    muns = meta["municipios"]

    def f(i):
        mk = muns[int(z["mun"][i])]["k"]
        uf, cd = mk[:2], mk[2:]
        zo, se = f"{int(z['zona'][i]):04d}", f"{int(z['secao'][i]):04d}"
        d = get_json(f"{DIV}/{uf}/{cd}/{zo}/{se}/p{int(PLEITO):06d}-{uf}-m{cd}-z{zo}-s{se}-aux.json")
        if not d:
            return i, None
        melhor = None
        for h in d.get("hashes", []):
            if not h.get("dr") or not h.get("hr"):
                continue
            dd, mm, aa = h["dr"].split("/")
            hh, mi, ss = h["hr"].split(":")
            minutos = (int(dd) - int(os.environ.get("DIA", "4"))) * 1440 + int(hh) * 60 + int(mi) + int(ss) / 60 - 17 * 60
            melhor = minutos if melhor is None else min(melhor, minutos)
        return i, melhor

    t0 = time.time()
    res = {}
    with cf.ThreadPoolExecutor(48) as ex:
        for n, (i, m) in enumerate(ex.map(f, amostra)):
            if m is not None:
                res[int(i)] = m
            if n % 5000 == 0:
                print(n, len(res), f"{time.time()-t0:.0f}s", flush=True)
    ii = np.array(sorted(res), dtype=np.int64)
    mm = np.array([res[i] for i in ii], dtype=np.float32)
    np.savez_compressed(os.environ.get("SAIDA", "base/chegada_amostra.npz"), idx=ii, minutos=mm)
    print("ok", len(ii), "percentis (min após 17h):", np.percentile(mm, [1, 10, 50, 90, 99]).round(1),
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
