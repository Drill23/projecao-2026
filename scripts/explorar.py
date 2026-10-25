"""Explora os arquivos da divulgação do TSE a partir do runner do GitHub (rodada 2)."""
import json
import os
import sys
import time
import urllib.request

BASE = "https://resultados.tse.jus.br/oficial/ele2026"
OUT = "exploracao"
os.makedirs(OUT, exist_ok=True)
LOG = []


def get(url, save=None, keep=200_000):
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 projecao-2026",
                                                   "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            hdr = {k: v for k, v in r.headers.items() if k.lower() in (
                "content-length", "last-modified", "access-control-allow-origin", "cache-control")}
            LOG.append({"url": url, "status": r.status, "bytes": len(data),
                        "s": round(time.time() - t0, 2), "headers": hdr})
            if save:
                with open(os.path.join(OUT, save), "wb") as f:
                    f.write(data[:keep])
            return data
    except Exception as e:  # noqa: BLE001
        LOG.append({"url": url, "erro": repr(e)[:200]})
        return None


def main():
    pats = []
    for ele in ("6257", "6258"):
        e = f"{int(ele):06d}"
        for uf, mun in (("br", ""), ("sp", ""), ("ac", ""), ("ac", "01392"), ("sp", "71072")):
            for suf in ("u", "e", "r", "v", "t"):
                for c in ("c0001-", ""):
                    pats.append((f"{BASE}/{ele}/dados/{uf}/{uf}{mun}-{c}e{e}-{suf}.json",
                                 f"res-{uf}{mun}-{c}e{e}-{suf}.json"))
    for url, nm in pats:
        get(url, nm, keep=60_000)
    # 2022: códigos de pleito/eleição para replay
    get("https://resultados.tse.jus.br/oficial/ele2022/arquivo-urna/407/config/ac/ac-p000407-cs.json", "cs-ac-p000407.json", keep=20000)
    get("https://resultados.tse.jus.br/oficial/ele2022/arquivo-urna/406/config/ac/ac-p000406-cs.json", "cs-ac-p000406.json", keep=20000)
    with open(os.path.join(OUT, "log3.json"), "w") as f:
        json.dump(LOG, f, indent=1, ensure_ascii=False)
    for item in LOG:
        if "erro" not in item:
            print(json.dumps(item, ensure_ascii=False)[:300])


if __name__ == "__main__":
    sys.exit(main())
