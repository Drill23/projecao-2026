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
    for ele in ("6257", "6258"):
        e = f"{int(ele):06d}"
        get(f"{BASE}/{ele}/config/mun-e{e}-cm.json", f"mun-e{e}-cm.json")
        for uf in ("br", "ac", "sp"):
            get(f"{BASE}/{ele}/dados-simplificados/{uf}/{uf}-c0001-e{e}-r.json", f"simpl-{uf}-e{e}.json")
            get(f"{BASE}/{ele}/dados/{uf}/{uf}-e{e}-ab.json", f"ab-{uf}-e{e}.json")
            get(f"{BASE}/{ele}/dados/{uf}/{uf}-e{e}-u.json", f"u-{uf}-e{e}.json")
            get(f"{BASE}/{ele}/dados/{uf}/{uf}-c0001-e{e}-v.json", f"v-{uf}-e{e}.json")
            get(f"{BASE}/{ele}/dados/{uf}/{uf}-c0001-e{e}-f.json", f"f-{uf}-e{e}.json")
        # município de Rio Branco (AC) = 01392 ; São Paulo = 71072
        for mun in ("ac01392", "sp71072"):
            get(f"{BASE}/{ele}/dados-simplificados/{mun[:2]}/{mun}-c0001-e{e}-r.json", f"simpl-{mun}-e{e}.json")
            get(f"{BASE}/{ele}/dados/{mun[:2]}/{mun}-c0001-e{e}-v.json", f"v-{mun}-e{e}.json")
            get(f"{BASE}/{ele}/dados/{mun[:2]}/{mun}-c0001-e{e}-f.json", f"f-{mun}-e{e}.json")
    for pl in ("3220", "3221"):
        p = f"{int(pl):06d}"
        raw = get(f"{BASE}/arquivo-urna/{pl}/config/ac/ac-p{p}-cs.json", f"cs-ac-p{p}.json", keep=400_000)
        get(f"{BASE}/arquivo-urna/{pl}/config/sp/sp-p{p}-cs.json", f"cs-sp-p{p}.json", keep=300_000)
        get(f"{BASE}/arquivo-urna/{pl}/config/mu-p{p}-cm.json", f"mu-p{p}-cm.json")
        if raw:
            try:
                d = json.loads(raw)
                LOG.append({"cs_keys": list(d.keys())})
                # tenta achar a primeira seção para buscar o aux
                txt = json.dumps(d)
                LOG.append({"cs_inicio": txt[:1500]})
                ab = d["abr"][0]
                mu = ab["mu"][0]
                zo = mu["zon"][0]
                se = zo["sec"][0]
                m, z, s = mu["cd"], zo["cd"], se["ns"]
                get(f"{BASE}/arquivo-urna/{pl}/dados/ac/{m}/{z}/{s}/p{p}-ac-m{m}-z{z}-s{s}-aux.json",
                    f"aux-ac-{m}-{z}-{s}-p{p}.json")
            except Exception as ex:  # noqa: BLE001
                LOG.append({"cs_parse_erro": repr(ex)[:300]})

    with open(os.path.join(OUT, "log2.json"), "w") as f:
        json.dump(LOG, f, indent=1, ensure_ascii=False)
    for item in LOG:
        print(json.dumps(item, ensure_ascii=False)[:500])


if __name__ == "__main__":
    sys.exit(main())
