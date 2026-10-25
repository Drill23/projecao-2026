"""Explora os arquivos da divulgação do TSE a partir do runner do GitHub.

Salva amostras em exploracao/ para inspecionar o formato real dos arquivos de 2026.
"""
import json
import os
import sys
import time
import urllib.request

BASE = "https://resultados.tse.jus.br/oficial"
CDN = "https://cdn.tse.jus.br/estatistica/sead/odsele"
OUT = "exploracao"
os.makedirs(OUT, exist_ok=True)
LOG = []


def get(url, limit=None, save=None):
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 projecao-2026"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read() if limit is None else r.read(limit)
            hdr = {k: v for k, v in r.headers.items() if k.lower() in (
                "content-type", "content-length", "last-modified", "etag",
                "access-control-allow-origin", "cache-control", "date")}
            LOG.append({"url": url, "status": r.status, "bytes": len(data),
                        "s": round(time.time() - t0, 2), "headers": hdr})
            if save:
                with open(os.path.join(OUT, save), "wb") as f:
                    f.write(data)
            return data
    except Exception as e:  # noqa: BLE001
        LOG.append({"url": url, "erro": repr(e)[:300], "s": round(time.time() - t0, 2)})
        return None


def head(url):
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            LOG.append({"url": url, "HEAD": r.status,
                        "len": r.headers.get("content-length"),
                        "lm": r.headers.get("last-modified")})
    except Exception as e:  # noqa: BLE001
        LOG.append({"url": url, "HEAD_erro": repr(e)[:200]})


def main():
    raw = get(f"{BASE}/comum/config/ele-c.json", save="ele-c.json")
    ciclos = []
    if raw:
        cfg = json.loads(raw)
        ciclos = cfg.get("c", cfg) if isinstance(cfg, dict) else cfg
    # tenta achar eleições de 2026 de forma genérica
    texto = raw.decode("utf-8", "replace") if raw else ""
    eles = set()
    for tok in ("544", "545", "546", "547", "548", "549", "550", "551", "552", "553",
                "619", "620", "621", "622", "623", "624", "625", "626"):
        if f'"{tok}"' in texto or f":{tok}" in texto:
            eles.add(tok)
    # extrai todos os códigos de eleição listados no ciclo 2026
    try:
        cfg = json.loads(raw)
        for c in cfg.get("c", []):
            if "2026" in json.dumps(c)[:200]:
                for p in c.get("p", []):
                    for e in p.get("e", []):
                        eles.add(str(e.get("cd")))
    except Exception:  # noqa: BLE001
        pass
    LOG.append({"eleicoes_detectadas": sorted(eles)})

    for ele in sorted(eles):
        e6 = f"{int(ele):06d}"
        get(f"{BASE}/ele2026/{ele}/config/mun-e{e6}-cm.json", save=f"mun-e{e6}-cm.json")
        get(f"{BASE}/ele2026/{ele}/dados-simplificados/br/br-c0001-e{e6}-r.json",
            save=f"br-c0001-e{e6}-r.json")
        get(f"{BASE}/ele2026/{ele}/dados-simplificados/sp/sp-c0001-e{e6}-r.json",
            save=f"sp-c0001-e{e6}-r.json")
        get(f"{BASE}/ele2026/{ele}/dados-simplificados/sp/sp71072-c0001-e{e6}-r.json",
            save=f"sp71072-c0001-e{e6}-r.json")
        get(f"{BASE}/ele2026/{ele}/dados/sp/sp71072-c0001-e{e6}-v.json",
            save=f"dados-sp71072-c0001-e{e6}-v.json")
        get(f"{BASE}/ele2026/{ele}/dados/sp/sp-c0001-e{e6}-v.json",
            save=f"dados-sp-c0001-e{e6}-v.json")
        get(f"{BASE}/ele2026/{ele}/dados/br/br-c0001-e{e6}-v.json",
            save=f"dados-br-c0001-e{e6}-v.json")
        get(f"{BASE}/ele2026/{ele}/dados/ac/ac-e{e6}-ab.json", save=f"ac-e{e6}-ab.json")
        get(f"{BASE}/ele2026/{ele}/dados/br/br-e{e6}-ab.json", save=f"br-e{e6}-ab.json")

    # arquivo-urna (pleitos): tenta descobrir pelo config
    pleitos = set()
    try:
        cfg = json.loads(raw)
        for c in cfg.get("c", []):
            for p in c.get("p", []):
                pleitos.add(str(p.get("cd")))
    except Exception:  # noqa: BLE001
        pass
    LOG.append({"pleitos_detectados": sorted(pleitos)})
    for pl in sorted(pleitos):
        p6 = f"{int(pl):06d}"
        get(f"{BASE}/ele2026/arquivo-urna/{pl}/config/ac/ac-p{p6}-cs.json",
            save=f"ac-p{p6}-cs.json")
        get(f"{BASE}/ele2026/arquivo-urna/{pl}/config/mu-e{p6}-cm.json",
            save=f"au-mu-p{p6}-cm.json")

    for ano in (2026, 2022):
        for uf in ("AC", "BR"):
            head(f"{CDN}/votacao_secao/votacao_secao_{ano}_{uf}.zip")
        head(f"{CDN}/detalhe_votacao_secao/detalhe_votacao_secao_{ano}.zip")
        head(f"{CDN}/eleitorado_locais_votacao/eleitorado_local_votacao_{ano}.zip")
    head(f"{CDN}/votacao_candidato_munzona/votacao_candidato_munzona_2026.zip")

    with open(os.path.join(OUT, "log.json"), "w") as f:
        json.dump(LOG, f, indent=1, ensure_ascii=False)
    for item in LOG:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
