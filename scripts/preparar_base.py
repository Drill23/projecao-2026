"""Monta a base por seção para a noite de um 2º turno (presidente ou governador).

Uso: python scripts/preparar_base.py eleicoes/presidente-2026.json

Lê os dados abertos do TSE do 1º turno (votos por seção, aptos e comparecimento), a lista
oficial de seções do 2º turno e os municípios; agrupa os candidatos do 1º turno em
A, B (os finalistas, pela configuração), os 3 maiores eliminados e o resto; e define a
hierarquia geográfica do modelo:
  presidente: nível 1 = região, nível 2 = UF;
  governador: nível 1 e nível 2 = grupos de municípios vizinhos (pela posição geográfica).
Saída: base/{id}.npz e base/{id}.json
"""
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from historico import CDN, baixar, chave, ler  # noqa: E402
from projecao.base import REG, REGIOES, UFS  # noqa: E402

DIV = "https://resultados.tse.jus.br/oficial"


def get_json(url):
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 projecao-eleitoral"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            print("falha", url, e, flush=True)
    return None


def centroides(ano, ufs):
    """lat/lon média dos locais de votação de cada município (ponderada pelo eleitorado)."""
    for a in (ano, 2026, 2024, 2022):
        try:
            z = baixar(f"{CDN}/eleitorado_locais_votacao/eleitorado_local_votacao_{a}.zip", f"locais{a}.zip")
            break
        except Exception:  # noqa: BLE001
            continue
    partes = []
    with zipfile.ZipFile(z) as zf:
        for n in [n for n in zf.namelist() if n.lower().endswith(".csv")]:
            with zf.open(n) as fh:
                cab = fh.readline().decode("latin1").strip().replace('"', "").split(";")
            lat = [c for c in cab if "LATITUDE" in c][0]
            lon = [c for c in cab if "LONGITUDE" in c][0]
            el = [c for c in cab if c.startswith("QT_ELEITOR")]
            with zf.open(n) as fh:
                df = pd.read_csv(fh, sep=";", encoding="latin1", dtype=str,
                                 usecols=["SG_UF", "CD_MUNICIPIO", lat, lon] + el[:1])
            df = df[df.SG_UF.str.upper().isin(ufs)]
            df = df.rename(columns={lat: "lat", lon: "lon"})
            df["w"] = pd.to_numeric(df[el[0]], errors="coerce").fillna(1) if el else 1.0
            partes.append(df)
    df = pd.concat(partes)
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c].str.replace(",", "."), errors="coerce")
    df = df[(df.lat > -40) & (df.lat < 10) & (df.lon < -25) & (df.lon > -80)]
    df["k"] = df.SG_UF.str.upper() + df.CD_MUNICIPIO.str.zfill(5)
    g = df.groupby("k").apply(lambda x: (np.average(x.lat, weights=x.w), np.average(x.lon, weights=x.w)))
    return g.to_dict()


def main(caminho):
    cfg = json.load(open(caminho, encoding="utf-8"))
    ano, cargo, uf = cfg["ano"], str(cfg["cargo"]), cfg["uf"].upper()
    pres = cargo == "1"
    ufs = UFS if pres else [uf]
    A, B = int(cfg["A"]["numero"]), int(cfg["B"]["numero"])
    # 1º turno por seção
    arq = f"votacao_secao_{ano}_BR.zip" if pres else f"votacao_secao_{ano}_{uf}.zip"
    zv = baixar(f"{CDN}/votacao_secao/{arq}", arq)
    v = ler(zv, ["NR_TURNO", "CD_CARGO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "NR_VOTAVEL",
                 "NM_VOTAVEL", "QT_VOTOS"], lambda c: c[(c.CD_CARGO == cargo) & (c.NR_TURNO == "1")])
    v["QT_VOTOS"] = v.QT_VOTOS.astype(np.int64)
    v["NR_VOTAVEL"] = v.NR_VOTAVEL.astype(int)
    v["k"] = chave(v)
    bn = [95, 96, 97]
    tot = v[~v.NR_VOTAVEL.isin(bn + [A, B])].groupby("NR_VOTAVEL").QT_VOTOS.sum().sort_values(ascending=False)
    outros = [int(x) for x in tot.index[:3]]
    nomes = v.groupby("NR_VOTAVEL").NM_VOTAVEL.first().to_dict()
    g = {A: "A1", B: "B1", **{n: f"O{i+1}" for i, n in enumerate(outros)}}
    v["g"] = v.NR_VOTAVEL.map(g).fillna("OR")
    v.loc[v.NR_VOTAVEL.isin(bn), "g"] = "BN1"
    p1 = v.pivot_table(index="k", columns="g", values="QT_VOTOS", aggfunc="sum", fill_value=0)
    del v
    zd = baixar(f"{CDN}/detalhe_votacao_secao/detalhe_votacao_secao_{ano}.zip", f"dv{ano}.zip")
    d = ler(zd, ["NR_TURNO", "CD_CARGO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "QT_APTOS",
                 "QT_COMPARECIMENTO"], lambda c: c[(c.CD_CARGO == cargo) & (c.NR_TURNO == "1")
                                                   & c.SG_UF.str.upper().isin(ufs)])
    d["k"] = chave(d)
    d = d.groupby("k")[["QT_APTOS", "QT_COMPARECIMENTO"]].first().astype(np.int64)
    # seções oficiais do 2º turno
    secoes = []
    for u in ufs:
        cs = get_json(f"{DIV}/{cfg['ciclo']}/arquivo-urna/{cfg['pleito_2t']}/config/{u.lower()}/"
                      f"{u.lower()}-p{int(cfg['pleito_2t']):06d}-cs.json")
        if not cs:
            continue
        for ab in cs["abr"]:
            for mu in ab["mu"]:
                for zo in mu["zon"]:
                    for se in zo["sec"]:
                        secoes.append(f"{u}|{mu['cd']}|{zo['cd']}|{se['ns']}")
    idx = pd.Index(secoes) if secoes else p1.index
    t = pd.DataFrame(index=idx).join(p1).join(d).fillna(0)
    for c in ["A1", "B1", "O1", "O2", "O3", "OR", "BN1"]:
        if c not in t.columns:
            t[c] = 0
    print("seções do 2º turno:", len(t), "com base no 1º turno:", int(t.index.isin(p1.index).sum()), flush=True)
    parts = t.index.to_series().str.split("|", expand=True)
    mkeys = (parts[0] + parts[1].str.zfill(5)).to_numpy()
    umun, mun = np.unique(mkeys, return_inverse=True)
    # municípios: nome, código IBGE, capital
    cm = get_json(f"{DIV}/{cfg['ciclo']}/{cfg['eleicao_2t']}/config/mun-e{int(cfg['eleicao_2t']):06d}-cm.json") or {"abr": []}
    info = {}
    for ab in cm.get("abr", []):
        for mu in ab.get("mu", []):
            info[ab["cd"].upper() + mu["cd"]] = {"nm": mu.get("nm", ""), "cdi": mu.get("cdi", ""), "c": mu.get("c", "n")}
    E = t.QT_APTOS.to_numpy(float)
    Em = np.bincount(mun, weights=E, minlength=len(umun))
    # hierarquia
    if pres:
        l1 = np.array([REG[k[:2]] for k in umun])
        l2 = np.array([UFS.index(k[:2]) for k in umun])
        l1_nomes, l2_nomes = REGIOES, UFS
    else:
        geo = centroides(ano, ufs)
        xy = np.array([geo.get(k, (np.nan, np.nan)) for k in umun], float)
        falta = np.isnan(xy).any(1)
        xy[falta] = np.nanmean(xy[~falta], 0) if (~falta).any() else 0
        k1, k2 = max(1, min(4, len(umun) // 10)), max(1, min(12, len(umun) // 5))
        _, l1 = kmeans2(xy, k1, seed=1, minit="++") if k1 > 1 else (None, np.zeros(len(umun), int))
        _, l2 = kmeans2(xy, k2, seed=2, minit="++") if k2 > 1 else (None, np.zeros(len(umun), int))
        # nome de cada grupo = maior município do grupo
        def nomes_grupos(lab):
            out = []
            for gi in range(int(lab.max()) + 1):
                ms = np.where(lab == gi)[0]
                top = ms[np.argmax(Em[ms])] if len(ms) else None
                out.append(f"Região de {info.get(umun[top], {}).get('nm', umun[top]).title()}" if top is not None else f"Grupo {gi+1}")
            return out
        l1_nomes, l2_nomes = nomes_grupos(l1), nomes_grupos(l2)
    os.makedirs("base", exist_ok=True)
    arr = {"mun": mun.astype(np.int32), "zona": parts[2].astype(int).to_numpy(np.int16),
           "secao": parts[3].astype(int).to_numpy(np.int16),
           "APT": t.QT_APTOS.to_numpy(np.int32), "COMP1": t.QT_COMPARECIMENTO.to_numpy(np.int32)}
    for c in ["A1", "B1", "O1", "O2", "O3", "OR", "BN1"]:
        arr[c] = t[c].to_numpy(np.int32)
    np.savez_compressed(f"base/{cfg['id']}.npz", **arr)
    meta = {"id": cfg["id"], "cargo": "presidente" if pres else "governador", "uf": uf, "ano": ano,
            "A": {"n": A, "nome": nomes.get(A, cfg["A"]["nome"])}, "B": {"n": B, "nome": nomes.get(B, cfg["B"]["nome"])},
            "outros": [{"n": n, "nome": nomes.get(n, str(n))} for n in outros],
            "l1_nomes": list(l1_nomes), "l2_nomes": list(l2_nomes),
            "municipios": [{"k": k, "l1": int(l1[i]), "l2": int(l2[i]), "E": float(Em[i]),
                            **info.get(k, {"nm": k, "cdi": "", "c": "n"})} for i, k in enumerate(umun)],
            "totais_1t": {c: int(t[c].sum()) for c in ["A1", "B1", "O1", "O2", "O3", "OR", "BN1", "QT_APTOS", "QT_COMPARECIMENTO"]}}
    json.dump(meta, open(f"base/{cfg['id']}.json", "w"), ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("A", "B", "outros", "l1_nomes", "totais_1t")}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
