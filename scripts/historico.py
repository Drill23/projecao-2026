"""Monta, para um ano, as bases por seção de todos os 2º turnos (presidente e governadores).

Uso: python scripts/historico.py ANO
Saída: historico/{pres|gov}_{ANO}[_{UF}].npz + historico/{id}.json

Cada base junta, seção por seção (mesma UF/município/zona/seção nos dois turnos):
  1º turno: votos de A e B (os dois finalistas), dos 3 maiores eliminados (O1..O3), do resto (OR),
            brancos+nulos (BN1), aptos e comparecimento;
  2º turno: votos de A e B, brancos+nulos (BN2), aptos e comparecimento.
"""
import json
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

CDN = "https://cdn.tse.jus.br/estatistica/sead/odsele"
TMP = os.environ.get("TMPDIR_HIST", "/tmp/hist")
OUT = "historico"
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB",
       "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO", "ZZ"]


def baixar(url, nome):
    caminho = os.path.join(TMP, nome)
    if os.path.exists(caminho) and os.path.getsize(caminho) > 0:
        return caminho
    t0 = time.time()
    for tent in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 projecao-eleitoral"})
            with urllib.request.urlopen(req, timeout=900) as r, open(caminho, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            break
        except Exception as e:  # noqa: BLE001
            print("falha", url, e, flush=True)
            time.sleep(10 * (tent + 1))
    print(f"baixado {nome} {os.path.getsize(caminho)/1e6:.0f} MB {time.time()-t0:.0f}s", flush=True)
    return caminho


def ler(zip_path, cols, filtro):
    partes = []
    with zipfile.ZipFile(zip_path) as z:
        for n in [n for n in z.namelist() if n.lower().endswith(".csv")]:
            if "_BRASIL" in n.upper():
                continue
            with z.open(n) as fh:
                cab = fh.readline().decode("latin1").strip().replace('"', "").split(";")
            usar = [c for c in cols if c in cab]
            with z.open(n) as fh:
                for ch in pd.read_csv(fh, sep=";", encoding="latin1", usecols=usar, dtype=str,
                                      chunksize=3_000_000):
                    ch = filtro(ch)
                    if len(ch):
                        partes.append(ch)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=cols)


def chave(df):
    return (df["SG_UF"].str.upper() + "|" + df["CD_MUNICIPIO"].str.zfill(5) + "|"
            + df["NR_ZONA"].str.zfill(4) + "|" + df["NR_SECAO"].str.zfill(4))


COLS_V = ["NR_TURNO", "CD_CARGO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "NR_VOTAVEL",
          "NM_VOTAVEL", "QT_VOTOS"]
COLS_D = ["NR_TURNO", "CD_CARGO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "QT_APTOS",
          "QT_COMPARECIMENTO"]


def montar(eid, v, d, ano, cargo, uf):
    """v: votos por seção (ambos os turnos, um cargo); d: aptos/comparecimento."""
    v = v.copy()
    v["QT_VOTOS"] = v["QT_VOTOS"].astype(np.int64)
    v["NR_VOTAVEL"] = v["NR_VOTAVEL"].astype(int)
    v["NR_TURNO"] = v["NR_TURNO"].astype(int)
    v["k"] = chave(v)
    t1, t2 = v[v.NR_TURNO == 1], v[v.NR_TURNO == 2]
    if t2.empty:
        return None
    bn = [95, 96, 97]
    fin = t2[~t2.NR_VOTAVEL.isin(bn)].groupby("NR_VOTAVEL").QT_VOTOS.sum().sort_values(ascending=False)
    if len(fin) < 2:
        return None
    A, B = int(fin.index[0]), int(fin.index[1])      # A = vencedor no 2º turno
    nomes = t1.groupby("NR_VOTAVEL").NM_VOTAVEL.first().to_dict()
    nomes.update(t2.groupby("NR_VOTAVEL").NM_VOTAVEL.first().to_dict())
    tot1 = t1[~t1.NR_VOTAVEL.isin(bn + [A, B])].groupby("NR_VOTAVEL").QT_VOTOS.sum().sort_values(ascending=False)
    outros = [int(x) for x in tot1.index[:3]]
    g1 = {A: "A1", B: "B1"}
    for i, n in enumerate(outros):
        g1[n] = f"O{i+1}"
    t1 = t1.assign(g=t1.NR_VOTAVEL.map(g1).fillna("OR"))
    t1.loc[t1.NR_VOTAVEL.isin(bn), "g"] = "BN1"
    p1 = t1.pivot_table(index="k", columns="g", values="QT_VOTOS", aggfunc="sum", fill_value=0)
    t2 = t2.assign(g=np.where(t2.NR_VOTAVEL == A, "A2", np.where(t2.NR_VOTAVEL == B, "B2",
                              np.where(t2.NR_VOTAVEL.isin(bn), "BN2", "X2"))))
    p2 = t2.pivot_table(index="k", columns="g", values="QT_VOTOS", aggfunc="sum", fill_value=0)
    d = d.copy()
    d["k"] = chave(d)
    d[["QT_APTOS", "QT_COMPARECIMENTO"]] = d[["QT_APTOS", "QT_COMPARECIMENTO"]].astype(np.int64)
    d1 = d[d.NR_TURNO == "1"].groupby("k")[["QT_APTOS", "QT_COMPARECIMENTO"]].first()
    d2 = d[d.NR_TURNO == "2"].groupby("k")[["QT_APTOS", "QT_COMPARECIMENTO"]].first()
    d1.columns, d2.columns = ["APT1", "COMP1"], ["APT2", "COMP2"]
    t = p1.join(p2, how="inner").join(d1).join(d2).fillna(0)
    for c in ["A1", "B1", "O1", "O2", "O3", "OR", "BN1", "A2", "B2", "BN2"]:
        if c not in t.columns:
            t[c] = 0
    t = t[(t.APT2 > 0) | (t.A2 + t.B2 > 0)]
    parts = t.index.to_series().str.split("|", expand=True)
    arr = {
        "uf": parts[0].map({u: i for i, u in enumerate(UFS)}).fillna(-1).to_numpy(np.int16),
        "mun": parts[1].astype(int).to_numpy(np.int32),
        "zona": parts[2].astype(int).to_numpy(np.int16),
        "secao": parts[3].astype(int).to_numpy(np.int16),
    }
    for c in ["A1", "B1", "O1", "O2", "O3", "OR", "BN1", "APT1", "COMP1", "A2", "B2", "BN2", "APT2", "COMP2"]:
        arr[c] = t[c].to_numpy(np.int32)
    np.savez_compressed(os.path.join(OUT, f"{eid}.npz"), **arr)
    sA, sB = int(t.A2.sum()), int(t.B2.sum())
    meta = {"id": eid, "ano": ano, "cargo": "presidente" if cargo == "1" else "governador", "uf": uf,
            "A": {"n": A, "nome": nomes.get(A, str(A))}, "B": {"n": B, "nome": nomes.get(B, str(B))},
            "outros": [{"n": n, "nome": nomes.get(n, str(n))} for n in outros],
            "final_pA": sA / (sA + sB), "votos_A": sA, "votos_B": sB, "secoes": int(len(t)),
            "total_1t": {c: int(t[c].sum()) for c in ["A1", "B1", "O1", "O2", "O3", "OR", "BN1", "APT1", "COMP1"]},
            "total_2t": {c: int(t[c].sum()) for c in ["A2", "B2", "BN2", "APT2", "COMP2"]}}
    json.dump(meta, open(os.path.join(OUT, f"{eid}.json"), "w"), ensure_ascii=False, indent=1)
    print(eid, "seções", len(t), "A", meta["A"]["nome"], f"{100*meta['final_pA']:.2f}%", flush=True)
    return meta


def main(ano):
    ano = int(ano)
    t00 = time.time()
    # detalhe (aptos/comparecimento) de presidente e governador, os dois turnos
    zd = baixar(f"{CDN}/detalhe_votacao_secao/detalhe_votacao_secao_{ano}.zip", f"dv{ano}.zip")
    d = ler(zd, COLS_D, lambda c: c[c.CD_CARGO.isin(["1", "3"])])
    os.remove(zd)
    print("detalhe", len(d), flush=True)
    # governadores com 2º turno
    ufs_gov = sorted(d[(d.CD_CARGO == "3") & (d.NR_TURNO == "2")].SG_UF.str.upper().unique())
    print("UFs com 2º turno para governador:", ufs_gov, flush=True)
    feitos = []
    # presidente
    zv = baixar(f"{CDN}/votacao_secao/votacao_secao_{ano}_BR.zip", f"vs{ano}BR.zip")
    v = ler(zv, COLS_V, lambda c: c[c.CD_CARGO == "1"])
    os.remove(zv)
    m = montar(f"pres_{ano}", v, d[d.CD_CARGO == "1"], ano, "1", "BR")
    if m:
        feitos.append(m["id"])
    del v
    for uf in ufs_gov:
        zv = baixar(f"{CDN}/votacao_secao/votacao_secao_{ano}_{uf}.zip", f"vs{ano}{uf}.zip")
        v = ler(zv, COLS_V, lambda c: c[c.CD_CARGO == "3"])
        os.remove(zv)
        m = montar(f"gov_{ano}_{uf}", v, d[(d.CD_CARGO == "3") & (d.SG_UF.str.upper() == uf)], ano, "3", uf)
        if m:
            feitos.append(m["id"])
        del v
    print(f"ano {ano}: {feitos} em {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
