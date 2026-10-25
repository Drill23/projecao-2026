"""Monta a base por seção para a projeção do 2º turno de 2026.

Para cada seção do 2º turno de 2026 (lista oficial do arquivo de configuração de seções):
  - resultado do 1º turno de 2026 na própria seção (Lula, Flávio, Cury, Renan, Caiado,
    nanicos de esquerda, nanicos de direita, brancos+nulos, aptos, comparecimento);
  - resultado de 2022 (1º e 2º turnos) da seção com a mesma chave UF/município/zona/seção,
    usado como covariável e para a validação (reprise de 2022).

Saída: base/base.npz (arrays por seção) e base/meta.json (municípios, candidatos, grupos).
Roda no GitHub Actions (o TSE é acessível de lá).
"""
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

CDN = "https://cdn.tse.jus.br/estatistica/sead/odsele"
DIV = "https://resultados.tse.jus.br/oficial/ele2026"
TMP = os.environ.get("TMPDIR_BASE", "/tmp/base")
OUT = "base"
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

UFS = ["ac", "al", "am", "ap", "ba", "ce", "df", "es", "go", "ma", "mg", "ms", "mt", "pa", "pb",
       "pe", "pi", "pr", "rj", "rn", "ro", "rr", "rs", "sc", "se", "sp", "to", "zz"]
REGIAO = {"ac": "N", "am": "N", "ap": "N", "pa": "N", "ro": "N", "rr": "N", "to": "N",
          "al": "NE", "ba": "NE", "ce": "NE", "ma": "NE", "pb": "NE", "pe": "NE", "pi": "NE",
          "rn": "NE", "se": "NE", "df": "CO", "go": "CO", "ms": "CO", "mt": "CO",
          "es": "SE", "mg": "SE", "rj": "SE", "sp": "SE", "pr": "S", "rs": "S", "sc": "S", "zz": "EX"}
# nanicos que apoiaram Lula no 2º turno (UP, PSTU, PCB, PCO)
NANICOS_ESQ = {80, 16, 21, 29}


def baixar(url, nome):
    caminho = os.path.join(TMP, nome)
    if os.path.exists(caminho) and os.path.getsize(caminho) > 0:
        return caminho
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 projecao-2026"})
    with urllib.request.urlopen(req, timeout=600) as r, open(caminho, "wb") as f:
        while True:
            bloco = r.read(1 << 20)
            if not bloco:
                break
            f.write(bloco)
    print(f"baixado {nome}: {os.path.getsize(caminho)/1e6:.1f} MB em {time.time()-t0:.0f}s", flush=True)
    return caminho


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 projecao-2026"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def ler_csvs(zip_path, usecols_pref, filtro=None):
    """Lê todos os CSV do zip (latin1, ';'), só as colunas desejadas que existirem."""
    partes = []
    with zipfile.ZipFile(zip_path) as z:
        nomes = [n for n in z.namelist() if n.lower().endswith(".csv")]
        print(f"{os.path.basename(zip_path)}: {len(nomes)} csv -> {nomes[:4]}", flush=True)
        for n in nomes:
            with z.open(n) as fh:
                cab = fh.readline().decode("latin1").strip().replace('"', "").split(";")
            cols = [c for c in usecols_pref if c in cab]
            with z.open(n) as fh:
                for ch in pd.read_csv(fh, sep=";", encoding="latin1", usecols=cols,
                                      dtype=str, chunksize=2_000_000):
                    if filtro is not None:
                        ch = filtro(ch)
                    partes.append(ch)
    df = pd.concat(partes, ignore_index=True)
    return df


def chave(df):
    return (df["SG_UF"].str.lower() + "|" + df["CD_MUNICIPIO"].str.zfill(5) + "|"
            + df["NR_ZONA"].str.zfill(4) + "|" + df["NR_SECAO"].str.zfill(4))


def votos_secao(zip_path, ano):
    cols = ["NR_TURNO", "CD_CARGO", "DS_CARGO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO",
            "NR_VOTAVEL", "NM_VOTAVEL", "QT_VOTOS", "QT_APTOS", "QT_COMPARECIMENTO"]

    def filtro(ch):
        if "CD_CARGO" in ch.columns:
            ch = ch[ch["CD_CARGO"] == "1"]
        elif "DS_CARGO" in ch.columns:
            ch = ch[ch["DS_CARGO"].str.upper().str.startswith("PRESIDENTE")]
        return ch

    df = ler_csvs(zip_path, cols, filtro)
    df["QT_VOTOS"] = df["QT_VOTOS"].astype(np.int64)
    df["NR_VOTAVEL"] = df["NR_VOTAVEL"].astype(int)
    df["NR_TURNO"] = df["NR_TURNO"].astype(int)
    df["k"] = chave(df)
    print(f"{ano}: {len(df):,} linhas; turnos {sorted(df.NR_TURNO.unique())}", flush=True)
    return df


def main():
    t00 = time.time()
    # ---------- lista oficial de seções do 2º turno (configuração de seções, pleito 3221)
    secoes = []
    for uf in UFS:
        try:
            cs = get_json(f"{DIV}/arquivo-urna/3221/config/{uf}/{uf}-p003221-cs.json")
        except Exception as e:  # noqa: BLE001
            print("sem cs", uf, e)
            continue
        for ab in cs["abr"]:
            for mu in ab["mu"]:
                for zo in mu["zon"]:
                    for se in zo["sec"]:
                        secoes.append((uf, mu["cd"], zo["cd"], se["ns"]))
    sec = pd.DataFrame(secoes, columns=["uf", "mun", "zona", "secao"])
    # horário em que cada seção foi totalizada no 1º turno (para a reprise com ordem real)
    chegada = {}
    for uf in UFS:
        try:
            cs = get_json(f"{DIV}/arquivo-urna/3220/config/{uf}/{uf}-p003220-cs.json")
        except Exception as e:  # noqa: BLE001
            print("sem cs 1t", uf, e)
            continue
        for ab in cs["abr"]:
            for mu in ab["mu"]:
                for zo in mu["zon"]:
                    for se in zo["sec"]:
                        if se.get("da") and se.get("ha"):
                            d, m_, a = se["da"].split("/")
                            h, mi, s_ = se["ha"].split(":")
                            minutos = (int(d) - 4) * 1440 + int(h) * 60 + int(mi) + int(s_) / 60 - 17 * 60
                            chegada[f"{uf}|{mu['cd']}|{zo['cd']}|{se['ns']}"] = minutos
    print(f"horários de chegada do 1º turno: {len(chegada):,}", flush=True)
    sec["k"] = sec.uf + "|" + sec.mun + "|" + sec.zona + "|" + sec.secao
    print(f"seções 2º turno 2026: {len(sec):,}", flush=True)

    # ---------- 1º turno 2026 por seção
    z26 = baixar(f"{CDN}/votacao_secao/votacao_secao_2026_BR.zip", "vs26.zip")
    v26 = votos_secao(z26, 2026)
    v26 = v26[v26.NR_TURNO == 1]
    nac = v26.groupby(["NR_VOTAVEL", "NM_VOTAVEL"]).QT_VOTOS.sum().sort_values(ascending=False)
    print(nac.head(20), flush=True)
    cands = [n for n, _ in nac.index if n not in (95, 96, 97)]
    principais = [13, 22]
    outros = [n for n in cands if n not in principais]
    # os 3 maiores "outros" ficam individuais (Cury, Renan, Caiado); o resto vira esq/dir
    indiv = outros[:3]
    grupos = {"L": [13], "F": [22]}
    for i, n in enumerate(indiv):
        grupos[f"O{i+1}"] = [n]
    grupos["NE"] = [n for n in outros[3:] if n in NANICOS_ESQ]
    grupos["ND"] = [n for n in outros[3:] if n not in NANICOS_ESQ]
    grupos["BN"] = [95, 96, 97]
    nomes = {int(n): nm for n, nm in nac.index}
    mapa = {}
    for g, ns in grupos.items():
        for n in ns:
            mapa[n] = g
    v26["g"] = v26.NR_VOTAVEL.map(mapa).fillna("ND")
    p26 = v26.pivot_table(index="k", columns="g", values="QT_VOTOS", aggfunc="sum", fill_value=0)
    for g in grupos:
        if g not in p26.columns:
            p26[g] = 0
    p26 = p26[list(grupos)]

    # aptos e comparecimento: detalhe_votacao_secao (se o arquivo de votos não trouxer)
    if "QT_APTOS" in v26.columns and v26["QT_APTOS"].notna().any():
        ap = v26.groupby("k")[["QT_APTOS", "QT_COMPARECIMENTO"]].first().astype(np.int64)
    else:
        zd = baixar(f"{CDN}/detalhe_votacao_secao/detalhe_votacao_secao_2026.zip", "dv26.zip")
        cols = ["NR_TURNO", "CD_CARGO", "DS_CARGO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO",
                "QT_APTOS", "QT_COMPARECIMENTO"]

        def f(ch):
            ch = ch[ch["NR_TURNO"] == "1"]
            if "CD_CARGO" in ch.columns:
                ch = ch[ch["CD_CARGO"] == "1"]
            elif "DS_CARGO" in ch.columns:
                ch = ch[ch["DS_CARGO"].str.upper().str.startswith("PRESIDENTE")]
            return ch
        dv = ler_csvs(zd, cols, f)
        dv["k"] = chave(dv)
        ap = dv.groupby("k")[["QT_APTOS", "QT_COMPARECIMENTO"]].first().astype(np.int64)
    del v26

    # ---------- 2022 (1º e 2º turnos) por seção
    z22 = baixar(f"{CDN}/votacao_secao/votacao_secao_2022_BR.zip", "vs22.zip")
    v22 = votos_secao(z22, 2022)
    v22["g"] = np.select(
        [v22.NR_VOTAVEL == 13, v22.NR_VOTAVEL == 22, v22.NR_VOTAVEL == 15, v22.NR_VOTAVEL == 12,
         v22.NR_VOTAVEL.isin([95, 96, 97])],
        ["L", "B", "T", "C", "BN"], default="O")
    v22["gt"] = v22.g + v22.NR_TURNO.astype(str)
    p22 = v22.pivot_table(index="k", columns="gt", values="QT_VOTOS", aggfunc="sum", fill_value=0)
    C22 = ["L1", "B1", "T1", "C1", "O1", "BN1", "L2", "B2", "O2", "BN2"]
    for c in C22:
        if c not in p22.columns:
            p22[c] = 0
    p22 = p22[C22]
    p22.columns = ["x22_" + c for c in C22]
    if "QT_APTOS" in v22.columns and v22["QT_APTOS"].notna().any():
        ap22 = v22[v22.NR_TURNO == 2].groupby("k")[["QT_APTOS", "QT_COMPARECIMENTO"]].first().astype(np.int64)
        ap22_1 = v22[v22.NR_TURNO == 1].groupby("k")[["QT_COMPARECIMENTO"]].first().astype(np.int64)
    else:
        zd = baixar(f"{CDN}/detalhe_votacao_secao/detalhe_votacao_secao_2022.zip", "dv22.zip")
        cols = ["NR_TURNO", "CD_CARGO", "DS_CARGO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO",
                "QT_APTOS", "QT_COMPARECIMENTO"]

        def f2(ch):
            if "CD_CARGO" in ch.columns:
                ch = ch[ch["CD_CARGO"] == "1"]
            elif "DS_CARGO" in ch.columns:
                ch = ch[ch["DS_CARGO"].str.upper().str.startswith("PRESIDENTE")]
            return ch
        dv = ler_csvs(zd, cols, f2)
        dv["k"] = chave(dv)
        dv[["QT_APTOS", "QT_COMPARECIMENTO"]] = dv[["QT_APTOS", "QT_COMPARECIMENTO"]].astype(np.int64)
        ap22 = dv[dv.NR_TURNO == "2"].groupby("k")[["QT_APTOS", "QT_COMPARECIMENTO"]].first()
        ap22_1 = dv[dv.NR_TURNO == "1"].groupby("k")[["QT_COMPARECIMENTO"]].first()
    del v22

    # ---------- monta a tabela final na ordem das seções do 2º turno
    # (seções de 2026 que agregaram outras no 1º turno: chave igual; seções novas ficam sem base)
    todas = pd.Index(sec.k).union(p26.index)
    t = pd.DataFrame(index=todas)
    t = t.join(p26).join(ap).join(p22).join(ap22.rename(columns={"QT_APTOS": "APT22",
                                                                 "QT_COMPARECIMENTO": "COMP22b"}))
    t = t.join(ap22_1.rename(columns={"QT_COMPARECIMENTO": "COMP22a"}))
    t["cheg1"] = pd.Series(chegada, dtype=float).reindex(t.index).fillna(-1).to_numpy()
    t["no_t2"] = t.index.isin(sec.k)
    t["tem26"] = t.index.isin(p26.index)
    t["tem22"] = t.index.isin(p22.index)
    t = t.fillna(0)
    print("seções no 2º turno:", int(t.no_t2.sum()), " com base 2026:", int((t.no_t2 & t.tem26).sum()),
          " com 2022:", int((t.no_t2 & t.tem22).sum()), " só no 1º turno:", int((~t.no_t2).sum()), flush=True)
    partes = t.index.to_series().str.split("|", expand=True)
    t["uf"], t["mun"], t["zona"], t["secao"] = partes[0], partes[1], partes[2], partes[3]

    # ---------- metadados de municípios (nome, código IBGE, capital)
    cm = get_json(f"{DIV}/6258/config/mun-e006258-cm.json")
    muns = {}
    for ab in cm["abr"]:
        for mu in ab["mu"]:
            muns[ab["cd"] + mu["cd"]] = {"nm": mu["nm"], "cdi": mu.get("cdi", ""), "c": mu.get("c", "n")}

    uf_idx = {u: i for i, u in enumerate(UFS)}
    t = t[t.uf.isin(uf_idx)]
    mun_keys = sorted((t.uf + t.mun).unique())
    mun_idx = {m: i for i, m in enumerate(mun_keys)}
    arr = {
        "uf": t.uf.map(uf_idx).to_numpy(np.int16),
        "mun": (t.uf + t.mun).map(mun_idx).to_numpy(np.int32),
        "zona": t.zona.astype(int).to_numpy(np.int16),
        "secao": t.secao.astype(int).to_numpy(np.int16),
        "no_t2": t.no_t2.to_numpy(bool),
        "tem26": t.tem26.to_numpy(bool),
        "tem22": t.tem22.to_numpy(bool),
        "cheg1": t.cheg1.to_numpy(np.float32),
    }
    for c in list(grupos) + ["QT_APTOS", "QT_COMPARECIMENTO"] + ["x22_" + c for c in
                            ["L1", "B1", "T1", "C1", "O1", "BN1", "L2", "B2", "O2", "BN2"]] + ["APT22", "COMP22a", "COMP22b"]:
        arr[c] = t[c].to_numpy(np.int32)
    np.savez_compressed(os.path.join(OUT, "base.npz"), **arr)
    meta = {
        "ufs": UFS, "regiao": REGIAO,
        "municipios": [{"k": m, "uf": m[:2], "cd": m[2:], **muns.get(m, {"nm": m, "cdi": "", "c": "n"})}
                       for m in mun_keys],
        "grupos": {g: [{"n": int(n), "nm": nomes.get(int(n), str(n))} for n in ns] for g, ns in grupos.items()},
        "totais_1t": {g: int(t[g].sum()) for g in grupos},
        "gerado": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(json.dumps(meta["grupos"], ensure_ascii=False))
    print(json.dumps(meta["totais_1t"]))
    print(f"ok em {time.time()-t00:.0f}s; base.npz {os.path.getsize(os.path.join(OUT,'base.npz'))/1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
