"""Sonda: arquivos históricos do TSE (cabeçalhos, tamanhos) e API de localidades do IBGE."""
import io, json, os, urllib.request, zipfile
CDN = "https://cdn.tse.jus.br/estatistica/sead/odsele"
OUT = "exploracao"; os.makedirs(OUT, exist_ok=True)
R = {}
def head(url):
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return {"st": r.status, "len": r.headers.get("content-length"), "lm": r.headers.get("last-modified")}
    except Exception as e:
        return {"erro": repr(e)[:100]}
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()
for ano in (2010, 2014, 2018, 2022, 2026):
    R[f"vs_{ano}_BR"] = head(f"{CDN}/votacao_secao/votacao_secao_{ano}_BR.zip")
    R[f"vs_{ano}_SP"] = head(f"{CDN}/votacao_secao/votacao_secao_{ano}_SP.zip")
    R[f"vs_{ano}_AC"] = head(f"{CDN}/votacao_secao/votacao_secao_{ano}_AC.zip")
    R[f"dv_{ano}"] = head(f"{CDN}/detalhe_votacao_secao/detalhe_votacao_secao_{ano}.zip")
    R[f"mz_{ano}"] = head(f"{CDN}/votacao_candidato_munzona/votacao_candidato_munzona_{ano}.zip")
# cabeçalhos e amostra (arquivos pequenos do AC)
for ano in (2014, 2018, 2022):
    try:
        z = zipfile.ZipFile(io.BytesIO(get(f"{CDN}/votacao_secao/votacao_secao_{ano}_AC.zip")))
        nomes = z.namelist()
        csvs = [n for n in nomes if n.lower().endswith(".csv")]
        with z.open(csvs[0]) as fh:
            linhas = [fh.readline().decode("latin1").strip() for _ in range(3)]
        # cargos e turnos presentes
        import pandas as pd
        with z.open(csvs[0]) as fh:
            df = pd.read_csv(fh, sep=";", encoding="latin1", dtype=str, usecols=lambda c: c in ("NR_TURNO", "CD_CARGO", "DS_CARGO"))
        R[f"cab_vs_{ano}_AC"] = {"arquivos": nomes, "linhas": linhas,
                                 "cargos_turnos": df.groupby(list(df.columns)).size().reset_index().values.tolist()[:30]}
    except Exception as e:
        R[f"cab_vs_{ano}_AC"] = {"erro": repr(e)[:200]}
try:
    z = zipfile.ZipFile(io.BytesIO(get(f"{CDN}/detalhe_votacao_secao/detalhe_votacao_secao_2014.zip")))
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    ac = [n for n in csvs if "_AC" in n] or csvs
    with z.open(ac[0]) as fh:
        R["cab_dv_2014"] = {"arquivos": csvs[:5], "linhas": [fh.readline().decode("latin1").strip() for _ in range(2)]}
except Exception as e:
    R["cab_dv_2014"] = {"erro": repr(e)[:200]}
try:
    d = json.loads(get("https://servicodados.ibge.gov.br/api/v1/localidades/municipios"))
    R["ibge"] = {"n": len(d), "ex": d[0]}
except Exception as e:
    R["ibge"] = {"erro": repr(e)[:200]}
json.dump(R, open(f"{OUT}/sonda.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, ensure_ascii=False)[:3000])
