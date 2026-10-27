"""Centróide (lat/lon) de cada município a partir dos locais de votação de 2026."""
import json, os, sys, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from historico import baixar, CDN
z = baixar(f"{CDN}/eleitorado_locais_votacao/eleitorado_local_votacao_2026.zip", "locais2026.zip")
partes = []
with zipfile.ZipFile(z) as zf:
    for n in [n for n in zf.namelist() if n.lower().endswith(".csv")]:
        with zf.open(n) as fh:
            cab = fh.readline().decode("latin1").strip().replace('"', "").split(";")
        print(n, cab, flush=True)
        lat = [c for c in cab if "LATITUDE" in c][0]; lon = [c for c in cab if "LONGITUDE" in c][0]
        el = [c for c in cab if c.startswith("QT_ELEITOR")]
        cols = ["SG_UF", "CD_MUNICIPIO", lat, lon] + el[:1]
        with zf.open(n) as fh:
            df = pd.read_csv(fh, sep=";", encoding="latin1", usecols=cols, dtype=str)
        df = df.rename(columns={lat: "lat", lon: "lon"})
        df["w"] = pd.to_numeric(df[el[0]], errors="coerce").fillna(1) if el else 1.0
        partes.append(df)
df = pd.concat(partes)
for c in ("lat", "lon"):
    df[c] = pd.to_numeric(df[c].str.replace(",", "."), errors="coerce")
df = df[(df.lat.abs() > 0.01) & (df.lat > -40) & (df.lat < 10) & (df.lon < -25) & (df.lon > -80)]
df["k"] = df.SG_UF.str.upper() + df.CD_MUNICIPIO.str.zfill(5)
g = df.groupby("k").apply(lambda x: pd.Series({"lat": np.average(x.lat, weights=x.w), "lon": np.average(x.lon, weights=x.w)}))
os.makedirs("historico", exist_ok=True)
json.dump({k: [round(r.lat, 4), round(r.lon, 4)] for k, r in g.iterrows()}, open("historico/municipios_geo.json", "w"))
print("municípios com coordenadas:", len(g))
