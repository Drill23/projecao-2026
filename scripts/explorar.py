"""Diagnóstico ao vivo: compara andamento por UF, resultado por município e resultado Brasil."""
import json, os, time, urllib.request
BASE = "https://resultados.tse.jus.br/oficial/ele2026/6258"
OUT = "exploracao"; os.makedirs(OUT, exist_ok=True)
L = []
def get(url):
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 projecao-2026"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read()); L.append({"url": url[-55:], "st": r.status, "lm": r.headers.get("Last-Modified"), "cc": r.headers.get("Cache-Control"), "s": round(time.time()-t0, 2)}); return d
    except Exception as e:
        L.append({"url": url[-55:], "erro": repr(e)[:120]}); return None
br = get(f"{BASE}/dados/br/br-c0001-e006258-u.json")
L.append({"br": {"dg": br.get("dg"), "hg": br.get("hg"), "pst": br["s"]["pst"], "est": br["e"]["est"]} if br else None})
ab = get(f"{BASE}/dados/br/br-e006258-ab.json")
if ab:
    L.append({"br_ab_dg": [ab.get("dg"), ab.get("hg")], "ufs": [(a["cdabr"], a["s"]["st"], a["e"].get("est")) for a in ab["abr"]][:30]})
for uf in ("sp", "mg", "ba", "rs"):
    d = get(f"{BASE}/dados/{uf}/{uf}-e006258-ab.json")
    if not d: continue
    muns = [a for a in d["abr"] if a.get("tpabr") == "mun"]
    com = [a for a in muns if float(a["s"]["st"]) > 0]
    L.append({"uf": uf, "dg": [d.get("dg"), d.get("hg")], "n_mun": len(muns), "mun_com_st": len(com), "tipos": sorted({a.get("tpabr") for a in d["abr"]}),
              "ex": [(a["cdabr"], a["s"]["st"], a["e"].get("est")) for a in com[:5]]})
    for a in com[:3]:
        r = get(f"{BASE}/dados/{uf}/{uf}{a['cdabr']}-c0001-e006258-u.json")
        if r:
            L.append({"mun": uf + a["cdabr"], "ab_st": a["s"]["st"], "res_st": r["s"]["st"], "res_est": r["e"]["est"], "res_dg": [r.get("dg"), r.get("hg")]})
    u = get(f"{BASE}/dados/{uf}/{uf}-c0001-e006258-u.json")
    if u: L.append({"uf_res": uf, "pst": u["s"]["pst"], "est": u["e"]["est"], "dg": [u.get("dg"), u.get("hg")]})
json.dump(L, open(os.path.join(OUT, "diag.json"), "w"), indent=1, ensure_ascii=False)
for x in L: print(json.dumps(x, ensure_ascii=False)[:600])
