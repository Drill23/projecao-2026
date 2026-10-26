"""Baixa o resultado final do 2º turno por município (para a análise de transferências)."""
import concurrent.futures as cf, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ao_vivo as av
meta = json.load(open("base/meta.json"))
muns = [m["k"] for m in meta["municipios"]]
def f(k):
    d = av.get_json(f"{av.DIV}/{av.ELE}/dados/{k[:2]}/{k}-c0001-e{av.E6}-u.json")
    if not d: return k, None
    A = B = 0.0
    for cg in d.get("carg", []):
        for ag in cg.get("agr", []):
            for p in ag.get("par", []):
                for c in p.get("cand", []):
                    if str(c.get("n")) == "13": A = av.num(c.get("vap"))
                    elif str(c.get("n")) == "22": B = av.num(c.get("vap"))
    v = d.get("v", {}); e = d.get("e", {})
    return k, {"A": A, "B": B, "comp": av.num(e.get("c")), "te": av.num(e.get("te")), "est": av.num(e.get("est")),
               "vb": av.num(v.get("vb")), "vn": av.num(v.get("tvn")), "pst": av.num(d["s"].get("pst"))}
out = {}
with cf.ThreadPoolExecutor(12) as ex:
    for k, r in ex.map(f, muns):
        if r: out[k] = r
os.makedirs("exploracao", exist_ok=True)
json.dump(out, open("exploracao/final_mun.json", "w"))
print(len(out), "municípios;", "faltando", len(muns) - len(out))
