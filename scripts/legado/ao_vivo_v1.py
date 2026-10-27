"""Loop da noite da eleição: lê o TSE a cada ciclo, projeta e publica.

Uso:  python scripts/ao_vivo.py --ate 2026-10-26T01:00 [--intervalo 60] [--publicar]
Saída: saida/projecao.json (o que a página mostra). Com --publicar, cada ciclo vira um
commit único (forçado) no ramo `dados` do repositório.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projecao.dados import Base2026  # noqa: E402

DIV = "https://resultados.tse.jus.br/oficial/ele2026"
ELE = os.environ.get("ELEICAO", "6258")      # 6257 = 1º turno (ensaio)
PLEITO = os.environ.get("PLEITO", "3221")    # 3220 = 1º turno (ensaio)
E6, P6 = f"{int(ELE):06d}", f"{int(PLEITO):06d}"
BRT = dt.timezone(dt.timedelta(hours=-3))
OUT = "saida"
MIN_PCT = 0.01          # só mostra projeção a partir de 1% do eleitorado apurado
NOMES = {"A": {"nome": "Lula", "numero": 13, "partido": "PT"},
         "B": {"nome": "Flávio Bolsonaro", "numero": 22, "partido": "PL"}}


def agora():
    return dt.datetime.now(BRT)


import threading

import requests
from requests.adapters import HTTPAdapter

_local = threading.local()
ERROS = {"n": 0, "ult": "", "429": 0}


def _sessao():
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (projecao-2026; github.com/Drill23/projecao-2026)",
                          "Accept-Encoding": "gzip, deflate"})
        s.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0))
        _local.s = s
    return s


def get_json(url, tentativas=4):
    for t in range(tentativas):
        try:
            r = _sessao().get(url, timeout=(5, 30))
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.json()
            ERROS["n"] += 1
            ERROS["ult"] = f"HTTP {r.status_code} {url[-60:]}"
            if r.status_code == 429:
                ERROS["429"] += 1
                time.sleep(2 + 3 * t)
        except Exception as e:  # noqa: BLE001
            ERROS["n"] += 1
            ERROS["ult"] = f"{type(e).__name__} {url[-60:]}"
        time.sleep(0.5 * (2 ** t))
    return None


def num(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:  # noqa: BLE001
        return 0.0


class Estado:
    def __init__(self, base: Base2026):
        self.base = base
        n = base.n_mun
        self.A = np.zeros(n)
        self.B = np.zeros(n)
        self.comp = np.zeros(n)
        self.est = np.zeros(n)
        self.te = np.zeros(n)
        self.st_res = np.full(n, -1.0)       # seções totalizadas segundo o arquivo de resultado
        self.st_ab = np.zeros(n)              # segundo o andamento
        self.est_ab = np.zeros(n)             # eleitorado apurado segundo o andamento
        self.ufs_pend = set()                 # UFs cujo arquivo de seções falhou (tenta de novo)
        self.pct_ab = 0.0
        self.orcamento = 600                  # máx. de arquivos de município por ciclo (o TSE limita)
        self.ult_leitura = np.zeros(n)        # quando cada município foi lido pela última vez
        self.est_uf_res = {}                  # eleitorado apurado por UF segundo o resultado da UF
        self.uf_de_mun = np.array([k[:2] for k in base.mun_keys])
        self.cd_de_mun = np.array([k[2:] for k in base.mun_keys])
        self.n_lidos = 0
        self.cont = np.zeros(base.n, bool)
        self.uf_hg = {}
        self.serie = []
        self.modelo = base.modelo()
        self.oficial = {}
        # eleitorado total por município a partir da base (substituído pelo do TSE quando vier)
        self.te[:] = np.bincount(base.mun, weights=base.E, minlength=n)

    # -------------------------------------------------------------- leitura
    def ler_andamento(self):
        """Andamento por UF -> municípios. Devolve as UFs e os municípios que mudaram."""
        ufs_mud, muns_mud = set(), []

        def f(uf):
            return uf, get_json(f"{DIV}/{ELE}/dados/{uf}/{uf}-e{E6}-ab.json")
        with cf.ThreadPoolExecutor(6) as ex:
            res = list(ex.map(f, self.base.ufs))
        for uf, d in res:
            if not d:
                continue
            for a in d.get("abr", []):
                if a.get("tpabr") != "mun":
                    continue
                k = uf + a["cdabr"]
                i = self.base.mun_idx.get(k)
                if i is None:
                    continue
                st = num(a["s"]["st"])
                self.te[i] = num(a["e"]["te"]) or self.te[i]
                self.est_ab[i] = num(a["e"].get("est"))
                if st != self.st_ab[i]:
                    ufs_mud.add(uf)
                if st > 0 and self.st_res[i] != st:
                    muns_mud.append((uf, a["cdabr"], i))
                self.st_ab[i] = st
        return ufs_mud, muns_mud

    def ler_resultados(self, muns):
        mudou = set()
        agora_ = time.time()
        for _, _, i in muns:
            self.ult_leitura[i] = agora_
        self.n_lidos = len(muns)

        def f(item):
            uf, cd, i = item
            return i, get_json(f"{DIV}/{ELE}/dados/{uf}/{uf}{cd}-c0001-e{E6}-u.json")
        with cf.ThreadPoolExecutor(12) as ex:
            for i, d in ex.map(f, muns):
                if not d:
                    continue
                A = B = 0.0
                for cg in d.get("carg", []):
                    for ag in cg.get("agr", []):
                        for p in ag.get("par", []):
                            for c in p.get("cand", []):
                                if str(c.get("n")) == "13":
                                    A = num(c.get("vap"))
                                elif str(c.get("n")) == "22":
                                    B = num(c.get("vap"))
                novo_est = num(d["e"].get("est"))
                if novo_est != self.est[i] or A != self.A[i]:
                    mudou.add(self.uf_de_mun[i])
                self.A[i], self.B[i] = A, B
                self.comp[i] = num(d["e"].get("c"))
                self.est[i] = novo_est
                self.te[i] = num(d["e"].get("te")) or self.te[i]
                self.st_res[i] = num(d["s"].get("st"))
        return mudou

    def ler_ufs(self):
        """Resultado agregado de cada UF: mostra em que UFs há votos que ainda não lemos."""
        def f(uf):
            return uf, get_json(f"{DIV}/{ELE}/dados/{uf}/{uf}-c0001-e{E6}-u.json")
        with cf.ThreadPoolExecutor(6) as ex:
            for uf, d in ex.map(f, self.base.ufs):
                if d:
                    self.est_uf_res[uf] = max(self.est_uf_res.get(uf, 0), num(d["e"].get("est")))
        # andamento nacional por UF (outro arquivo, às vezes à frente dos demais)
        d = get_json(f"{DIV}/{ELE}/dados/br/br-e{E6}-ab.json")
        if d:
            for a in d.get("abr", []):
                if a.get("tpabr") == "uf" and a.get("cdabr") in self.base.ufs:
                    uf = a["cdabr"]
                    self.est_uf_res[uf] = max(self.est_uf_res.get(uf, 0), num(a["e"].get("est")))

    def ler_secoes(self, ufs):
        def f(uf):
            return uf, get_json(f"{DIV}/arquivo-urna/{PLEITO}/config/{uf}/{uf}-p{P6}-cs.json")
        with cf.ThreadPoolExecutor(4) as ex:
            for uf, d in ex.map(f, sorted(ufs)):
                if not d:
                    self.ufs_pend.add(uf)
                    continue
                self.ufs_pend.discard(uf)
                for ab in d.get("abr", []):
                    for mu in ab.get("mu", []):
                        mk = uf + mu["cd"]
                        for zo in mu.get("zon", []):
                            z = int(zo["cd"])
                            for se in zo.get("sec", []):
                                if se.get("da"):
                                    i = self.base.chave.get((mk, z, int(se["ns"])))
                                    if i is not None:
                                        self.cont[i] = True

    def ler_brasil(self):
        d = get_json(f"{DIV}/{ELE}/dados/br/br-c0001-e{E6}-u.json")
        if not d:
            return
        A = B = 0.0
        for cg in d.get("carg", []):
            for ag in cg.get("agr", []):
                for p in ag.get("par", []):
                    for c in p.get("cand", []):
                        if str(c.get("n")) == "13":
                            A = num(c.get("vap"))
                        elif str(c.get("n")) == "22":
                            B = num(c.get("vap"))
        self.oficial = {
            "A": A, "B": B, "pA": A / (A + B) if A + B > 0 else None,
            "pst": num(d["s"].get("pst")), "pest": num(d["e"].get("pest")),
            "te": num(d["e"].get("te")), "est": num(d["e"].get("est")),
            "comp": num(d["e"].get("c")), "brancos": num(d.get("v", {}).get("vb")),
            "nulos": num(d.get("v", {}).get("tvn")),
            "dt": d.get("dt"), "ht": d.get("ht"), "tf": d.get("tf"),
        }

    # -------------------------------------------------------------- ciclo
    def ciclo(self, n_boot=150):
        t0 = time.time()
        e429 = ERROS["429"]
        ufs, muns = self.ler_andamento()
        self.ler_ufs()
        agora_ = time.time()
        # 1) municípios que o andamento diz que mudaram (maior avanço primeiro)
        muns = sorted(muns, key=lambda it: self.est_ab[it[2]] - self.est[it[2]], reverse=True)
        ja = {it[2] for it in muns}
        # 2) varredura das UFs cujo resultado agregado tem mais eleitorado apurado do que a soma
        #    dos municípios que já lemos (o andamento por município às vezes atrasa)
        gap_ufs = []
        for uf in self.base.ufs:
            sel = self.uf_de_mun == uf
            if self.est_uf_res.get(uf, 0) > self.est[sel].sum() + 0.5:
                gap_ufs.append(uf)
        if gap_ufs:
            sel = np.isin(self.uf_de_mun, gap_ufs) & (self.ult_leitura < agora_ - 50)
            cand = [i for i in np.where(sel)[0] if i not in ja]
            cand.sort(key=lambda i: (self.ult_leitura[i], -self.te[i]))
            muns += [(self.uf_de_mun[i], self.cd_de_mun[i], i) for i in cand]
        muns = muns[:int(self.orcamento)]
        self.gap_ufs = gap_ufs
        if muns:
            ufs = set(ufs) | self.ler_resultados(muns)
        else:
            self.n_lidos = 0
        ufs = set(ufs) | self.ufs_pend
        if ufs:
            self.ler_secoes(ufs)
        self.ler_brasil()
        # ajusta o ritmo: se o TSE devolveu 429, reduz; senão, aumenta devagar
        if ERROS["429"] > e429:
            self.orcamento = max(80, self.orcamento * 0.5)
        else:
            self.orcamento = min(1500, self.orcamento * 1.15)
        t_leitura = time.time() - t0
        obs = {"A": self.A, "B": self.B, "comp": self.comp, "est": self.est, "te": self.te}
        pct = self.est.sum() / max(self.te.sum(), 1)
        self.pct_ab = self.est_ab.sum() / max(self.te.sum(), 1)
        proj = None
        if pct >= MIN_PCT:
            try:
                o = self.modelo.projetar(self.cont, obs, n_boot=n_boot, seed=len(self.serie))
                if o.get("ok"):
                    proj = o
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        t_modelo = time.time() - t0 - t_leitura
        if getattr(self, "retomando", False):
            atrasados = int(((self.st_ab > 0) & (self.st_res < 0)).sum())
            if atrasados > 0:
                print(f"retomando: {atrasados} municípios ainda por ler; não publica este ciclo", flush=True)
                return None
            self.retomando = False
        return self.saida(pct, proj, t_leitura, t_modelo)

    def saida(self, pct, proj, t_leitura, t_modelo):
        Aap, Bap = float(self.A.sum()), float(self.B.sum())
        bruto = Aap / (Aap + Bap) if Aap + Bap > 0 else None
        ofic = self.oficial
        pct_of = (ofic.get("pest") or 0) / 100 if ofic else self.pct_ab
        status = "aguardando" if Aap + Bap == 0 else ("encerrado" if pct_of >= 0.9999 else "apurando")
        ponto = {"t": agora().strftime("%H:%M:%S"), "pct": round(pct, 6),
                 "bruto": None if bruto is None else round(bruto, 6)}
        if proj:
            ponto.update({"proj": round(proj["pA"], 6), "lo": round(proj["pA_lo"], 6),
                          "hi": round(proj["pA_hi"], 6), "prob": round(proj["prob_A"], 4)})
        if status != "aguardando" and (not self.serie or self.serie[-1]["pct"] != ponto["pct"]
                                      or self.serie[-1].get("proj") != ponto.get("proj")):
            self.serie.append(ponto)
        ufs = []
        for j, u in enumerate(self.base.ufs):
            linha = {"uf": u.upper()}
            m = self.base.uf == j
            if proj:
                linha.update({"proj": round(proj["uf_pA"][j], 5), "apur": round(proj["uf_apur"][j], 5),
                              "bruto": round(proj["uf_obs_pA"][j], 5), "votos": round(proj["uf_V"][j])})
            ufs.append(linha)
        out = {
            "atualizado": agora().isoformat(timespec="seconds"),
            "status": status,
            "candidatos": NOMES,
            "apurado": {"A": Aap, "B": Bap, "pA": bruto, "pct_eleitorado": pct,
                        "comparecimento": float(self.comp.sum())},
            "oficial": ofic,
            "projecao": None if not proj else {
                "pA": proj["pA"], "lo": proj["pA_lo"], "hi": proj["pA_hi"], "prob_A": proj["prob_A"],
                "A": proj["A"], "B": proj["B"], "q": dict(zip(["Cury", "Renan Santos", "Caiado"],
                                                             [round(x, 3) for x in proj["q"][:3]])),
                "comparecimento_rel": proj["turn_nat"], "validos_rel": proj["val_nat"],
                "n_mun": proj["n_mun_obs"]},
            "ufs": ufs,
            "serie": self.serie,
            "tempos": {"leitura_s": round(t_leitura, 1), "modelo_s": round(t_modelo, 1)},
        }
        os.makedirs(OUT, exist_ok=True)
        tmp = os.path.join(OUT, "projecao.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, os.path.join(OUT, "projecao.json"))
        return out


def publicar(repo_dir):
    """Commit único e forçado no ramo `dados` com saida/projecao.json."""
    pub = os.path.join(repo_dir, ".pub")
    os.makedirs(pub, exist_ok=True)
    src = os.path.join(OUT, "projecao.json")
    with open(src, "rb") as f, open(os.path.join(pub, "projecao.json"), "wb") as g:
        g.write(f.read())
    env = dict(os.environ, GIT_DIR=os.path.join(repo_dir, ".git"), GIT_WORK_TREE=pub,
               GIT_INDEX_FILE=os.path.join(repo_dir, ".git", "index-dados"))
    subprocess.run(["git", "add", "-A", "."], env=env, cwd=pub, capture_output=True)
    tree = subprocess.run(["git", "write-tree"], env=env, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "commit-tree", tree, "-m", "dados " + agora().strftime("%H:%M:%S")],
                            env=env, capture_output=True, text=True).stdout.strip()
    r = subprocess.run(["git", "push", "-q", "-f", "origin", f"{commit}:refs/heads/dados"],
                       cwd=repo_dir, capture_output=True, text=True)
    if r.returncode != 0:
        print("push falhou:", r.stderr[-300:], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ate", default=None, help="hora de parar (Brasília), ex. 2026-10-26T01:00")
    ap.add_argument("--max-min", type=float, default=345, help="duração máxima em minutos")
    ap.add_argument("--intervalo", type=float, default=60)
    ap.add_argument("--publicar", action="store_true")
    ap.add_argument("--uma-vez", action="store_true")
    ap.add_argument("--boot", type=int, default=150)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--ciclos", type=int, default=0, help="para após N ciclos (ensaio)")
    a = ap.parse_args()
    fim = agora() + dt.timedelta(minutes=a.max_min)
    if a.ate:
        fim = min(fim, dt.datetime.fromisoformat(a.ate).replace(tzinfo=BRT))
    base = Base2026()
    est = Estado(base)
    # retoma a série publicada (se o robô for reiniciado no meio da apuração)
    try:
        url = "https://raw.githubusercontent.com/Drill23/projecao-2026/dados/projecao.json"
        ant = get_json(url)
        if ant and ant.get("serie") and ant.get("atualizado", "")[:10] == agora().strftime("%Y-%m-%d"):
            est.serie = ant["serie"]
            est.retomando = True     # reinício no meio da apuração: só publica depois de pôr tudo em dia
    except Exception:  # noqa: BLE001
        pass
    encerrado_desde = None
    while True:
        t0 = time.time()
        try:
            out = est.ciclo(n_boot=a.boot)
            if out is None:
                if agora() >= fim:
                    break
                time.sleep(5)
                continue
            p = out["projecao"]
            print(f"{agora():%H:%M:%S} {out['status']} apurado {100*out['apurado']['pct_eleitorado']:.2f}% "
                  f"bruto {out['apurado']['pA']} proj {None if not p else round(100*p['pA'],2)} "
                  f"[{None if not p else round(100*p['lo'],2)}, {None if not p else round(100*p['hi'],2)}] "
                  f"tempos {out['tempos']}", flush=True)
            if a.diag:
                of = est.oficial
                print("DIAG municípios com resultado:", int((est.st_res > 0).sum()), "de", base.n_mun,
                      "| soma A,B:", int(est.A.sum()), int(est.B.sum()), "| oficial A,B:", of.get("A"), of.get("B"),
                      "| est soma/oficial:", int(est.est.sum()), of.get("est"),
                      "| te soma/oficial:", int(est.te.sum()), of.get("te"),
                      "| seções marcadas apuradas:", int(est.cont.sum()), "de", base.n,
                      "| eleitorado nessas seções:", int(base.E[est.cont].sum()),
                      "| erros de leitura:", ERROS, "| lidos:", est.n_lidos, "| orçamento:", int(est.orcamento),
                      "| UFs com votos ainda não lidos:", getattr(est, "gap_ufs", []), flush=True)
            if a.publicar:
                publicar(os.getcwd())
            if out["status"] == "encerrado":
                encerrado_desde = encerrado_desde or time.time()
                if time.time() - encerrado_desde > 20 * 60:
                    print("apuração encerrada; fim.")
                    break
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        est.n_ciclo = getattr(est, "n_ciclo", 0) + 1
        if a.uma_vez or agora() >= fim or (a.ciclos and est.n_ciclo >= a.ciclos):
            # parou só pelo limite de tempo da execução, com apuração ainda em curso e antes do
            # horário final: pede ao workflow que ligue uma execução nova (que retoma daqui)
            limite_final = dt.datetime.fromisoformat(a.ate).replace(tzinfo=BRT) if a.ate else None
            if (a.publicar and not a.uma_vez and encerrado_desde is None
                    and limite_final and agora() < limite_final - dt.timedelta(minutes=5)):
                os.makedirs(OUT, exist_ok=True)
                open(os.path.join(OUT, "renovar"), "w").write(agora().isoformat())
                print("limite de tempo da execução: pedindo renovação", flush=True)
            break
        time.sleep(max(5, a.intervalo - (time.time() - t0)))


if __name__ == "__main__":
    main()
