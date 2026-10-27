"""Gráficos da avaliação v1 x v2 (PNG estáticos para a documentação).

Uso: python scripts/graficos.py resultados/resumo.json pasta_saida/
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

SUP = "#fcfcfb"
T1, T2, T3 = "#0b0b0b", "#52514e", "#8a8985"
GRADE = "#e6e5e1"
V2, V1, BRUTO = "#2a78d6", "#eb6834", "#8a8985"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRADE, "axes.labelcolor": T2,
    "xtick.color": T2, "ytick.color": T2, "axes.facecolor": SUP, "figure.facecolor": SUP,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRADE,
    "grid.linewidth": 1, "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
})


def br(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def num_curto(v, _=None):
    t = f"{v:.3f}".rstrip("0").rstrip(".")
    return t.replace(".", ",")


def legenda(ax, ncol=3):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol, fontsize=10, handlelength=2.2,
              columnspacing=1.6, borderaxespad=0)


def eixo_apuracao(ax, cps):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([c for c in cps if c in (0.5, 1, 2, 5, 10, 20, 50, 90)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: (br(v, 1) if v < 1 else f"{v:.0f}") + "%"))
    ax.set_xlabel("% do eleitorado já apurado")


def titulo(fig, t, s):
    fig.text(0.012, 0.965, t, fontsize=15, fontweight="bold", color=T1, va="top")
    fig.text(0.012, 0.91, s, fontsize=11, color=T2, va="top")


def rotulo_fim(ax, x, y, txt, cor):
    ax.plot([x], [y], "o", ms=8, color=cor, mec=SUP, mew=2, zorder=5)
    ax.annotate(txt, (x, y), xytext=(8, 0), textcoords="offset points", va="center", color=T1, fontsize=10.5)


def erro(res, grupo, out, nome, t, s):
    pc = res["por_checkpoint"][grupo]
    cps = sorted(float(k) for k in pc)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(top=0.76, left=0.09, right=0.95, bottom=0.12)
    titulo(fig, t, s)
    series = [("Contagem do TSE", "bruto", BRUTO, (0, (4, 3))), ("Modelo v1", "v1", V1, "-"), ("Modelo v2", "v2", V2, "-")]
    for lab, k, cor, ls in series:
        ys = [pc[(str(c) if str(c) in pc else str(int(c)) if c == int(c) else str(c))][k]["erro_medio_abs"] for c in cps]
        ax.plot(cps, ys, color=cor, ls=ls, label=lab)
    eixo_apuracao(ax, cps)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(num_curto))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("erro médio (pontos percentuais)")
    legenda(ax)
    fig.savefig(os.path.join(out, nome), dpi=160)
    plt.close(fig)


def cobertura(res, grupo, out, nome, t, s):
    pc = res["por_checkpoint"][grupo]
    ks = sorted(pc, key=float)
    cps = [float(k) for k in ks]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(top=0.76, left=0.09, right=0.95, bottom=0.12)
    titulo(fig, t, s)
    ax.axhline(90, color=T3, lw=1)
    ax.annotate("meta: 90%", (cps[-1], 90), xytext=(0, -14), textcoords="offset points", color=T2, fontsize=10, ha="right")
    for lab, k, cor in (("Modelo v1", "v1", V1), ("Modelo v2", "v2", V2)):
        ys = [100 * pc[c][k]["cobertura_90"] for c in ks]
        ax.plot(cps, ys, color=cor, label=lab, marker="o", ms=6, mec=SUP, mew=1.5)
    eixo_apuracao(ax, cps)
    ax.set_ylim(60, 102)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("vezes em que o resultado caiu na faixa")
    legenda(ax, 2)
    fig.savefig(os.path.join(out, nome), dpi=160)
    plt.close(fig)


def trajetoria(res, eid, out, nome, t, s):
    tr = res["trajetorias"][eid]
    x = [p["cp"] for p in tr]
    fin = 100 * tr[0]["final"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(top=0.76, left=0.09, right=0.95, bottom=0.12)
    titulo(fig, t, s)
    ax.fill_between(x, [100 * p["v2"]["lo"] for p in tr], [100 * p["v2"]["hi"] for p in tr], color=V2, alpha=0.12, lw=0,
                    label="Faixa de 90% (v2)")
    ax.plot(x, [100 * p["bruto"] for p in tr], color=BRUTO, ls=(0, (4, 3)), label="Contagem do TSE")
    ax.plot(x, [100 * p["v1"]["pA"] for p in tr], color=V1, label="Modelo v1")
    ax.plot(x, [100 * p["v2"]["pA"] for p in tr], color=V2, label="Modelo v2")
    ax.axhline(fin, color=T1, lw=1)
    ax.annotate(f"resultado final: {br(fin, 2)}%", (x[0], fin), xytext=(0, 6), textcoords="offset points", color=T1, fontsize=10)
    ax.axhline(50, color=T3, lw=1)
    rotulo_fim(ax, x[-1], 100 * tr[-1]["v2"]["pA"], "v2", V2)
    eixo_apuracao(ax, x)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{br(v, 0)}%"))
    ax.set_ylabel("vencedor, % dos votos válidos")
    legenda(ax, 4)
    fig.savefig(os.path.join(out, nome), dpi=160)
    plt.close(fig)


def cravadas(res, out, nome, t, s):
    import numpy as np
    cr = res["cravadas"]["apertadas_ate_3pts"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(top=0.72, left=0.16, right=0.95, bottom=0.16)
    titulo(fig, t, s)
    for i, (lab, k, cor) in enumerate((("Modelo v2", "v2", V2), ("Modelo v1", "v1", V1))):
        cps = [c["cp"] for c in cr[k] if c["cp"] is not None]
        med = float(np.median(cps))
        ax.barh(i, med, height=0.42, color=cor)
        ax.annotate(f"{br(med, 0)}% (mediana)", (med, i), xytext=(-6, 0), textcoords="offset points",
                    va="center", ha="right", color="white", fontsize=10.5, fontweight="bold")
        ax.scatter(cps, [i - 0.33] * len(cps), s=16, color=T2, alpha=0.6, zorder=4)
    ax.set_axisbelow(True)
    ax.set_yticks([0, 1], ["Modelo v2", "Modelo v1"])
    ax.set_ylim(-0.55, 1.35)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 100)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 90]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: (br(v, 1) if v < 1 else f"{v:.0f}") + "%"))
    ax.set_xlabel("% do eleitorado apurado no momento de cravar (cada ponto = uma reprise)")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(out, nome), dpi=160)
    plt.close(fig)


def main(caminho, out):
    res = json.load(open(caminho))
    os.makedirs(out, exist_ok=True)
    n = res.get("n_eleicoes", 0)
    erro(res, "todas", out, "erro_por_apuracao.png", "Quanto cada um erra o resultado final",
         f"Média de {n} segundos turnos (presidente e governador, 2010–2026), 3 ordens de chegada cada")
    cobertura(res, "todas", out, "cobertura_faixa.png", "A faixa de 90% acerta 90% das vezes?",
              "Porcentagem de vezes em que o resultado final caiu dentro da faixa mostrada")
    cravadas(res, out, "cravar_apertadas.png", "Em disputas apertadas, quando dá para cravar",
             "Eleições decididas por até 3 pontos: % apurado a partir do qual a chance passou de 95% e não caiu mais")
    if "pres_2026" in res["trajetorias"]:
        trajetoria(res, "pres_2026", out, "reprise_2026.png", "Reprise da noite de 25/10/2026",
                   "Presidente, com a ordem real de chegada das urnas do 2º turno")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
