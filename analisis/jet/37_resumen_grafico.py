"""Paper 2, paso 37: resumen gráfico (50 × 60 mm) exigido por la International Journal of Climatology.
Título, autor (con asterisco), ≤ 80 palabras y una figura: tendencia agrupada del jet (pp por década) con IC 95 % de 24_cascada.csv y 22_tendencias.csv.
Salida: figuras/jet/graphical_abstract.png (600 dpi)
"""
import os
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["font.family"] = "Helvetica Neue" if "Helvetica Neue" in {f.name for f in mpl.font_manager.fontManager.ttflist} else "DejaVu Sans"


def main():
    K = pd.read_csv("analisis/jet/24_cascada.csv")
    T = pd.read_csv("analisis/jet/22_tendencias.csv")

    def fila(c, v):
        q = K[(K.desde == 1980) & (K.viento == v) & (K.control == c)].iloc[0]
        return q.beta, q.lo, q.hi
    d = T[(T.desde == 1980) & (T.serie == "crudo") & (T.con_fuente)].iloc[0]
    filas = [("Full profile, raw", fila("sólo mes y estación", "crudo"), "#9a9a9a"),
             ("+ source archive,\nbreaks removed", fila("+ fuente, incertidumbre del ajuste", "ajustado"), "#8c510a"),
             ("Deep jets,\n+ source archive", (d.beta, d.lo95, d.hi95), "#2166ac")]
    fig = plt.figure(figsize=(50 / 25.4, 60 / 25.4), dpi=600)
    fig.text(0.04, 0.965, "Has the South American low-level jet changed?", fontsize=5.2, fontweight="bold", va="top", wrap=True)
    fig.text(0.04, 0.875, "Juan Nesis*", fontsize=3.6, va="top", color="#444")
    texto = ("Jet counts in radiosonde profiles rose 2.4 points per decade, but changes in archives and wind measurements explain part "
             "of it and the rest cannot be separated from them. The strongest jets show no trend. ERA5 agreement with soundings changed abruptly near 1993.")
    fig.text(0.04, 0.83, "\n".join(textwrap.wrap(texto, 54)), fontsize=3.4, va="top", color="#222", linespacing=1.15)
    ax = fig.add_axes([0.40, 0.09, 0.56, 0.44])
    for k, (lab, (b, lo, hi), c) in enumerate(filas[::-1]):
        ax.errorbar(b, k, xerr=[[b - lo], [hi - b]], fmt="o", color=c, ms=2.2, elinewidth=0.7, capsize=1)
    ax.axvline(0, color="#555", lw=0.5)
    ax.set_yticks(range(3), [f[0] for f in filas[::-1]], fontsize=3.4)
    ax.set_ylim(-0.6, 2.6)
    ax.tick_params(axis="x", labelsize=3.4, length=1.5, width=0.4, pad=1)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel("Trend (pp decade$^{-1}$), 95 % interval", fontsize=3.4, labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.4)
    fig.savefig("figuras/jet/graphical_abstract.png", dpi=600)
    print(len(texto.split()), "palabras de texto")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
