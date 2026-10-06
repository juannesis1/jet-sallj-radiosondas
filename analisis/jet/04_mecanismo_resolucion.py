"""Paper 2, paso 4: ¿la tendencia del jet observado viene de la resolución vertical de los sondeos?

Experimento controlado: el mismo criterio de perfil completo (Bonner 1) aplicado
  (a) a todos los niveles reportados (jet_norte) y
  (b) solo a los niveles estándar, comunes a todo el período (superficie, 1000, 925, 850, 700, 500 hPa; jet_norte_std).
Si la tendencia de (a) desaparece en (b), la causa es el aumento de niveles reportados, no un cambio del jet.
Además: número mediano de niveles en 0-250 hPa sobre la superficie por año, año en que cambia la fuente dominante de los
datos (código de origen IGRA) y salto de la frecuencia de jet en (a) alrededor de ese cambio.
Oct-mar, 12 UTC, 1980-2025. Salida: analisis/jet/04_resumen.txt y 04_estaciones.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from estadistica import mk_hamed_rao, sen  # noqa: E402


def temporada(idx):
    return np.where(idx.month >= 10, idx.year + 1, idx.year)


def tendencia(x, t):
    g = x.astype(float).groupby(t)
    f = (g.mean()[g.count() >= 60] * 100).loc[1980:2025]
    if len(f) < 20:
        return np.nan, np.nan, f
    xs = f.index.values.astype(float)
    return sen(xs, f.values) * 10, mk_hamed_rao(xs, f.values)[1], f


def main():
    filas = []
    red = set(pd.read_csv("data/igra_sa/estaciones.txt", sep=r"\s+", header=None, usecols=[0])[0])
    for f in sorted(glob.glob("data/igra_sa/perfiles/*.parquet")):
        s = os.path.basename(f)[:11]
        if s not in red:  # red de 31 estaciones
            continue
        d = pd.read_parquet(f)
        d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3]) & (d.index.year >= 1980)]
        d = d.dropna(subset=["jet_norte", "jet_norte_std"])
        if len(d) < 2000:
            continue
        t = temporada(d.index)
        ta, pa, fa = tendencia(d.jet_norte, t)
        tb, pb, fb = tendencia(d.jet_norte_std, t)
        niv = d.n_niveles.groupby(t).median()
        dom = d.fuente.groupby(t).agg(lambda x: x.value_counts().index[0])
        cambio = dom[dom != dom.shift()].index[1:].tolist()
        filas.append({"sid": s, "temporadas": len(fa), "frec_todos": fa.mean(), "frec_std": fb.mean(),
                      "tend_todos": ta, "p_todos": pa, "tend_std": tb, "p_std": pb,
                      "niveles_1980s": niv.loc[1980:1989].median(), "niveles_2010s": niv.loc[2010:2025].median(),
                      "cambios_fuente": ";".join(f"{y}:{dom[y]}" for y in cambio)})
    r = pd.DataFrame(filas).set_index("sid")
    r.round(2).to_csv("analisis/jet/04_estaciones.csv")
    sig_a = ((r.p_todos < 0.05) & (r.tend_todos > 0)).sum()
    sig_b = ((r.p_std < 0.05) & (r.tend_std > 0)).sum()
    L = ["Jet del norte oct-mar 12 UTC: perfil con todos los niveles reportados vs solo niveles estándar",
         r.drop(columns="cambios_fuente").round(2).to_string(),
         f"\nEstaciones con tendencia positiva significativa: todos los niveles {sig_a}/{len(r)}; niveles estándar {sig_b}/{len(r)}",
         f"Mediana de tendencias (pp/déc): todos {r.tend_todos.median():+.2f}; estándar {r.tend_std.median():+.2f}",
         "\nCambios de fuente dominante por estación:", r.cambios_fuente.to_string()]
    open("analisis/jet/04_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
