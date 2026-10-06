"""Paper 2, paso 28: Tabla 1, estaciones de IGRA usadas y análisis en que entra cada una.

Para cada estación: nombre, latitud, longitud, altitud (lista de estaciones de IGRA v2), período con sondeos de 12 UTC en oct-mar, número de
sondeos de 12 UTC en oct-mar 1980-2025, niveles medianos en los 250 hPa inferiores en 1980-1995 y 2005-2025, y los análisis en que entra
(T: tendencia por estación de la Fig. 1, ≥ 20 temporadas; P: agrupado de 10 estaciones; E: comparación con ERA5; B: pares de 00 y 12 UTC de la sección 4.5; N: rama norte de la sección 4.6).
Salida: analisis/jet/28_tabla_estaciones.csv y .md
"""
import glob
import os

import numpy as np
import pandas as pd

AGRUPADO = ["ARM00087155", "ARM00087344", "ARM00087623", "ARM00087576", "BRM00083827", "BRM00083612", "BRM00083928", "BRM00083362",
            "BRM00083208", "BRM00082824"]
ERA5 = AGRUPADO + ["BRM00083768"]
NORTE = ["VEM00080413", "VEM00080447", "COM00080241", "COM00080035", "UCM00078988", "TDM00078970", "BRM00082022"]


def main():
    lst = pd.read_fwf("data/igra/igra2-station-list.txt", colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (41, 71)], header=None,
                      names=["sid", "lat", "lon", "elev", "nombre"]).set_index("sid")
    red = list(pd.read_csv("data/igra_sa/estaciones.txt", sep=r"\s+", header=None, usecols=[0])[0])
    tend = set(pd.read_csv("analisis/jet/03b_tendencias.csv").sid)                       # estaciones con tendencia (≥ 20 temporadas)
    BR7 = {"BRM00083827", "BRM00083612", "BRM00083362", "BRM00083928", "BRM00083768", "BRM00083208", "BRM00082824"}
    filas = []
    for f in sorted(glob.glob("data/igra_sa/perfiles/*.parquet")):
        sid = os.path.basename(f)[:11]
        d = pd.read_parquet(f)
        d = d[d.index.month.isin([10, 11, 12, 1, 2, 3]) & (d.index.year >= 1980)]
        d12 = d[d.index.hour == 12]
        nv = lambda x: f"{x.n_niveles.median():.0f}" if len(x) >= 30 else "–"  # noqa: E731
        an = ("T " if sid in tend else "") + ("P " if sid in AGRUPADO else "") + ("E " if sid in ERA5 else "") + \
             ("B " if sid in BR7 else "") + ("N" if sid in NORTE else "")
        filas.append({"IGRA": sid, "Estación": lst.nombre[sid].strip().title(), "Lat (°)": round(lst.lat[sid], 2), "Lon (°)": round(lst.lon[sid], 2),
                      "Alt (m)": int(lst.elev[sid]), "Período 12 UTC": f"{d12.index.year.min()}–{d12.index.year.max()}" if len(d12) else "–",
                      "Sondeos 12 UTC": len(d12), "Niveles 1980–95": nv(d12[d12.index.year <= 1995]), "Niveles 2005–25": nv(d12[d12.index.year >= 2005]),
                      "Análisis": an.strip()})
    T = pd.DataFrame(filas)
    T = T[T.IGRA.isin(red + NORTE)].sort_values("Lat (°)", ascending=False)
    T.to_csv("analisis/jet/28_tabla_estaciones.csv", index=False)
    md = ["| " + " | ".join(T.columns) + " |", "|" + "|".join("---" for _ in T.columns) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in T.itertuples(index=False)]
    open("analisis/jet/28_tabla_estaciones.md", "w").write("\n".join(md) + "\n")
    print(T.to_string(index=False))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
