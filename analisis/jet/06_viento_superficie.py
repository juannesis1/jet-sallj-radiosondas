"""Paper 2, paso 6: viento de superficie de los aeropuertos como prueba independiente de los sondeos.

Si el jet bajo hubiera aumentado de verdad en Santa Rosa y Córdoba, el viento nocturno del norte en superficie
(que el jet acompaña) debería cambiar de forma gradual; si el aumento en los sondeos es un artefacto del equipo, el
viento de superficie no debería mostrar el salto en las fechas de los saltos de los sondeos (Santa Rosa 2000-02,
Córdoba 2002-11).
Datos: NOAA ISD (global-hourly), campo WND (dirección, velocidad×10 m/s). Oct-mar, noche 03-09 UTC.
Métricas por temporada: velocidad media nocturna y frecuencia de horas con viento del norte (292.5-67.5°) ≥ 5 m/s.
Salida: analisis/jet/06_resumen.txt
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from estadistica import mk_hamed_rao, sen  # noqa: E402

ESTACIONES = {"87623099999": ("Santa Rosa", "2000-02"), "87344099999": ("Córdoba", "2002-11"),
              "87155099999": ("Resistencia", None), "87576099999": ("Ezeiza", None)}


def leer(usaf):
    partes = []
    for f in sorted(glob.glob(f"data/isd_full/{usaf}_*.csv")):
        d = pd.read_csv(f, usecols=["DATE", "WND"], dtype=str)
        w = d.WND.str.split(",", expand=True)
        x = pd.DataFrame({"dir": pd.to_numeric(w[0], errors="coerce").values, "qd": w[1].values,
                          "v": pd.to_numeric(w[3], errors="coerce").values / 10, "qv": w[4].values},
                         index=pd.to_datetime(d.DATE).values)
        partes.append(x)
    x = pd.concat(partes)
    x = x[(x.v < 99) & x.qv.isin(["1", "5", "0", "4", "9"])]
    x.loc[x.dir == 999, "dir"] = np.nan
    return x[~x.index.duplicated()]


def snht(z):
    z = (z - z.mean()) / z.std()
    n = len(z)
    t = np.array([k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2 for k in range(12, n - 12)])
    return t.max(), int(np.argmax(t)) + 12


def main():
    L = ["Viento de superficie nocturno (03-09 UTC), oct-mar"]
    for usaf, (nom, salto) in ESTACIONES.items():
        x = leer(usaf)
        x = x[x.index.month.isin([10, 11, 12, 1, 2, 3]) & x.index.hour.isin(range(3, 10)) & (x.index.year >= 1980)]
        norte = ((x.dir >= 292.5) | (x.dir <= 67.5)) & (x.v >= 5)
        t = np.where(x.index.month >= 10, x.index.year + 1, x.index.year)
        g = x.groupby(t)
        ok = g.size() >= 300
        vel = g.v.mean()[ok]
        fn = (norte.groupby(t).mean() * 100)[ok]
        lin = []
        for nombre, s in (("velocidad", vel), ("% norte≥5", fn)):
            xs = s.index.values.astype(float)
            lin.append(f"{nombre} tend {sen(xs, s.values) * 10:+.2f}/déc (p {mk_hamed_rao(xs, s.values)[1]:.2f})")
        m = x.v.resample("MS").mean().dropna()
        m = m[m.index.month.isin([10, 11, 12, 1, 2, 3])]
        m = m - m.groupby(m.index.month).transform("mean")
        T, k = snht(m.values)
        lin.append(f"salto más fuerte en velocidad: {m.index[k]:%Y-%m} (T {T:.1f}, {m.iloc[k:].mean() - m.iloc[:k].mean():+.2f} m/s)")
        if salto:
            fa = pd.Timestamp(salto)
            mm = norte.groupby(x.index.to_period("M")).mean() * 100
            mm.index = mm.index.to_timestamp()
            ant = mm[(mm.index < fa) & (mm.index >= fa - pd.DateOffset(years=5))].mean()
            des = mm[(mm.index >= fa) & (mm.index < fa + pd.DateOffset(years=5))].mean()
            lin.append(f"% norte≥5, 5 años antes/después del salto del sondeo ({salto}): {ant:.1f} → {des:.1f}")
        L.append(f"\n{nom} ({len(vel)} temporadas): " + "; ".join(lin))
    open("analisis/jet/06_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
