"""Paper 2, paso 3: climatología observada del jet del norte y control de homogeneidad del viento de los sondeos.

Con los perfiles del paso 1 (31 estaciones, 1979-2025):
  (a) frecuencia de jet del norte por estación, mes y hora (00 y 12 UTC donde hay ambas), altura y velocidad del núcleo;
  (b) tendencia observada de la frecuencia de oct-mar a 12 UTC (Sen + Mann-Kendall Hamed-Rao), 1980-2025;
  (c) homogeneidad del viento (Gruber y Haimberger, 2008): serie mensual de la componente meridional a 850 hPa de cada
      estación menos la media de sus vecinas (< 1000 km, r > 0.5 de las anomalías), segmentada con SNHT; se reportan
      los quiebres y su magnitud. Un quiebre grande en una estación invalida su tendencia observada.
  (d) cantidad de sondeos por año en la región (proxy del sistema de observación de altura).
Salidas: analisis/jet/03_resumen.txt, analisis/jet/03_estaciones.csv, analisis/jet/03_sondeos_por_anio.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from estadistica import mk_hamed_rao, sen  # noqa: E402


def snht(x):
    """Estadístico SNHT de un único quiebre (Alexandersson 1986) y su posición."""
    z = (x - x.mean()) / x.std()
    n = len(z)
    t = np.array([k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2 for k in range(12, n - 12)])
    return t.max(), int(np.argmax(t)) + 12


def dist(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 6371 * np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1))


def main():
    est = pd.read_csv("data/igra_sa/estaciones.txt", sep=r"\s+", header=None, usecols=range(3),
                      names=["sid", "lat", "lon"]).set_index("sid")
    d = {s: pd.read_parquet(f) for f in sorted(glob.glob("data/igra_sa/perfiles/*.parquet"))
         for s in [os.path.basename(f)[:11]] if s in est.index}  # red de 31 estaciones (las del norte: sección 4.6)
    filas, L = [], []
    mens = {}
    for s, x in d.items():
        x = x[(x.index.year >= 1980) & (x.index.year <= 2025)]
        h12 = x[x.index.hour == 12]
        cal = h12[h12.index.month.isin([10, 11, 12, 1, 2, 3])].dropna(subset=["jet_norte"])
        temp = np.where(cal.index.month >= 10, cal.index.year + 1, cal.index.year)
        g = cal.jet_norte.astype(float).groupby(temp)
        f = (g.mean()[g.size() >= 60] * 100).loc[1980:2025]
        tr, p = (sen(f.index.values.astype(float), f.values) * 10, mk_hamed_rao(f.index.values.astype(float), f.values)[1]) \
            if len(f) >= 20 else (np.nan, np.nan)
        j = cal[cal.jet_norte == True]
        fila = {"sid": s, "lat": est.lat[s], "lon": est.lon[s], "n12_calido": len(cal), "temporadas": len(f),
                "frec_jet_12": 100 * cal.jet_norte.astype(float).mean(), "tend_pp_dec": tr, "p_tend": p,
                "nucleo_v_med": j.nucleo_v.median(), "nucleo_dp_med": j.nucleo_dp.median()}
        h00 = x[(x.index.hour == 0) & x.index.month.isin([10, 11, 12, 1, 2, 3])].dropna(subset=["jet_norte"])
        if len(h00) > 500:
            fila["frec_jet_00"] = 100 * h00.jet_norte.astype(float).mean()
        if "v850" not in x:
            filas.append(fila)
            continue
        m = x[x.index.hour == 12].v850.resample("MS").mean()
        mens[s] = m - m.groupby(m.index.month).transform("mean")
        filas.append(fila)
    r = pd.DataFrame(filas).set_index("sid")
    # (c) homogeneidad del viento meridional a 850 hPa contra vecinas
    M = pd.DataFrame(mens)
    quiebres = {}
    for s in r.index:
        dd = dist(r.lat[s], r.lon[s], r.lat.values, r.lon.values)
        vec = [o for o, k in zip(r.index, dd) if o != s and k < 1000 and M[s].corr(M[o]) > 0.5]
        if len(vec) < 2:
            continue
        dif = (M[s] - M[vec].mean(axis=1)).dropna()
        if len(dif) < 120:
            continue
        T, k = snht(dif.values)
        quiebres[s] = (T, dif.index[k], dif.iloc[k:].mean() - dif.iloc[:k].mean(), len(vec))
    r["snht"] = pd.Series({s: q[0] for s, q in quiebres.items()})
    r["quiebre"] = pd.Series({s: q[1].strftime("%Y-%m") for s, q in quiebres.items()})
    r["salto_v850"] = pd.Series({s: q[2] for s, q in quiebres.items()})
    r.round(3).to_csv("analisis/jet/03_estaciones.csv")
    # (d) sondeos por año en la región
    n = pd.DataFrame({s: x.index.year.value_counts() for s, x in d.items()}).fillna(0).sort_index()
    n.loc[1979:2025].sum(axis=1).rename("sondeos").to_csv("analisis/jet/03_sondeos_por_anio.csv")
    L.append("Jet del norte (Bonner 1, perfil completo), oct-mar, 1980-2025")
    L.append(r[["lat", "lon", "temporadas", "frec_jet_12", "frec_jet_00", "tend_pp_dec", "p_tend", "nucleo_v_med",
                "nucleo_dp_med", "snht", "quiebre", "salto_v850"]].round(2).sort_values("lat").to_string())
    L.append("\nSNHT crítico ~9-11 (95 %, n 300-550): valores mayores indican quiebre en el viento meridional a 850 hPa.")
    open("analisis/jet/03_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
