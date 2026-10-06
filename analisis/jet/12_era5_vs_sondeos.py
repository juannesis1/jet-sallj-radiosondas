"""Paper 2, paso 12: ¿cómo ve ERA5 el jet que miden los sondeos, y cambia eso con el tiempo?

Perfiles de ERA5 (paso 2: 12 niveles de 1000 a 700 hPa, 12 UTC) en el punto de grilla de cada estación. Se descartan los
niveles por debajo de la superficie (presión de superficie = mediana de la de los sondeos de la estación) y se aplican
los mismos criterios que a los sondeos:
  - fijos: 850/700 hPa (Bonner 1, Salio et al. 2002);
  - perfil: máximo entre la superficie y 250 hPa sobre ella y caída hasta 300 hPa sobre ella (o hasta 700 hPa, el tope
    disponible), con todos los niveles de ERA5 (resolución constante en el tiempo).
Día a día contra el sondeo del mismo día (fijos con fijos; perfil de ERA5 contra perfil completo del sondeo): POD, FAR,
frecuencias y tendencias por período (1980-1995, 1996-2010, 2011-2025). Oct-mar.
Salida: analisis/jet/12_resumen.txt y 12_diario.parquet
"""
import glob
import os

import numpy as np
import pandas as pd

NOMBRES = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087576": "Ezeiza", "ARM00087623": "Santa Rosa",
           "BRM00083827": "Foz", "BRM00083928": "Uruguaiana", "BRM00083612": "Campo Grande", "BRM00083362": "Cuiabá",
           "BRM00083208": "Vilhena", "BRM00082824": "Porto Velho", "BRM00083768": "Londrina"}


def detectar(g, p_sup):
    """g: perfil ERA5 de un instante (nivel, u, v). Devuelve (jet_fijos, jet_perfil)."""
    g = g[g.nivel <= p_sup].sort_values("nivel", ascending=False)
    w = np.hypot(g.u.values, g.v.values)
    d = (np.degrees(np.arctan2(-g.u.values, -g.v.values)) + 360) % 360
    p = g.nivel.values
    norte = lambda x: (x >= 292.5) | (x <= 67.5)
    i8, i7 = np.where(p == 850)[0], np.where(p == 700)[0]
    fijos = bool(len(i8) and len(i7) and w[i8[0]] >= 12 and w[i8[0]] - w[i7[0]] >= 6 and norte(d[i8[0]]))
    dp = p_sup - p
    bajo = dp <= 250
    perfil = False
    if bajo.sum() >= 2:
        k = np.where(bajo)[0][np.argmax(w[bajo])]
        arriba = (dp > dp[k]) & (dp <= 300)
        if arriba.any():
            perfil = bool(w[k] >= 12 and w[k] - w[arriba].min() >= 6 and ((d[k] >= 292.5) or (d[k] <= 67.5)))
    return fijos, perfil


def main():
    e = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob("data/era5_perfiles/*.parquet")))
    e["fecha"] = pd.to_datetime(e.fecha)
    e = e[(e.fecha.dt.hour == 12) & e.fecha.dt.month.isin([10, 11, 12, 1, 2, 3])]
    filas = []
    for sid, nom in NOMBRES.items():
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[s.index.hour == 12]
        p_sup = s.p_sup.median()
        w8, w7 = np.hypot(s.u850, s.v850), np.hypot(s.u700, s.v700)
        dr = (np.degrees(np.arctan2(-s.u850, -s.v850)) + 360) % 360
        s_fijos = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).where(w8.notna() & w7.notna())
        s.index = s.index.normalize()
        s_fijos.index = s.index
        for fecha, g in e[e.sid == sid].groupby("fecha"):
            jf, jp = detectar(g, p_sup)
            dia = fecha.normalize()
            filas.append({"sid": sid, "nombre": nom, "fecha": dia, "era_fijos": jf, "era_perfil": jp,
                          "obs_fijos": s_fijos.get(dia, np.nan), "obs_perfil": s.jet_norte.get(dia, np.nan) if dia in s.index else np.nan})
    r = pd.DataFrame(filas)
    r["temp"] = np.where(r.fecha.dt.month >= 10, r.fecha.dt.year + 1, r.fecha.dt.year)
    r.to_parquet("analisis/jet/12_diario.parquet")
    L = [f"ERA5 disponible: temporadas {r.temp.min()}-{r.temp.max()} ({r.temp.nunique()})"]
    r["periodo"] = pd.cut(r.temp, [1979, 1995, 2010, 2025], labels=["1980-1995", "1996-2010", "2011-2025"])
    for crit in ("fijos", "perfil"):
        L.append(f"\nCriterio {crit}: frecuencia obs / ERA5 (%), POD, FAR por período")
        for (nom, per), x in r.dropna(subset=[f"obs_{crit}"]).groupby(["nombre", "periodo"], observed=True):
            o, m = x[f"obs_{crit}"].astype(bool), x[f"era_{crit}"].astype(bool)
            h, mi, fa = (o & m).sum(), (o & ~m).sum(), (~o & m).sum()
            L.append(f"  {nom:13s} {per}: obs {100 * o.mean():5.1f}  ERA5 {100 * m.mean():5.1f}  POD {h / max(h + mi, 1):.2f}  "
                     f"FAR {fa / max(h + fa, 1):.2f}  n {len(x)}")
    open("analisis/jet/12_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
