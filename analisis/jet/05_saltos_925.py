"""Paper 2, paso 5: ¿el aumento reciente del jet bajo (1998-2025) es un cambio de equipo o un cambio real?

Pruebas sobre el viento a 925 hPa de las 12 UTC, oct-mar (sondeos con 925 reportado):
  (a) "huella" del sistema de medición: resolución del viento reportado. Si la velocidad viene de nudos enteros
      convertidos a m/s, sus valores caen en múltiplos de 0.5144; un cambio de equipo o de software cambia esa fracción.
      Se reporta, por año, la fracción de velocidades múltiplos de nudos enteros y la de valores con décimas arbitrarias;
  (b) saltos (SNHT de un quiebre) en la serie mensual de la velocidad media a 925 hPa de cada estación y en la
      diferencia con sus vecinas (< 800 km): un cambio de equipo deja un escalón local en una fecha; un cambio
      climático es gradual y compartido por las vecinas;
  (c) la frecuencia del jet (niveles estándar) antes y después del salto.
Salida: analisis/jet/05_resumen.txt
"""
import glob
import os

import numpy as np
import pandas as pd

ESTUDIO = ["ARM00087623", "BRM00083827", "BRM00083612", "ARM00087344", "ARM00087155", "ARM00087576",
           "BRM00083928", "BRM00083840", "BRM00083768"]


def snht(x):
    z = (x - x.mean()) / x.std()
    n = len(z)
    t = np.array([k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2 for k in range(12, n - 12)])
    return t.max(), int(np.argmax(t)) + 12


def dist(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a.lat, a.lon, b.lat, b.lon))
    return 6371 * np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1))


def main():
    est = pd.read_csv("data/igra_sa/estaciones.txt", sep=r"\s+", header=None, usecols=range(3),
                      names=["sid", "lat", "lon"]).set_index("sid")
    w, jet, L = {}, {}, []
    L.append("(a) Fracción de velocidades a 925 hPa que son nudos enteros convertidos (múltiplos de 0.5144 m/s), por quinquenio; sólo descriptivo: no tiene hipótesis nula ni se usa como evidencia de cambio de equipo")
    for f in sorted(glob.glob("data/igra_sa/perfiles/*.parquet")):
        s = os.path.basename(f)[:11]
        if s not in est.index:  # red de 31 estaciones
            continue
        d = pd.read_parquet(f)
        if "u925" not in d:
            continue
        d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3]) & (d.index.year >= 1990)]
        d = d[d.u925.notna()]
        if len(d) < 1500:
            continue
        v = np.hypot(d.u925, d.v925)
        w[s] = v.resample("MS").mean()
        jet[s] = d.jet_norte_std.astype(float).resample("MS").mean()
        if s in ESTUDIO:
            nud = (np.abs(v / 0.5144 - np.round(v / 0.5144)) < 0.02).groupby((d.index.year // 5) * 5).mean()
            L.append(f"  {s}: " + " ".join(f"{k}:{100 * x:.0f}%" for k, x in nud.items()))
    W = pd.DataFrame(w).loc["1995":"2025"]
    J = pd.DataFrame(jet).loc["1995":"2025"]
    L.append("\n(b) Saltos en la velocidad a 925 hPa (SNHT; crítico 95 % ≈ 9-10)")
    L.append("estación      propia: T  fecha    salto(m/s) | vs vecinas: T  fecha   salto(m/s)  n_vec | jet_std antes→después (%)")
    for s in ESTUDIO:
        if s not in W:
            continue
        x = W[s].dropna()
        x = x - x.groupby(x.index.month).transform("mean")
        T1, k1 = snht(x.values)
        vec = [o for o in W if o != s and dist(est.loc[s], est.loc[o]) < 800]
        if vec:
            ref = W[vec].sub(W[vec].groupby(W.index.month).transform("mean")).mean(axis=1)
            dif = (x - ref.reindex(x.index)).dropna()
            T2, k2 = snht(dif.values)
            s2 = f"{T2:6.1f} {dif.index[k2]:%Y-%m} {dif.iloc[k2:].mean() - dif.iloc[:k2].mean():+6.2f}  {len(vec):3d}"
        else:
            s2 = "   sin vecinas"
        fecha = x.index[k1]
        j = J[s].dropna()
        ja, jd = 100 * j[j.index < fecha].mean(), 100 * j[j.index >= fecha].mean()
        L.append(f"{s}  {T1:6.1f} {fecha:%Y-%m} {x.iloc[k1:].mean() - x.iloc[:k1].mean():+6.2f} | {s2} | {ja:5.1f}→{jd:5.1f}")
    open("analisis/jet/05_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
