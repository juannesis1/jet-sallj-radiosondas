"""Paper 2, paso 36: (a) tendencia del perfil completo en ventanas fijas y (b) perfil de verosimilitud de la fecha del escalón argentino.

(a) Tendencia agrupada (modelo de probabilidad lineal con efectos de mes, estación y fuente; bootstrap de temporadas, 1000 réplicas, IC 95 %)
    para 1980-2025, 1993-2025, 2001-2025 y 2006-2025, con el viento original y ajustado por los quiebres, con y sin Santa Rosa.
(b) Para cada fecha candidata t0 (temporadas 1990-2012): modelo jet ~ mes + estación + fuente + tendencia + Argentina × 1[temporada ≥ t0];
    se reporta el AIC (gaussiano) y el escalón estimado; la fecha preferida y el conjunto con ΔAIC < 2. El AIC no se penaliza por la búsqueda de
    t0 (se comparan modelos con el mismo número de parámetros): sirve para ver qué tan nítida es la fecha, no como prueba de existencia.
Salida: analisis/jet/36_resumen.txt
"""
import os

import numpy as np
import pandas as pd

ARG = ["ARM00087155", "ARM00087344", "ARM00087623", "ARM00087576"]


def diseno(d, extra, paso=None):
    cols = [pd.get_dummies(d.mes).values.astype(float), ((d.temp.values - 2000) / 10.0)[:, None], pd.get_dummies(d.sid, drop_first=True).values.astype(float)]
    for e in extra:
        cols.append(pd.get_dummies(d[e], drop_first=True).values.astype(float))
    if paso is not None:
        cols.append(paso[:, None])
    return np.hstack(cols)


def tend(d, y):
    x = d.dropna(subset=[y])
    X = diseno(x, ["fuente"])
    return 100 * np.linalg.lstsq(X, x[y].values.astype(float), rcond=None)[0][len(np.unique(x.mes))]


def boot(d, y, n=1000, semilla=36):
    r = np.random.default_rng(semilla)
    temps = d.temp.unique()
    g = {t: d[d.temp == t] for t in temps}
    out = [tend(pd.concat([g[t] for t in r.choice(temps, len(temps))]), y) for _ in range(n)]
    return np.percentile(out, [2.5, 97.5])


def main():
    D = pd.read_parquet("analisis/jet/24_perfil.parquet").reset_index(drop=True)
    L = ["(a) Tendencia del perfil completo (pp/década) con efectos de fuente en ventanas fijas [IC 95 %, 1000 réplicas]"]
    for ini in (1980, 1993, 2001, 2006):
        for nom, q in (("10 estaciones", D), ("sin Santa Rosa", D[D.sid != "ARM00087623"])):
            q = q[q.temp >= ini]
            fila = []
            for y in ("crudo", "ajustado"):
                lo, hi = boot(q, y)
                fila.append(f"{y} {tend(q, y):+.2f} [{lo:+.2f},{hi:+.2f}]")
            L.append(f"  {ini}-2025 {nom:15s} " + " | ".join(fila) + f"  (n {len(q)})")
    L.append("\n(b) Perfil de la fecha del escalón argentino (temporada t0): AIC y escalón (pp), con efecto de fuente")
    x = D.dropna(subset=["crudo"]).reset_index(drop=True)
    res = []
    for t0 in range(1990, 2013):
        paso = ((x.temp >= t0) & x.sid.isin(ARG)).astype(float).values
        X = diseno(x, ["fuente"], paso)
        b, rss, *_ = np.linalg.lstsq(X, x.crudo.values.astype(float), rcond=None)
        r = x.crudo.values - X @ b
        n = len(x)
        aic = n * np.log((r ** 2).sum() / n) + 2 * X.shape[1]
        res.append((t0, aic, 100 * b[-1], 100 * b[len(np.unique(x.mes))]))
    R = pd.DataFrame(res, columns=["t0", "aic", "escalon", "tendencia"])
    R["d_aic"] = R.aic - R.aic.min()
    for r in R.itertuples():
        L.append(f"  t0 = {r.t0}: ΔAIC {r.d_aic:6.1f}; escalón {r.escalon:+.1f} pp; tendencia {r.tendencia:+.2f} pp/déc")
    best = R.loc[R.aic.idxmin()]
    plaus = R[R.d_aic < 2].t0.tolist()
    L.append(f"  Fecha preferida: {int(best.t0)} (escalón {best.escalon:+.1f} pp, tendencia {best.tendencia:+.2f}); fechas con ΔAIC < 2: {plaus}")
    L.append("  Sin escalón: " + f"tendencia {tend(x, 'crudo'):+.2f} pp/déc")
    open("analisis/jet/36_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
