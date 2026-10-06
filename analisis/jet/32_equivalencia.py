"""Paper 2, paso 32: ¿qué se puede decir de la tendencia de los jets profundos? Marco de equivalencia, punto de quiebre estimado y
dejar una estación afuera (respuesta a la revisión de la ronda 3, comentarios 4, 12 y 13).

(a) Un solo estimador principal: modelo de probabilidad lineal agrupado del jet con niveles fijos (efectos de mes, estación y fuente de archivo),
    1980-2025, con IC de 90 % y de 95 % por bootstrap de temporadas (mismo generador que los pasos 10 y 22, así que las cifras coinciden).
(b) Margen de equivalencia adoptado por convención (no preregistrado; elegido después de ver los primeros resultados): ±10 % de la frecuencia media por década (≈ ±0.3 pp por década), y variante de ±15 %. Se informa si el IC de
    90 % (prueba de dos pruebas unilaterales, TOST, al 5 %) cae dentro del margen (equivalencia) o no.
(c) Punto de quiebre estimado: serie agrupada de anomalías estacionales (estaciones con ≥ 60 sondeos por temporada, ≥ 3 estaciones por año); ajuste
    lineal por tramos continuo con quiebre en t0 entre 1988 y 2012, t0 elegido por mínimo error cuadrático; IC de t0 y pendientes por bootstrap de
    temporadas. Se compara con la recta por AIC.
(d) Dejar una estación afuera: tendencia agrupada con fuente, estimaciones puntuales e IC (100 réplicas).
Salida: analisis/jet/32_resumen.txt
"""
import os

import numpy as np
import pandas as pd

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande", "BRM00083928": "Uruguaiana", "BRM00083362": "Cuiabá",
       "BRM00083208": "Vilhena", "BRM00082824": "Porto Velho"}


def celdas(d, y):
    """Agrega por (estación, mes, fuente, temporada): la regresión ponderada por n de las medias de celda es idéntica a la de los sondeos."""
    x = d.dropna(subset=[y])
    g = x.groupby(["sid", "mes", "fuente", "temp"])[y].agg(["sum", "count"]).reset_index()
    return g


def beta(d, y, fuente=True):
    g = celdas(d, y)
    X = [pd.get_dummies(g.mes).values.astype(float), ((g.temp.values - 2000) / 10.0)[:, None], pd.get_dummies(g.sid, drop_first=True).values.astype(float)]
    if fuente:
        X.append(pd.get_dummies(g.fuente, drop_first=True).values.astype(float))
    w = np.sqrt(g["count"].values)
    X = np.hstack(X) * w[:, None]
    yy = g["sum"].values / np.sqrt(g["count"].values)
    return 100 * np.linalg.lstsq(X, yy, rcond=None)[0][len(np.unique(g.mes))]


def boot(d, y, fuente=True, n=2000, semilla=123):
    r = np.random.default_rng(semilla)
    temps = d.temp.unique()
    g = {t: d[d.temp == t] for t in temps}
    out = [beta(pd.concat([g[t] for t in r.choice(temps, len(temps))]), y, fuente) for _ in range(n)]
    return np.percentile(out, [5, 95]), np.percentile(out, [2.5, 97.5])


def tramos(t, y, t0):
    X = np.column_stack([np.ones_like(t), t - t0, np.maximum(t - t0, 0)])
    c, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c, float(np.sum((y - X @ c) ** 2))


def ajustar(t, y):
    mejor = min(((tramos(t, y, t0)[1], t0) for t0 in range(1988, 2013)))
    return mejor[1], tramos(t, y, mejor[1])[0]


def main():
    F = pd.read_parquet("analisis/jet/22_fijos.parquet")
    L = []
    media = 100 * F.fijos.mean()
    L.append(f"Frecuencia media de jets profundos (10 estaciones): {media:.2f} %")
    res = {}
    for y, lab in (("fijos", "viento original"), ("fijos_adj", "viento ajustado por los quiebres propios de cada estación (paso 22)")):
        b = beta(F, y, True)
        (lo, hi), (lo95, hi95) = boot(F, y, True)
        (lo2, hi2), _ = boot(F, y, True, semilla=124)
        res[y] = (b, lo, hi, lo95, hi95)
        L.append(f"    Error de Monte Carlo (otra semilla, 2000 réplicas): IC 90 % [{lo2:+.3f},{hi2:+.3f}] frente a [{lo:+.3f},{hi:+.3f}]")
        L.append(f"(a) Tendencia agrupada de jets profundos, 1980-2025, efectos de mes, estación y fuente, {lab}: {b:+.2f} pp/déc; "
                 f"IC 90 % [{lo:+.2f},{hi:+.2f}]; IC 95 % [{lo95:+.2f},{hi95:+.2f}]")
    L.append("(b) Marco de equivalencia (margen ADOPTADO por convención, no preregistrado: ±10 % de la media por década; variantes ±15 y ±20 %):")
    for y, (b, lo, hi, lo95, hi95) in res.items():
        for pct in (10, 15, 20):
            d = media * pct / 100
            ok = lo > -d and hi < d
            L.append(f"    {y:10s} margen ±{pct} %/déc = ±{d:.2f} pp/déc: IC 90 % [{lo:+.2f},{hi:+.2f}] → {'equivalencia (TOST, 5 %)' if ok else 'no se establece equivalencia'}")
        L.append(f"    {y:10s} cota superior: IC 90 % {hi:+.2f} pp/déc = {100 * hi / media:+.0f} %/déc; IC 95 % {hi95:+.2f} = {100 * hi95 / media:+.0f} %/déc")
    # (b2) diferencia de tendencias perfil completo − jets profundos sobre los mismos sondeos, con IC apareado por temporadas
    P = pd.read_parquet("analisis/jet/24_perfil.parquet")
    Fi = F.copy()
    Fi = Fi.rename_axis(None).assign(fecha=Fi.index.normalize())
    M = P.merge(Fi[["sid", "fecha", "fijos"]], on=["sid", "fecha"], how="inner")
    M["dif"] = M.crudo - M.fijos
    ok = M.dropna(subset=["crudo", "fijos"])
    bp, bf, bd = beta(ok, "crudo", True), beta(ok, "fijos", True), beta(ok, "dif", True)
    (l1, h1), (l2, h2) = boot(ok, "dif", True)
    L.append(f"(b2) Mismos sondeos evaluables ({len(ok)}): tendencia con fuente del perfil completo {bp:+.2f}, de los jets profundos {bf:+.2f}; diferencia"
             f" (perfil completo − profundos, es decir, jets sólo perfil completo) {bd:+.2f} pp/déc, IC 90 % [{l1:+.2f},{h1:+.2f}], IC 95 % [{l2:+.2f},{h2:+.2f}]")
    # (c) quiebre estimado, con la serie original y con la ajustada por los quiebres de viento propios de cada estación
    for col, lab in (("fijos", "viento original"), ("fijos_adj", "viento ajustado")):
        g = F.groupby(["sid", "temp"])[col].agg(["mean", "count"])
        g = g[g["count"] >= 60]["mean"] * 100
        an = g - g.groupby("sid").transform("mean")
        cnt = an.groupby("temp").size()
        serie = an.groupby("temp").mean()[cnt >= 3]
        t, y = serie.index.values.astype(float), serie.values
        t0, c = ajustar(t, y)
        sse1 = float(np.sum((y - np.polyval(np.polyfit(t, y, 1), t)) ** 2))
        sse2 = tramos(t, y, t0)[1]
        n = len(t)
        aic = lambda sse, k: n * np.log(sse / n) + 2 * k  # noqa: E731
        r = np.random.default_rng(32)
        bs = []
        for _ in range(500):
            i = r.integers(0, n, n)
            if len(np.unique(t[i])) < 8:
                continue
            t0b, cb = ajustar(t[i], y[i])
            bs.append((t0b, 10 * cb[1], 10 * (cb[1] + cb[2])))
        bs = np.array(bs)
        L.append(f"(c) Serie agrupada de anomalías, {lab} ({n} temporadas). Ajuste lineal por tramos: quiebre estimado en {t0}; pendiente antes {10 * c[1]:+.2f} pp/déc,"
                 f" después {10 * (c[1] + c[2]):+.2f}; AIC {aic(sse2, 4):.1f} frente a {aic(sse1, 2):.1f} de la recta (menor = mejor)")
        L.append(f"    bootstrap de temporadas ({len(bs)} réplicas): t0 mediana {np.median(bs[:, 0]):.0f}, IC 90 % [{np.percentile(bs[:, 0], 5):.0f},{np.percentile(bs[:, 0], 95):.0f}];"
                 f" pendiente antes {np.median(bs[:, 1]):+.2f} [{np.percentile(bs[:, 1], 5):+.2f},{np.percentile(bs[:, 1], 95):+.2f}];"
                 f" después {np.median(bs[:, 2]):+.2f} [{np.percentile(bs[:, 2], 5):+.2f},{np.percentile(bs[:, 2], 95):+.2f}]"
                 f" (= {100 * np.median(bs[:, 2]) / media:+.0f} %/déc)")
    L.append("    Estaciones por temporada en la serie agrupada: " + ", ".join(f"{int(a)}–{int(b_)}: {cnt[(cnt.index >= a) & (cnt.index <= b_)].mean():.1f}"
                                                                         for a, b_ in ((1980, 1992), (1993, 2000), (2001, 2025))))
    # (d) dejar una estación afuera
    L.append("(d) Dejando una estación afuera (con fuente; viento ajustado), pp/déc [IC 90 %, 100 réplicas]:")
    pts = []
    for sid, nom in EST.items():
        x = F[F.sid != sid]
        bb = beta(x, "fijos_adj", True)
        (l, h), _ = boot(x, "fijos_adj", True, n=100)
        pts.append(bb)
        L.append(f"    sin {nom:13s} {bb:+.2f} [{l:+.2f},{h:+.2f}]")
    L.append(f"    rango de las estimaciones puntuales: {min(pts):+.2f} a {max(pts):+.2f}")
    open("analisis/jet/32_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
