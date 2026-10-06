"""Paper 2, paso 34: ¿el aumento del perfil completo es real? Fuente, eras homogéneas y diferencia en diferencias Argentina–Brasil
(respuesta a las rondas 4-6).

(a) Coeficientes de fuente del modelo agrupado del paso 24 (referencia: GTS), con IC 95 % por bootstrap de temporadas, y su comparación con
    el solapamiento directo del paso 19 (USAF − GTS en las mismas temporadas).
(b) Tendencias del perfil completo por era homogénea: USAF (temporadas 1980-2006) y GTS (2001-2025 y 2006-2025), agrupadas y por país.
(c) Escalón de la red argentina: diferencia en diferencias (DiD) del cambio entre las 8 temporadas antes y las 8 después de 1998, 2000 y 2003
    (fechas de los quiebres de 925 hPa del paso 22; elegidas a partir de los datos: el análisis es exploratorio), Argentina menos Brasil, con bootstrap por temporadas.
    Modelo lineal: jet ~ mes + estación + tendencia + Argentina × (año ≥ t0), en 1980-2025, con y sin fuente.
(d) Tendencia agrupada solo Argentina (4 estaciones) y solo Brasil (6 estaciones), crudo y con fuente.
Salida: analisis/jet/34_resumen.txt
"""
import os

import numpy as np
import pandas as pd

ARG = ["ARM00087155", "ARM00087344", "ARM00087623", "ARM00087576"]
rng_seed = 34


def diseno(d, extra, paso=None):
    cols = [pd.get_dummies(d.mes).values.astype(float), ((d.temp.values - 2000) / 10.0)[:, None],
            pd.get_dummies(d.sid, drop_first=True).values.astype(float)]
    for e in extra:
        cols.append(pd.get_dummies(d[e], drop_first=True).values.astype(float))
    if paso is not None:
        cols.append(paso[:, None])
    return np.hstack(cols)


def coefs(d, y, extra, paso=None):
    x = d.dropna(subset=[y])
    X = diseno(x, extra, None if paso is None else paso.loc[x.index].values if hasattr(paso, "loc") else paso[x.index])
    return np.linalg.lstsq(X, x[y].values.astype(float), rcond=None)[0], x


def tend(d, y, extra):
    b, x = coefs(d, y, extra)
    return 100 * b[len(np.unique(x.mes))]


def boot(d, fun, n=500):
    r = np.random.default_rng(rng_seed)
    temps = d.temp.unique()
    g = {t: d[d.temp == t] for t in temps}
    out = []
    for _ in range(n):
        out.append(fun(pd.concat([g[t] for t in r.choice(temps, len(temps))])))
    out = np.array(out)
    return np.nanpercentile(out, [2.5, 5, 95, 97.5], axis=0)


def main():
    D = pd.read_parquet("analisis/jet/24_perfil.parquet").reset_index(drop=True)
    D["arg"] = D.sid.isin(ARG).astype(float)
    L = ["(a) Coeficientes de fuente (pp, respecto de GTS) del modelo agrupado de perfil completo, 1980-2025, efectos de mes y estación, con tendencia"]
    fuentes = ["usaf-ds3", "ncdc6314", "ncdc6316", "ncdc6322"]
    x = D.dropna(subset=["crudo"])
    X = diseno(x, ["fuente"])
    nombres = sorted(x.fuente.unique())          # get_dummies(drop_first=True) elimina la primera, alfabéticamente
    ref = nombres[0]
    b = np.linalg.lstsq(X, x.crudo.values.astype(float), rcond=None)[0]
    k0 = len(np.unique(x.mes)) + 1 + (x.sid.nunique() - 1)
    cf = dict(zip(nombres[1:], 100 * b[k0:k0 + len(nombres) - 1]))
    # re-expresar respecto de GTS (ncdc-gts)
    base = cf.get("ncdc-gts", 0.0)
    L.append("    (referencia interna del modelo: " + ref + "); respecto de GTS: " + ", ".join(f"{k} {v - base:+.1f}" for k, v in cf.items() if k != "ncdc-gts")
             + (f", {ref} {-base:+.1f}" if ref != "ncdc-gts" else ""))
    L.append("    Solapamiento directo (paso 19): USAF − GTS = +2.7 pp [IC 90 % +0.4,+5.1]; el modelo implica USAF − GTS = "
             f"{(cf.get('usaf-ds3', 0) - base):+.1f} pp (ver arriba).")

    # (b) eras
    L.append("\n(b) Tendencias del perfil completo (pp/década) por era homogénea [IC 95 %, bootstrap de temporadas, 500 réplicas]")
    eras = (("USAF, temporadas 1980-2006", (D.temp <= 2006) & (D.fuente == "usaf-ds3")),
            ("GTS, temporadas 2001-2025", (D.temp >= 2001) & (D.fuente == "ncdc-gts")),
            ("GTS, temporadas 2006-2025", (D.temp >= 2006) & (D.fuente == "ncdc-gts")))
    for nom, m in eras:
        for grupo, sel in (("10 estaciones", slice(None)), ("Argentina", D.arg == 1), ("Brasil", D.arg == 0)):
            q = D[m & (sel if not isinstance(sel, slice) else True)]
            for y in ("crudo", "ajustado"):
                f = lambda d, y=y: tend(d, y, [])  # noqa: E731
                lo95, lo90, hi90, hi95 = boot(q, f, 300)
                L.append(f"    {nom:28s} {grupo:13s} {y:9s} {f(q):+.2f} [{lo95:+.2f},{hi95:+.2f}]  (n {len(q)})")


    L.append("\n(b2) GTS, temporadas 2001-2025, por país sin Santa Rosa (crudo / ajustado) [IC 95 %, 300 réplicas]")
    m = (D.temp >= 2001) & (D.fuente == "ncdc-gts") & (D.sid != "ARM00087623")
    for grupo, sel in (("Argentina sin Santa Rosa", D.arg == 1), ("Brasil", D.arg == 0)):
        q = D[m & sel]
        for y in ("crudo", "ajustado"):
            f = lambda d, y=y: tend(d, y, [])  # noqa: E731
            lo95, lo90, hi90, hi95 = boot(q, f, 300)
            L.append(f"    {grupo:26s} {y:9s} {f(q):+.2f} [{lo95:+.2f},{hi95:+.2f}]  (n {len(q)})")

    # (c) DiD
    L.append("\n(c) Escalón argentino: cambio entre las 8 temporadas antes y las 8 después de cada fecha (sin solapamiento), media de estaciones;"
             " DiD = Argentina − Brasil, con IC 95 % por bootstrap de temporadas")
    L.append("    Regla de inclusión única: estaciones con ≥ 4 temporadas en cada ventana de 8.")
    for fecha, c in (("fin de 1997", 1998), ("feb 2000", 2000), ("nov 2002", 2003)):
        def did(d, c=c, y="crudo"):
            a = d[(d.temp >= c - 8) & (d.temp <= c - 1)]
            p = d[(d.temp >= c + 1) & (d.temp <= c + 8)]
            out = []
            for g in (1.0, 0.0):
                ya, yp = a[a.arg == g].groupby("sid")[y].mean(), p[p.arg == g].groupby("sid")[y].mean()
                na, np_ = a[a.arg == g].groupby("sid").temp.nunique(), p[p.arg == g].groupby("sid").temp.nunique()
                ok = ya.index.intersection(yp.index)
                ok = [i for i in ok if na[i] >= 4 and np_[i] >= 4]          # una regla de inclusión: ≥ 4 temporadas en cada ventana
                out.append(100 * (yp[ok] - ya[ok]).mean())
            return out[0] - out[1]
        lo95, lo90, hi90, hi95 = boot(D, did, 300)
        a_ = D[(D.temp >= c - 8) & (D.temp <= c - 1)]
        p_ = D[(D.temp >= c + 1) & (D.temp <= c + 8)]
        det = []
        for g, nm in ((1.0, "Argentina"), (0.0, "Brasil")):
            ya, yp = a_[a_.arg == g].groupby("sid").crudo.mean(), p_[p_.arg == g].groupby("sid").crudo.mean()
            na, np_ = a_[a_.arg == g].groupby("sid").temp.nunique(), p_[p_.arg == g].groupby("sid").temp.nunique()
            ok = [i for i in ya.index.intersection(yp.index) if na[i] >= 4 and np_[i] >= 4]
            det.append(f"{nm} {100 * (yp[ok] - ya[ok]).mean():+.1f} (n {len(ok)})")
        L.append(f"    {fecha:12s} {', '.join(det)}; DiD {did(D):+.1f} pp [{lo95:+.1f},{hi95:+.1f}]")
    L.append("    Modelo con escalón argentino (jet ~ mes + estación + tendencia + Argentina × 1[temporada ≥ t0]); coeficiente del escalón (pp) y de la tendencia (pp/déc):")
    for t0 in (1998, 2000, 2003):
        for extra, lab in (([], "sin fuente"), (["fuente"], "con fuente")):
            paso = pd.Series(((D.temp >= t0) & (D.arg == 1)).astype(float).values, index=D.index)

            def f(d, extra=extra, paso=paso):
                b, x = coefs(d, "crudo", extra, paso)
                return [100 * b[len(np.unique(x.mes))], 100 * b[-1]]
            est = f(D)
            lo95, lo90, hi90, hi95 = boot(D, f, 200)
            L.append(f"    t0 = {t0} {lab:10s} tendencia {est[0]:+.2f} [{lo95[0]:+.2f},{hi95[0]:+.2f}]; escalón argentino {est[1]:+.1f} [{lo95[1]:+.1f},{hi95[1]:+.1f}]")


    # (c2) DiD con efectos de fuente, e inferencia por permutación de la asignación de estaciones
    L.append("\n(c2) DiD con efectos de mes, estación y fuente (ventana de 8 temporadas a cada lado; coeficiente Argentina × posterior, pp) [IC 95 %, 500 réplicas]"
             " y p por permutación (todas las asignaciones de 4 estaciones entre las 9 con datos suficientes, sin ajustar por fuente)")
    import itertools
    for fecha, c in (("fin de 1997", 1998), ("feb 2000", 2000), ("nov 2002", 2003)):
        W = D[(D.temp >= c - 8) & (D.temp <= c + 8) & (D.temp != c)].copy()
        W["post"] = (W.temp > c).astype(float)
        W = W.dropna(subset=["crudo"]).reset_index(drop=True)

        def did_f(d, W_=None):
            x = d
            X = np.hstack([pd.get_dummies(x.mes).values.astype(float), pd.get_dummies(x.sid, drop_first=True).values.astype(float),
                           pd.get_dummies(x.fuente, drop_first=True).values.astype(float), x.post.values[:, None], (x.post * x.arg).values[:, None]])
            return 100 * np.linalg.lstsq(X, x.crudo.values.astype(float), rcond=None)[0][-1]
        est = did_f(W)
        lo95, lo90, hi90, hi95 = boot(W, did_f, 500)
        # permutación: estadístico simple (media de cambios) para cada asignación de 4 estaciones tratadas
        cambios = {}
        for sid, g in W.groupby("sid"):
            if g[g.post == 0].temp.nunique() >= 4 and g[g.post == 1].temp.nunique() >= 4:
                cambios[sid] = 100 * (g[g.post == 1].crudo.mean() - g[g.post == 0].crudo.mean())
        ids = list(cambios)
        obs = np.mean([cambios[i] for i in ids if i in ARG]) - np.mean([cambios[i] for i in ids if i not in ARG])
        nul = []
        for tr in itertools.combinations(ids, sum(i in ARG for i in ids)):
            nul.append(np.mean([cambios[i] for i in tr]) - np.mean([cambios[i] for i in ids if i not in tr]))
        pperm = float(np.mean(np.abs(nul) >= abs(obs) - 1e-9))
        sinC = [i for i in ids if i in ARG and i != "ARM00087344"]
        obs2 = np.mean([cambios[i] for i in sinC]) - np.mean([cambios[i] for i in ids if i not in ARG])
        L.append(f"    {fecha:12s} DiD con fuente {est:+.1f} [{lo95:+.1f},{hi95:+.1f}]; sin fuente (media de estaciones) {obs:+.1f}, p por permutación {pperm:.3f} "
                 f"({len(nul)} asignaciones); sin Córdoba (sin quiebre en 925 hPa) {obs2:+.1f}")

    # (d) por país
    L.append("\n(d) Tendencia agrupada del perfil completo (pp/década) por país, 1980-2025 [IC 95 %]")
    for nom, q in (("Argentina (4 est.)", D[D.arg == 1]), ("Brasil (6 est.)", D[D.arg == 0])):
        for y in ("crudo", "ajustado"):
            for extra, lab in (([], "sin fuente"), (["fuente"], "con fuente")):
                f = lambda d, y=y, extra=extra: tend(d, y, extra)  # noqa: E731
                lo95, lo90, hi90, hi95 = boot(q, f, 300)
                L.append(f"    {nom:19s} {y:9s} {lab:10s} {f(q):+.2f} [{lo95:+.2f},{hi95:+.2f}]")
    open("analisis/jet/34_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
