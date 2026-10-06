"""Paper 2, paso 23: robustez del análisis de habilidad de ERA5 (respuesta a la revisión de la ronda 2).

(a) Forma del cambio de la habilidad: POD de ERA5 (niveles fijos) con modelo logístico lineal en el año, con escalón en 2000 y
    con escalón + pendiente (AIC), con efectos de estación, controlando la intensidad del jet observado; POD por estación.
(b) Métricas simétricas por período (9 estaciones): POD, razón de falsas alarmas (FAR), POFD, sesgo de frecuencia y ETS.
(c) Hipótesis de asimilación: RMS de la diferencia sondeo − ERA5 (rapidez a 850 hPa) por período, en todos los días y en los días
    con jet observado. Si ERA5 se ajusta cada vez mejor a estos sondeos, el RMS cae.
(d) Sesgo de condicionamiento: noches (00→12 UTC, estaciones brasileñas, 1998-2025) condicionadas en el jet de ERA5 (no en el
    del sondeo): aceleración nocturna del sondeo vs ERA5; y cuantiles de la aceleración y del viento a 12 UTC sin condicionar.
    Regresión a la media predice que el sondeo sea MENOS extremo que ERA5 cuando se condiciona en ERA5; un sesgo del modelo
    predice que sea MÁS extremo en ambos condicionamientos.
(e) Selección: frecuencia de jet fijo en ERA5 en días con y sin sondeo de 12 UTC, por período.
(f) Imparcialidad en la vertical: ERA5 se bajó sólo hasta 700 hPa; se repite la detección por perfil de los sondeos truncando
    también el perfil observado en 700 hPa.
Salida: analisis/jet/23_resumen.txt
"""
import glob
import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
p19 = importlib.import_module("19_emulacion_resolucion")
p20 = importlib.import_module("20_habilidad_temporal")
NOMBRES = p20.NOMBRES
BR = ["BRM00083827", "BRM00083612", "BRM00083362", "BRM00083928", "BRM00083768", "BRM00083208", "BRM00082824"]
rng = np.random.default_rng(23)


def loglik(X, y):
    b = p20.logit(X, y)
    p = np.clip(expit(X @ b), 1e-9, 1 - 1e-9)
    return b, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metricas(o, m):
    a, b, c, d = (o & m).sum(), (~o & m).sum(), (o & ~m).sum(), (~o & ~m).sum()
    n = a + b + c + d
    aref = (a + b) * (a + c) / n
    return {"POD": a / max(a + c, 1), "FAR": b / max(a + b, 1), "POFD": b / max(b + d, 1),
            "sesgo": (a + b) / max(a + c, 1), "ETS": (a - aref) / max(a + b + c - aref, 1e-9)}


def main():
    r = pd.read_parquet("analisis/jet/12_diario.parquet")
    for c in ("obs_fijos", "obs_perfil", "era_fijos", "era_perfil"):
        r[c] = r[c].astype(float)
    r = r[(r.temp >= 1980) & (r.temp <= 2025)]
    L = []
    # ERA5 y sondeos: rapidez a 850 hPa y a 12 UTC, para intensidad y RMS
    e = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob("data/era5_perfiles/*.parquet")))
    e["fecha"] = pd.to_datetime(e.fecha)
    e = e[e.nivel.isin([925.0, 850.0, 700.0]) & e.fecha.dt.month.isin([10, 11, 12, 1, 2, 3])]
    e["w"] = np.hypot(e.u, e.v)
    e["h"], e["dia"] = e.fecha.dt.hour, e.fecha.dt.normalize()
    e12 = e[e.h == 12].pivot_table(index=["sid", "dia"], columns="nivel", values="w")
    obs = []
    for sid in NOMBRES:
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[s.index.hour == 12]
        s.index = s.index.normalize()
        s = s[~s.index.duplicated()]
        obs.append(pd.DataFrame({"sid": sid, "fecha": s.index, "w850": np.hypot(s.u850, s.v850).values,
                                 "w700": np.hypot(s.u700, s.v700).values}))
    obs = pd.concat(obs)
    r = r.merge(obs, on=["sid", "fecha"], how="left")
    r["caida"] = r.w850 - r.w700
    r = r.join(e12[850.0].rename("e850"), on=["sid", "fecha"]) if False else r.merge(
        e12[850.0].rename("e850").reset_index().rename(columns={"dia": "fecha"}), on=["sid", "fecha"], how="left")
    r["per"] = pd.cut(r.temp, [1979, 1995, 2010, 2025], labels=["1980-95", "1996-2010", "2011-25"])

    # ---- (a) forma del cambio de habilidad
    x = r.dropna(subset=["obs_fijos", "w850", "caida"])
    hits = x[x.obs_fijos == 1].copy()
    hits = hits[hits.groupby("sid").sid.transform("size") >= 40]
    y = hits.era_fijos.values
    est = pd.get_dummies(hits.sid, drop_first=True).values.astype(float)
    ctl = np.column_stack([((hits[c] - hits[c].mean()) / hits[c].std()).values for c in ("w850", "caida")])
    t10 = ((hits.temp.values - 2000) / 10.0)[:, None]
    esc = (hits.temp.values >= 2001).astype(float)[:, None]
    uno = np.ones((len(hits), 1))
    mod = {"lineal": np.hstack([uno, t10, ctl, est]), "escalón 2001": np.hstack([uno, esc, ctl, est]),
           "escalón + pendiente": np.hstack([uno, esc, t10, ctl, est]), "sin cambio": np.hstack([uno, ctl, est])}
    L.append(f"(a) Forma del cambio de la POD de ERA5 (jets con niveles fijos, {len(hits)} jets, {hits.sid.nunique()} estaciones)")
    for nom, X in mod.items():
        b, ll = loglik(X, y)
        L.append(f"  modelo {nom:20s} AIC {2 * X.shape[1] - 2 * ll:8.1f}")
    # fecha del escalón estimada por máxima verosimilitud (modelo con escalón en el año y, efectos de estación y de intensidad)
    ajustes = []
    for yr in range(1990, 2013):
        Xs = np.hstack([uno, (hits.temp.values > yr).astype(float)[:, None], ctl, est])
        ajustes.append((yr, 2 * Xs.shape[1] - 2 * loglik(Xs, y)[1]))
    mejor = min(ajustes, key=lambda a_: a_[1])
    aic_lin = 2 * mod["lineal"].shape[1] - 2 * loglik(mod["lineal"], y)[1]
    L.append(f"  fecha del escalón estimada: {mejor[0]} (AIC {mejor[1]:.1f}; recta {aic_lin:.1f}); AIC por año de escalón: "
             + ", ".join(f"{a_}: {b_:.0f}" for a_, b_ in ajustes if a_ in (1992, 1995, 1998, 2000, 2001, 2003, 2005, 2008, 2010)))
    X = mod["escalón + pendiente"]
    b, _ = loglik(X, y)
    L.append(f"  escalón + pendiente: escalón {b[1]:+.2f} log-odds, pendiente después {b[2]:+.2f} por década")
    pre, post = hits[hits.temp <= 2000], hits[hits.temp >= 2001]
    L.append(f"  POD 1980-2000 {pre.era_fijos.mean():.2f} (n {len(pre)}), 2001-2025 {post.era_fijos.mean():.2f} (n {len(post)})")
    L.append("  POD por estación y período (n):")
    for sid, q in hits.groupby("sid"):
        t = q.groupby("per", observed=True).era_fijos.agg(["mean", "size"])
        L.append(f"    {NOMBRES[sid]:13s} " + "  ".join(f"{p}: {m:.2f} ({n})" for p, (m, n) in t.iterrows()))

    # ---- (b) métricas simétricas
    L.append("\n(b) Métricas de contingencia de ERA5 contra el sondeo, niveles fijos (estaciones con ≥ 40 jets observados)")
    ok = hits.sid.unique()
    tabla = []
    for p, q in x[x.sid.isin(ok)].groupby("per", observed=True):
        m = metricas(q.obs_fijos.values == 1, q.era_fijos.values == 1)
        tabla.append({"periodo": str(p), **m})
        L.append(f"  {p:10s} " + "  ".join(f"{k} {v:.2f}" for k, v in m.items()) + f"  (n {len(q)})")

    # ---- (c) asimilación: RMS de la diferencia a 850 hPa
    L.append("\n(c) RMS de (sondeo − ERA5) en la rapidez a 850 hPa (m/s), 12 UTC, por período")
    z = r.dropna(subset=["w850", "e850"]).copy()
    z["d"] = z.w850 - z.e850
    for nom, q in (("todos los días", z), ("días con jet observado", z[z.obs_fijos == 1]), ("días con viento ≥ 10 m/s", z[z.w850 >= 10])):
        t = q.groupby("per", observed=True).d.agg(lambda v: np.sqrt(np.mean(v ** 2)))
        for pp in t.index:
            for tt in tabla:
                if tt["periodo"] == str(pp):
                    tt["RMS " + nom] = t[pp]
                    if nom == "todos los días":
                        q_ = q[q.per == pp]
                        gs = q_.groupby("temp").d.agg(lambda v: np.sum(v ** 2)), q_.groupby("temp").d.size()
                        rb = np.random.default_rng(23)
                        bs = []
                        for _ in range(500):
                            ii = rb.integers(0, len(gs[0]), len(gs[0]))
                            bs.append(np.sqrt(gs[0].values[ii].sum() / gs[1].values[ii].sum()))
                        tt["RMS_lo"], tt["RMS_hi"] = np.percentile(bs, 5), np.percentile(bs, 95)
        n = q.groupby("per", observed=True).size()
        L.append(f"  {nom:26s} " + "  ".join(f"{p}: {t[p]:.2f} (n {n[p]})" for p in t.index))

    # composición fija: estaciones con ≥ 20 jets observados en cada uno de los tres períodos
    cuenta = hits.groupby(["sid", "per"], observed=True).size().unstack(fill_value=0)
    fijas = list(cuenta.index[(cuenta >= 20).all(axis=1)])
    L.append("  Con composición fija de estaciones (" + ", ".join(NOMBRES[i] for i in fijas) + "):")
    for p, q in x[x.sid.isin(fijas)].groupby("per", observed=True):
        m = metricas(q.obs_fijos.values == 1, q.era_fijos.values == 1)
        L.append(f"    {p:10s} " + "  ".join(f"{k} {v:.2f}" for k, v in m.items()) + f"  (n {len(q)})")
    pd.DataFrame(tabla).to_csv("analisis/jet/23_metricas.csv", index=False)
    # ---- (e) selección de días con sondeo
    L.append("\n(e) Frecuencia de jet fijo en ERA5 (%) en días CON sondeo de 12 UTC y SIN sondeo, por período, dentro de temporadas-estación con ≥ 60 sondeos")
    q = r[r.sid.isin(ok)].copy()
    q["con"] = q.w850.notna()
    q = q[q.groupby(["sid", "temp"]).con.transform("sum") >= 60]   # sólo temporadas-estación con registro (≥ 60 sondeos)
    for p, g in q.groupby("per", observed=True):
        L.append(f"  {p:10s} con sondeo {100 * g[g.con].era_fijos.mean():.2f} (n {g.con.sum()}) | sin sondeo "
                 f"{100 * g[~g.con].era_fijos.mean():.2f} (n {(~g.con).sum()})")

    # ---- (g) tendencias de ERA5 y de los sondeos en los mismos días (agrupado de las estaciones con registro suficiente)
    from estadistica import mk_hamed_rao, sen
    L.append("\n(g) Tendencia (pp/década, Sen) de la frecuencia de jet fijo en los MISMOS días con sondeo: sondeo | ERA5 (agrupado de anomalías)")
    qq = r.dropna(subset=["obs_fijos"])
    qq = qq[qq.groupby(["sid", "temp"]).obs_fijos.transform("size") >= 60]
    g = qq.groupby(["sid", "temp"]).agg(obs=("obs_fijos", "mean"), era=("era_fijos", "mean")).reset_index()
    ests = [i for i in g.sid.unique() if (g.sid == i).sum() >= 35]
    g = g[g.sid.isin(ests)]
    L.append("  estaciones: " + ", ".join(NOMBRES[i] for i in ests))
    for ini in (1980, 1993, 2000):
        z = g[g.temp >= ini]
        fila = []
        for c in ("obs", "era"):
            an = 100 * (z[c] - z.groupby("sid")[c].transform("mean"))
            v = an.groupby(z.temp).mean()
            bs = [10 * sen(v.index.values[i].astype(float), v.values[i]) for i in (rng.integers(0, len(v), len(v)) for _ in range(500))]
            fila.append(f"{10 * sen(v.index.values.astype(float), v.values):+.2f} [{np.percentile(bs, 5):+.2f},{np.percentile(bs, 95):+.2f}]")
        L.append(f"  desde {ini}: sondeo {fila[0]} | ERA5 {fila[1]}")

    # ---- (d) condicionamiento simétrico
    P = pd.read_parquet("analisis/jet/15_pares.parquet")
    P.index.name = "dia"
    P = P.reset_index().set_index(["sid", "dia"])
    ee = e[e.nivel.isin([925.0, 850.0])].pivot_table(index=["sid", "dia"], columns=["h", "nivel"], values="w")
    ee.columns = [f"e{h}_{int(n)}" for h, n in ee.columns]
    P = P.join(ee, how="inner")
    P["e_dw925"], P["e_dw850"] = P.e12_925 - P.e0_925, P.e12_850 - P.e0_850
    J = r[["sid", "fecha", "era_perfil", "era_fijos"]].rename(columns={"fecha": "dia"}).set_index(["sid", "dia"])
    J = J[~J.index.duplicated()]
    P = P.join(J, how="left")
    P = P[P.index.get_level_values("sid").isin(BR)]
    L.append("\n(d) Noches 00→12 UTC (Brasil, 1998-2025): Δw (m/s) a 925 hPa, mediana, según el jet de ERA5 (perfil) a 12 UTC")
    for lab, q in (("ERA5 con jet", P[P.era_perfil == 1]), ("ERA5 sin jet", P[P.era_perfil == 0]),
                   ("sondeo con jet (A o B)", P[P.clase_12.isin(["A", "B"])]), ("sondeo sin jet (C)", P[P.clase_12 == "C"])):
        b = [np.median(rng.choice(q.dw925.dropna().values, len(q.dw925.dropna()))) for _ in range(300)]
        be = [np.median(rng.choice(q.e_dw925.dropna().values, len(q.e_dw925.dropna()))) for _ in range(300)]
        L.append(f"  {lab:24s} n {len(q):5d}  sondeo {q.dw925.median():+.1f} [{np.percentile(b, 5):+.1f},{np.percentile(b, 95):+.1f}]"
                 f"  ERA5 {q.e_dw925.median():+.1f} [{np.percentile(be, 5):+.1f},{np.percentile(be, 95):+.1f}]")
    L.append("  Cuantiles sin condicionar (todas las noches pareadas): Δw925 / viento a 12 UTC a 925 hPa")
    for qq in (50, 75, 90, 95, 99):
        L.append(f"    p{qq}: Δw925 sondeo {np.nanpercentile(P.dw925, qq):+.1f}  ERA5 {np.nanpercentile(P.e_dw925, qq):+.1f}"
                 f" | w925(12 UTC) sondeo {np.nanpercentile(np.hypot(P.u925_12, P.v925_12), qq):.1f}  ERA5 {np.nanpercentile(P.e12_925, qq):.1f}")

    # ---- (f) truncar el perfil observado en 700 hPa
    L.append("\n(f) Detección por perfil con el perfil del sondeo truncado en 700 hPa (como ERA5): POD / FAR de ERA5 por período")
    filas = []
    for sid in ("ARM00087155", "ARM00087344", "ARM00087576", "ARM00087623", "BRM00083827", "BRM00083612"):
        perf = p19.leer(sid)
        for k, (ps, a, fu) in perf.items():
            filas.append({"sid": sid, "fecha": k, "obs_trunc": p19.jet(ps, a[a[:, 0] >= 700.0]), "obs_full": p19.jet(ps, a)})
    T = pd.DataFrame(filas).merge(r[["sid", "fecha", "era_perfil", "temp"]], on=["sid", "fecha"]).dropna()
    T["per"] = pd.cut(T.temp, [1979, 1995, 2010, 2025], labels=["1980-95", "1996-2010", "2011-25"])
    for p, g in T.groupby("per", observed=True):
        m1 = metricas(g.obs_full.values == 1, g.era_perfil.values == 1)
        m2 = metricas(g.obs_trunc.values == 1, g.era_perfil.values == 1)
        L.append(f"  {p:10s} perfil completo POD {m1['POD']:.2f} FAR {m1['FAR']:.2f} | truncado en 700 hPa POD {m2['POD']:.2f} FAR {m2['FAR']:.2f}"
                 f" | frecuencia obs {100 * g.obs_full.mean():.1f} → {100 * g.obs_trunc.mean():.1f} %")
    open("analisis/jet/23_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
