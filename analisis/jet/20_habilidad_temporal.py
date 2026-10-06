"""Paper 2, paso 20: ¿cambia con el tiempo la habilidad de ERA5 para ver el jet, y qué tendencia muestra ERA5?

(a) Habilidad. Sobre los días en que el sondeo detecta un jet con niveles fijos (850/700 hPa, criterio homogéneo del lado
    observado), regresión logística de la detección por ERA5 (mismo criterio, mismo día) contra el año, con efectos de
    estación y controlando la intensidad del jet observado (rapidez a 850 hPa y caída 850-700): si el coeficiente del año
    es positivo, ERA5 detecta mejor jets de la misma intensidad en los años recientes. Lo mismo para las falsas alarmas
    (días sin jet observado). IC 90 % por bootstrap de temporadas.
(b) Tendencias. Frecuencia de jet en ERA5 en el punto de cada estación (todas las fechas oct-mar 12 UTC, no sólo las
    que tienen sondeo), con niveles fijos y con perfil completo (resolución constante): Sen + Mann-Kendall Hamed-Rao,
    1980-2025, y agrupado (media de anomalías). Se compara con la tendencia observada con niveles fijos.
(c) Herencia de saltos. Frecuencia de jet de perfil en ERA5 8 temporadas antes y después de cada salto de los sondeos,
    junto a la del sondeo.
(d) Atribución. En los días con sondeo, la frecuencia de ERA5 = aciertos + falsas alarmas = POD·f_obs + falsas alarmas.
    Contrafáctico con la POD fija en su valor de 1980-1995 en cada estación: si la tendencia de ERA5 desaparece, se debe
    al cambio de habilidad y no a más jets observados.
(e) Período homogéneo de alta resolución (2006-2025): tendencia agrupada (6 estaciones, media de anomalías) de jets
    profundos (A) y someros (B) en los sondeos, y el salto de jets someros de Santa Rosa en 2016 comparado con ERA5.
(f) Detección por ERA5 (perfil completo) de los jets profundos (A) y someros (B) observados, 1998-2025.
Salida: analisis/jet/20_resumen.txt, 20_series.parquet, 20_atribucion.parquet y 20_aciertos.parquet (figura 7).
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
from estadistica import mk_hamed_rao, sen  # noqa: E402

p12 = importlib.import_module("12_era5_vs_sondeos")
NOMBRES = p12.NOMBRES
_q = pd.read_csv("analisis/jet/22_quiebres.csv")
_q = _q[(_q.nivel == 925) & (~_q.regional) & _q.BH_q10]
SALTOS = {r.sid: r.fecha for r in _q.itertuples() if r.sid in ("ARM00087155", "ARM00087576", "ARM00087623", "BRM00083827")}  # quiebres de 925 hPa del paso 22
rng = np.random.default_rng(20)


def logit(X, y, it=50):
    """Regresión logística por Newton-Raphson (con un poco de regularización ridge para estabilidad)."""
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = expit(X @ b)
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        paso = np.linalg.solve(H, X.T @ (y - p))
        b += paso
        if np.abs(paso).max() < 1e-8:
            break
    return b


def diseño(x, controles):
    cols = [np.ones(len(x)), (x.temp.values - 2000) / 10.0]
    cols += [((x[c] - x[c].mean()) / x[c].std()).values for c in controles]
    X = np.column_stack(cols)
    est = pd.get_dummies(x.sid, drop_first=True).values.astype(float)
    return np.hstack([X, est])


def efecto_año(x, y, controles, n=300):
    b = logit(diseño(x, controles), x[y].values.astype(float))[1]
    temps = x.temp.unique()
    g = {t: x[x.temp == t] for t in temps}
    bs = []
    for _ in range(n):
        z = pd.concat([g[t] for t in rng.choice(temps, len(temps))])
        bs.append(logit(diseño(z, controles), z[y].values.astype(float))[1])
    return b, np.percentile(bs, [5, 95])


def main():
    r = pd.read_parquet("analisis/jet/12_diario.parquet")
    for c in ("obs_fijos", "obs_perfil", "era_fijos", "era_perfil"):
        r[c] = r[c].astype(float)
    L = []
    # ---- (a) habilidad: intensidad observada a 850/700
    obs = []
    for sid in NOMBRES:
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[s.index.hour == 12].copy()
        s.index = s.index.normalize()
        s = s[~s.index.duplicated()]
        w8, w7 = np.hypot(s.u850, s.v850), np.hypot(s.u700, s.v700)
        obs.append(pd.DataFrame({"sid": sid, "fecha": s.index, "w850": w8.values, "caida": (w8 - w7).values}))
    obs = pd.concat(obs)
    r = r.merge(obs, on=["sid", "fecha"], how="left")
    r = r[(r.temp >= 1980) & (r.temp <= 2025)]
    x = r.dropna(subset=["obs_fijos", "w850", "caida"]).copy()
    x["era_fijos"] = x.era_fijos.astype(float)
    hits = x[x.obs_fijos == 1]
    hits = hits[hits.groupby("sid").sid.transform("size") >= 40]
    b, (lo, hi) = efecto_año(hits, "era_fijos", ["w850", "caida"])
    b0, (lo0, hi0) = efecto_año(hits, "era_fijos", [])
    L.append("(a) Probabilidad de que ERA5 detecte un jet observado (niveles fijos), regresión logística con efectos de"
             f" estación; n {len(hits)} jets, {hits.sid.nunique()} estaciones")
    L.append(f"  coef. del año (por década, log-odds): {b0:+.2f} [{lo0:+.2f},{hi0:+.2f}] sin controles; "
             f"{b:+.2f} [{lo:+.2f},{hi:+.2f}] controlando rapidez a 850 y caída 850-700")
    for per, z in hits.groupby(pd.cut(hits.temp, [1979, 1995, 2010, 2025]), observed=True):
        L.append(f"  POD {per}: {z.era_fijos.mean():.2f} (n {len(z)}); rapidez observada media {z.w850.mean():.1f} m/s")
    noj = x[x.obs_fijos == 0]
    noj = noj[noj.groupby("sid").sid.transform("size") >= 40]
    bf, (lof, hif) = efecto_año(noj, "era_fijos", ["w850", "caida"], n=100)
    L.append(f"  falsas alarmas (ERA5 jet sin jet observado): coef. del año {bf:+.2f} [{lof:+.2f},{hif:+.2f}] por década")
    # ---- (b) tendencias de ERA5 en todas las fechas
    e = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob("data/era5_perfiles/*.parquet")))
    e["fecha"] = pd.to_datetime(e.fecha)
    e = e[(e.fecha.dt.hour == 12) & e.fecha.dt.month.isin([10, 11, 12, 1, 2, 3])]
    filas = []
    for sid in NOMBRES:
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        p_sup = s[s.index.hour == 12].p_sup.median()
        for fecha, g in e[e.sid == sid].groupby("fecha"):
            jf, jp = p12.detectar(g, p_sup)
            filas.append({"sid": sid, "fecha": fecha.normalize(), "era_fijos": jf, "era_perfil": jp})
    E = pd.DataFrame(filas)
    E["temp"] = np.where(E.fecha.dt.month >= 10, E.fecha.dt.year + 1, E.fecha.dt.year)
    E = E[(E.temp >= 1980) & (E.temp <= 2025)]
    S = (100 * E.groupby(["sid", "temp"])[["era_fijos", "era_perfil"]].mean()).reset_index()
    o = r.groupby(["sid", "temp"]).agg(obs_fijos=("obs_fijos", "mean"), obs_perfil=("obs_perfil", "mean"),
                                       n=("obs_fijos", "count"))
    o.loc[o.n < 60, ["obs_fijos", "obs_perfil"]] = np.nan
    S = S.merge((100 * o[["obs_fijos", "obs_perfil"]]).reset_index(), on=["sid", "temp"], how="left")
    S.to_parquet("analisis/jet/20_series.parquet")
    L.append("\n(b) Tendencia 1980-2025 (pp/década, Sen; p Mann-Kendall Hamed-Rao), frecuencia en el punto de grilla")
    for sid, nom in NOMBRES.items():
        z = S[S.sid == sid].set_index("temp")
        t = []
        for c in ("era_fijos", "era_perfil", "obs_fijos"):
            y = z[c].dropna()
            if len(y) >= 20:
                pend = 10 * sen(y.index.values.astype(float), y.values)
                t.append(f"{c} {pend:+.2f} (p {mk_hamed_rao(y.index.values.astype(float), y.values)[1]:.2f})" if abs(pend) >= 0.05 else
                         f"{c} ≈0 (pendiente degenerada: serie de eventos raros con mediana de diferencias nula)")
        L.append(f"  {nom:13s} " + "  ".join(t))
    for c in ("era_fijos", "era_perfil"):
        an = S[c] - S.groupby("sid")[c].transform("mean")
        y = an.groupby(S.temp).mean()
        bs = [10 * sen(y.index.values[i].astype(float), y.values[i]) for i in
              (rng.integers(0, len(y), len(y)) for _ in range(500))]
        L.append(f"  AGRUPADO {c}: {10 * sen(y.index.values.astype(float), y.values):+.2f} pp/década "
                 f"[{np.percentile(bs, 5):+.2f},{np.percentile(bs, 95):+.2f}] "
                 f"(p {mk_hamed_rao(y.index.values.astype(float), y.values)[1]:.3f})")
    # ---- (c) herencia de los saltos
    L.append("\n(c) Frecuencia de jet (perfil completo, %) 8 temporadas antes → después de cada salto: sondeo | ERA5")
    for sid, f in SALTOS.items():
        t0 = pd.Timestamp(f)
        c = t0.year + 1 if t0.month >= 10 else t0.year
        z = S[S.sid == sid].set_index("temp")
        a, d = z.loc[c - 8:c - 1], z.loc[c + 1:c + 8]
        L.append(f"  {NOMBRES[sid]:13s} {f}: sondeo {a.obs_perfil.mean():5.1f} → {d.obs_perfil.mean():5.1f} | "
                 f"ERA5 {a.era_perfil.mean():5.1f} → {d.era_perfil.mean():5.1f}")
    # control brasileño de (c): mismos tramos de 8 temporadas en estaciones sin salto en esas fechas
    L.append("  Control: cambio (pp) entre las 8 temporadas antes y después de cada fecha, media de estaciones, sondeo | ERA5 (perfil completo)")
    ARG = ["ARM00087155", "ARM00087576", "ARM00087623", "ARM00087344"]
    BRC = ["BRM00083612", "BRM00083362", "BRM00083208", "BRM00083928", "BRM00082824"]
    for fecha, c in (("fin de 1997", 1998), ("feb 2000", 2000), ("nov 2002", 2003)):
        fila = []
        for nombre, grupo in (("Argentina", ARG), ("Brasil", BRC)):
            do, de = [], []
            for sid in grupo:
                z = S[S.sid == sid].set_index("temp")
                a, d_ = z.loc[c - 8:c - 1], z.loc[c + 1:c + 8]
                if a.obs_perfil.notna().sum() >= 4 and d_.obs_perfil.notna().sum() >= 4:
                    do.append(d_.obs_perfil.mean() - a.obs_perfil.mean())
                    de.append(d_.era_perfil.mean() - a.era_perfil.mean())
            fila.append(f"{nombre} (n {len(do)}): {np.mean(do):+.1f} | {np.mean(de):+.1f}" if do else f"{nombre}: sin datos")
        L.append(f"    {fecha:12s} " + "   ".join(fila))
    # ---- (d) atribución
    L.append("\n(d) Tendencia de ERA5 (niveles fijos, días con sondeo) real vs con POD fija en la de 1980-1995 (pp/década)")
    z = r.dropna(subset=["obs_fijos"]).copy()
    z["hit"] = z.obs_fijos * z.era_fijos
    z["fa"] = (1 - z.obs_fijos) * z.era_fijos
    g = z.groupby(["sid", "temp"]).agg(obs=("obs_fijos", "mean"), hit=("hit", "mean"), fa=("fa", "mean"),
                                       era=("era_fijos", "mean"), n=("obs_fijos", "size")).reset_index()
    g = g[g.n >= 60]
    pod0 = z[z.temp <= 1995].groupby("sid").apply(lambda q: q.hit.sum() / max(q.obs_fijos.sum(), 1), include_groups=False)
    g = g[g.sid.isin(pod0.index[z[z.temp <= 1995].groupby("sid").obs_fijos.sum() >= 20])]
    g["cf"] = g.sid.map(pod0) * g.obs + g.fa
    g.to_parquet("analisis/jet/20_atribucion.parquet")
    hits[["sid", "temp", "era_fijos", "w850"]].to_parquet("analisis/jet/20_aciertos.parquet")
    L.append(f"  estaciones: {', '.join(NOMBRES[i] for i in g.sid.unique())}")
    # diferencia pareada (ERA5 − sondeo) de las anomalías agrupadas, IC por bootstrap de temporadas
    an_e = (100 * (g.era - g.groupby("sid").era.transform("mean"))).groupby(g.temp).mean()
    an_o = (100 * (g.obs - g.groupby("sid").obs.transform("mean"))).groupby(g.temp).mean()
    dif_ = (an_e - an_o).dropna()
    bsd = [10 * sen(dif_.index.values[i].astype(float), dif_.values[i]) for i in (rng.integers(0, len(dif_), len(dif_)) for _ in range(1000))]
    L.append(f"  Diferencia pareada ERA5 − sondeos (mismas temporadas y días): {10 * sen(dif_.index.values.astype(float), dif_.values):+.2f} pp/década"
             f" [IC 90 % {np.percentile(bsd, 5):+.2f},{np.percentile(bsd, 95):+.2f}; IC 95 % {np.percentile(bsd, 2.5):+.2f},{np.percentile(bsd, 97.5):+.2f}]")
    # lo mismo contra los sondeos con el viento ajustado por los quiebres propios (paso 22): ¿la diferencia con ERA5 persiste?
    F = pd.read_parquet("analisis/jet/22_fijos.parquet")
    F = F.rename_axis(None).assign(fecha=F.index.normalize())[["sid", "fecha", "fijos_adj"]].drop_duplicates(["sid", "fecha"])
    z2 = z.copy()
    z2["fecha"] = pd.to_datetime(z2.fecha).dt.normalize()
    z2 = z2.merge(F, on=["sid", "fecha"], how="inner").dropna(subset=["fijos_adj"])
    g2 = z2.groupby(["sid", "temp"]).agg(obs=("fijos_adj", "mean"), era=("era_fijos", "mean"), n=("fijos_adj", "size")).reset_index()
    g2 = g2[(g2.n >= 60) & g2.sid.isin(g.sid.unique())]
    e2 = (100 * (g2.era - g2.groupby("sid").era.transform("mean"))).groupby(g2.temp).mean()
    o2 = (100 * (g2.obs - g2.groupby("sid").obs.transform("mean"))).groupby(g2.temp).mean()
    d2 = (e2 - o2).dropna()
    bs2 = [10 * sen(d2.index.values[i].astype(float), d2.values[i]) for i in (rng.integers(0, len(d2), len(d2)) for _ in range(1000))]
    L.append(f"  Diferencia pareada ERA5 − sondeos AJUSTADOS por los quiebres propios (mismas temporadas y días): {10 * sen(d2.index.values.astype(float), d2.values):+.2f} pp/década"
             f" [IC 95 % {np.percentile(bs2, 2.5):+.2f},{np.percentile(bs2, 97.5):+.2f}]")
    for c, lab in (("era", "ERA5 real"), ("cf", "POD fija 1980-95"), ("obs", "observado")):
        an = 100 * (g[c] - g.groupby("sid")[c].transform("mean"))
        y = an.groupby(g.temp).mean()
        bs = [10 * sen(y.index.values[i].astype(float), y.values[i]) for i in
              (rng.integers(0, len(y), len(y)) for _ in range(500))]
        L.append(f"  {lab:17s}: {10 * sen(y.index.values.astype(float), y.values):+.2f} "
                 f"[{np.percentile(bs, 5):+.2f},{np.percentile(bs, 95):+.2f}] ({g.sid.nunique()} estaciones)")
    # contrafáctico equivalente para el criterio de perfil completo (misma contabilidad; sirve para ver si los jets someros pueden explicar el
    # aumento de ERA5 en el perfil)
    z2 = r.dropna(subset=["obs_perfil"]).copy()
    z2["hit"] = z2.obs_perfil * z2.era_perfil
    z2["fa"] = (1 - z2.obs_perfil) * z2.era_perfil
    g2 = z2.groupby(["sid", "temp"]).agg(obs=("obs_perfil", "mean"), hit=("hit", "mean"), fa=("fa", "mean"), era=("era_perfil", "mean"),
                                         n=("obs_perfil", "size")).reset_index()
    g2 = g2[g2.n >= 60]
    pod2 = z2[z2.temp <= 1995].groupby("sid").apply(lambda q: q.hit.sum() / max(q.obs_perfil.sum(), 1), include_groups=False)
    ok2 = z2[z2.temp <= 1995].groupby("sid").obs_perfil.sum()
    g2 = g2[g2.sid.isin(ok2.index[ok2 >= 20])]
    g2["cf"] = g2.sid.map(pod2) * g2.obs + g2.fa
    L.append("\n(d2) Mismo contrafáctico con el criterio de perfil completo (días con sondeo; estaciones: " + ", ".join(NOMBRES[i] for i in g2.sid.unique()) + ")")
    for c, lab in (("era", "ERA5 real"), ("cf", "POD fija 1980-95"), ("obs", "observado")):
        an = 100 * (g2[c] - g2.groupby("sid")[c].transform("mean"))
        y = an.groupby(g2.temp).mean()
        bs = [10 * sen(y.index.values[i].astype(float), y.values[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
        L.append(f"  {lab:17s}: {10 * sen(y.index.values.astype(float), y.values):+.2f} [{np.percentile(bs, 5):+.2f},{np.percentile(bs, 95):+.2f}]")
    # ---- (e) período homogéneo 2006-2025
    L.append("\n(e) Sondeos 2006-2025 (alta resolución): tendencia de jets profundos (A) y someros (B), pp/década")
    an = []
    for sid in ("ARM00087155", "ARM00087344", "ARM00087623", "ARM00087576", "BRM00083827", "BRM00083612"):
        d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3])].copy()
        d["temp"] = np.where(d.index.month >= 10, d.index.year + 1, d.index.year)
        d = d[(d.temp >= 2006) & (d.temp <= 2025)]
        w8, w7 = np.hypot(d.u850, d.v850), np.hypot(d.u700, d.v700)
        dr = (np.degrees(np.arctan2(-d.u850, -d.v850)) + 360) % 360
        f = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).astype(float).where(w8.notna() & w7.notna())
        jn = d.jet_norte.astype(float)
        d["A"], d["B"] = f, ((jn == 1) & (f == 0)).astype(float).where(f.notna() & jn.notna())
        g = d.groupby("temp")
        t = (100 * g[["A", "B"]].mean()).where(g.B.count() >= 60).dropna().astype(float)
        tt = [f"{c} {10 * sen(t.index.values.astype(float), t[c].values):+.2f} "
              f"(p {mk_hamed_rao(t.index.values.astype(float), t[c].values)[1]:.2f})" for c in "AB"]
        L.append(f"  {NOMBRES[sid]:13s} " + "  ".join(tt))
        an.append(t - t.mean())
    P = pd.concat(an).groupby(level=0).mean()
    for c in "AB":
        L.append(f"  AGRUPADO {c}: {10 * sen(P.index.values.astype(float), P[c].values):+.2f} "
                 f"(p {mk_hamed_rao(P.index.values.astype(float), P[c].values)[1]:.2f})")
    d = pd.read_parquet("data/igra_sa/perfiles/ARM00087623.parquet")
    d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3])].copy()
    d["temp"] = np.where(d.index.month >= 10, d.index.year + 1, d.index.year)
    w8, w7 = np.hypot(d.u850, d.v850), np.hypot(d.u700, d.v700)
    dr = (np.degrees(np.arctan2(-d.u850, -d.v850)) + 360) % 360
    f = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).astype(float)
    d["B"] = ((d.jet_norte.astype(float) == 1) & (f == 0)).astype(float)
    for per, q in (("2008-2015", d[(d.temp >= 2008) & (d.temp <= 2015)]), ("2016-2025", d[(d.temp >= 2016) & (d.temp <= 2025)])):
        L.append(f"  Santa Rosa {per}: jets someros {100 * q.B.mean():.1f} %, núcleo mediano {q.nucleo_dp[q.B == 1].median():.0f} hPa"
                 f" sobre la superficie, niveles en 0-250 hPa {q.n_niveles.median():.0f}")
    # viento a 925 hPa de Santa Rosa menos la media de Córdoba y Ezeiza, 2006-2025
    def m925(sid):
        q = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        q = q[(q.index.hour == 12) & (q.index.year >= 2006)]
        w = np.hypot(q.u925, q.v925)
        g = w.groupby(q.index.to_period("M"))
        return g.mean()[g.count() >= 10]
    dif = (m925("ARM00087623") - pd.concat([m925("ARM00087344"), m925("ARM00087576")], axis=1).mean(axis=1)).dropna()
    zz = (dif - dif.mean()) / dif.std()
    tt = np.array([k * zz[:k].mean() ** 2 + (len(zz) - k) * zz[k:].mean() ** 2 for k in range(12, len(zz) - 12)])
    k = int(np.argmax(tt)) + 12
    L.append(f"  Santa Rosa 925 hPa menos Córdoba-Ezeiza, 2006-2025: SNHT T={tt.max():.0f} en {dif.index[k]}, "
             f"{dif.values[:k].mean():+.2f} → {dif.values[k:].mean():+.2f} m/s")
    z = S[S.sid == "ARM00087623"].set_index("temp")
    a, d = z.loc[2008:2015], z.loc[2017:2025]
    L.append(f"  Santa Rosa, perfil completo 2008-2015 → 2017-2025: sondeo {a.obs_perfil.mean():.1f} → {d.obs_perfil.mean():.1f} %,"
             f" ERA5 {a.era_perfil.mean():.1f} → {d.era_perfil.mean():.1f} %")
    # ---- (f) detección de jets profundos y someros por ERA5
    L.append("\n(f) Fracción de jets observados que ERA5 detecta (criterio de perfil), 1998-2025: profundos A / someros B")
    q = r[(r.temp >= 1998)].dropna(subset=["obs_perfil", "obs_fijos"])
    q = q.assign(A=q.obs_fijos == 1, B=(q.obs_perfil == 1) & (q.obs_fijos == 0))
    for sid, z in q.groupby("sid"):
        if z.A.sum() >= 15:
            L.append(f"  {NOMBRES[sid]:13s} A {100 * z[z.A].era_perfil.mean():3.0f} % (n {z.A.sum()})  "
                     f"B {100 * z[z.B].era_perfil.mean():3.0f} % (n {z.B.sum()})")
    open("analisis/jet/20_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
