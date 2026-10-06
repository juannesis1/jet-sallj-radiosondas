"""Paper 2, paso 18: ¿cuánta lluvia traen los jets profundos y los someros?

El jet alimenta los sistemas convectivos de su región de salida, que dan buena parte de la lluvia de verano del noreste
argentino y el sur de Brasil (Salio et al. 2007; Saulo et al. 2007). Si los jets someros que revelan los sondeos
modernos fueran equivalentes a los profundos, su aumento aparente implicaría más transporte de humedad y más lluvia.
Día de jet regional (12 UTC, oct-mar 1998-2025): clase A si algún sondeo de Resistencia, Córdoba, Santa Rosa o Foz da
jet profundo; B si ninguno da A y alguno da somero; C si ninguno da jet y reportaron al menos dos.
Lluvia: CHIRPS v2.0 diario 0.25° (Funk et al. 2015), suma del día del jet y el siguiente (la convección nocturna de la
región de salida cae después del sondeo de la mañana), en la caja de salida 33-25°S, 62-53°W.
Resultados: lluvia media por evento con IC 90 % (bootstrap de días), fracción de la lluvia de la temporada que cae en
los dos días siguientes a cada clase, y mapas compuestos de anomalía (para la figura 6).
Revisión (ronda 2): anomalías respecto de la climatología mensual, bootstrap por temporadas, sensibilidad hasta 2015 y
estratificación por paso frontal.
Salida: analisis/jet/18_resumen.txt y 18_mapas.nc
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

EST = ["ARM00087155", "ARM00087344", "ARM00087623", "BRM00083827"]
CAJA = dict(lat=slice(-33, -25), lon=slice(-62, -53))
rng = np.random.default_rng(13)


def clase_regional():
    cols = {}
    for sid in EST:
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[(s.index.hour == 12) & (s.index.year >= 1997)].copy()
        s.index = s.index.normalize()
        s = s[~s.index.duplicated()]
        w8, w7 = np.hypot(s.u850, s.v850), np.hypot(s.u700, s.v700)
        dr = (np.degrees(np.arctan2(-s.u850, -s.v850)) + 360) % 360
        f = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).where(w8.notna() & w7.notna())
        c = pd.Series(np.where(f == 1, 2, np.where(s.jet_norte == True, 1, 0)), index=s.index, dtype=float)  # noqa: E712
        cols[sid] = c.where(f.notna() & s.jet_norte.notna())
    d = pd.DataFrame(cols)
    r = pd.Series(np.where(d.max(axis=1) == 2, "A", np.where(d.max(axis=1) == 1, "B", "C")), index=d.index)
    return r.where((d.notna().sum(axis=1) >= 2) | (d.max(axis=1) >= 1))


def chirps():
    ds = xr.open_mfdataset(sorted(glob.glob("data/chirps_diario/chirps_*.nc")), decode_times=False, combine="by_coords")
    p = ds.prcp.rename({"X": "lon", "Y": "lat", "T": "time"})
    # IRI codifica el tiempo como día juliano (units = "julian_day"); día juliano 2440587.5 = 1970-01-01 00 UTC
    t = pd.to_datetime(p.time.values.astype(float) - 2440587.5, unit="D").normalize()
    p = p.assign_coords(time=t).sortby("lat").load()
    return p


def main():
    import importlib
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    p16 = importlib.import_module("16_evolucion_sinoptica")
    c = clase_regional()
    p = chirps()
    p2 = p + p.shift(time=-1)  # día del jet + siguiente
    p2 = p2.sel(time=p2.time.dt.month.isin([10, 11, 12, 1, 2, 3]))
    caja = p2.sel(**CAJA).mean(["lat", "lon"]).to_series().dropna()
    clim = caja.groupby(caja.index.month).transform("mean")           # climatología mensual de la lluvia de 48 h
    anom = caja - clim
    c = c.reindex(caja.index).dropna()
    caja, anom = caja.reindex(c.index), anom.reindex(c.index)
    L = [f"Lluvia CHIRPS (mm en 48 h) en la región de salida 33-25°S, 62-53°W después de cada clase de jet regional,"
         f" oct-mar, temporadas {c.index.min().year + (c.index.min().month >= 10)}-{c.index.max().year + (c.index.max().month >= 10)}",
         "IC 90 % por bootstrap de temporadas completas; anomalías respecto de la media del mes calendario (todos los días)"]
    for k in "ABC":
        z, a = caja[c == k], anom[c == k]
        m, lo, hi = p16.ic(z)
        ma, loa, hia = p16.ic(a)
        L.append(f"  {k}: n {len(z):5d}  media {m:5.1f} mm [{lo:.1f},{hi:.1f}]  anomalía {ma:+.1f} [{loa:+.1f},{hia:+.1f}]"
                 f"  P(≥20 mm) {100 * (z >= 20).mean():4.1f} %")
    for a_, b_ in (("A", "C"), ("B", "C"), ("A", "B")):
        m, lo, hi = p16.ic(anom[c == a_], anom[c == b_])
        L.append(f"  anomalía {a_} − {b_}: {m:+.1f} mm [{lo:+.1f},{hi:+.1f}]")
    # sensibilidad: temporadas hasta 2015 (antes del salto de jets someros de Santa Rosa en 2016)
    cc = c[c.index < "2015-10-01"]
    for a_, b_ in (("A", "C"), ("B", "C"), ("A", "B")):
        m, lo, hi = p16.ic(anom.reindex(cc.index)[cc == a_], anom.reindex(cc.index)[cc == b_])
        L.append(f"  hasta 2015, anomalía {a_} − {b_}: {m:+.1f} mm [{lo:+.1f},{hi:+.1f}]")
    # estratificación por paso frontal en superficie (Resistencia o Córdoba, 48 h siguientes)
    fr = []
    for usaf in ("87155099999", "87344099999"):
        x = p16.superficie(usaf)
        dp, dT = x.p.diff(), x["T"].diff()
        f = ((dp >= 4) & (dT <= -3)).astype(float).where(dp.notna() & dT.notna())
        fr.append(pd.concat([f.shift(-1), f.shift(-2)], axis=1).max(axis=1))
    fr = pd.concat(fr, axis=1).max(axis=1).reindex(c.index)
    L.append("\n  Estratificado por paso frontal en las 48 h siguientes (Resistencia o Córdoba); anomalía media (mm) [n]:")
    for k in "ABC":
        t = []
        for v, lab in ((1.0, "con frente"), (0.0, "sin frente")):
            q = anom[(c == k) & (fr == v)]
            t.append(f"{lab} {q.mean():+.1f} [{len(q)}]")
        L.append(f"    {k}: " + "; ".join(t))
    # fracción de días y de la lluvia de 48 h por clase (los totales de 48 h se solapan entre días consecutivos, así que se informa la
    # fracción de la SUMA de esas lluvias y no una fracción de la lluvia estacional)
    n = c.value_counts()
    L.append("\n  Días regionales evaluados: " + ", ".join(f"{k} {n[k]} ({100 * n[k] / len(c):.0f} %)" for k in "ABC"))
    L.append("  Fracción de la suma de las lluvias de 48 h que corresponde a cada clase: " +
             ", ".join(f"{k} {100 * caja[c == k].sum() / caja.sum():.0f} %" for k in "ABC"))
    # regresión de la anomalía de lluvia sobre la clase y covariables sinópticas del día del jet (presión, tendencia de presión, frente)
    pres = []
    for usaf in ("87155099999", "87344099999", "87623099999", "87576099999"):
        x = p16.superficie(usaf)
        pres.append(x.p - x.groupby(x.index.month).p.transform("median"))
    pa = pd.concat(pres, axis=1).mean(axis=1)
    dp24 = pd.concat([x.p.diff() for x in [p16.superficie(u) for u in ("87155099999", "87344099999", "87623099999", "87576099999")]], axis=1).mean(axis=1)
    X = pd.DataFrame({"A": (c == "A").astype(float), "B": (c == "B").astype(float), "pa": pa.reindex(c.index), "dp24": dp24.reindex(c.index),
                      "frente": fr.reindex(c.index)}).dropna()
    yv = anom.reindex(X.index)
    tt = np.where(X.index.month >= 10, X.index.year + 1, X.index.year)

    def ols(ix):
        M = np.column_stack([np.ones(len(ix)), X.values[ix]])
        return np.linalg.lstsq(M, yv.values[ix], rcond=None)[0]

    todos = np.arange(len(X))
    b0 = ols(todos)
    gr = {t: np.where(tt == t)[0] for t in np.unique(tt)}
    rr = np.random.default_rng(18)
    bs = np.array([ols(np.concatenate([gr[t] for t in rr.choice(list(gr), len(gr))])) for _ in range(300)])
    nombres = ["const", "A (jet profundo)", "B (jet somero)", "anomalía de presión (hPa)", "Δp 24 h (hPa)", "frente en 48 h"]
    L.append("\n  Regresión de la anomalía de lluvia (mm) sobre la clase y covariables sinópticas del día del jet (IC 90 % por bootstrap de temporadas):")
    for i_, nm in enumerate(nombres):
        if i_:
            L.append(f"    {nm:28s} {b0[i_]:+.2f} [{np.percentile(bs[:, i_], 5):+.2f},{np.percentile(bs[:, i_], 95):+.2f}]")
    # clases por altura del núcleo (independientes de los niveles fijos)
    cols = {}
    for sid in EST:
        d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        d = d[(d.index.hour == 12) & (d.index >= "1997-10-01")].copy()
        d.index = d.index.normalize()
        d = d[~d.index.duplicated()]
        v = pd.Series(np.where(d.jet_norte == True, np.where(d.nucleo_dp <= 60, 1.0, 2.0), 0.0), index=d.index)  # noqa: E712
        cols[sid] = v.where(d.jet_norte.notna())
    dd = pd.DataFrame(cols)
    ch = pd.Series(np.where(dd.max(axis=1) == 2, "H", np.where(dd.max(axis=1) == 1, "L", "C")), index=dd.index).where(dd.notna().sum(axis=1) >= 2)
    ch = ch.reindex(caja.index).dropna()
    an2 = anom.reindex(ch.index)
    L.append("\n  Clases por altura del núcleo del jet (≤ 60 hPa sobre el suelo: L; alguna estación con núcleo > 60 hPa: H): anomalía de lluvia (mm) [IC 90 %]")
    for k in "LHC":
        m_, lo_, hi_ = p16.ic(an2[ch == k])
        L.append(f"    {k}: n {int((ch == k).sum()):5d}  {m_:+.1f} [{lo_:+.1f},{hi_:+.1f}]")
    for a_, b_ in (("L", "C"), ("H", "C"), ("H", "L")):
        m_, lo_, hi_ = p16.ic(an2[ch == a_], an2[ch == b_])
        L.append(f"    {a_} − {b_}: {m_:+.1f} [{lo_:+.1f},{hi_:+.1f}]")
    m = {k: p2.sel(time=c.index[c == k].intersection(pd.DatetimeIndex(p2.time.values))).mean("time") for k in "ABC"}
    xr.Dataset({f"lluvia_{k}": m[k] for k in "ABC"}).to_netcdf("analisis/jet/18_mapas.nc")
    open("analisis/jet/18_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
