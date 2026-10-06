"""Paper 2, paso 16: evolución sinóptica en superficie antes y después de cada clase de jet (compuestos con desfasaje).

Hipótesis (Salio et al. 2002; Seluchi et al. 2003; Saulo et al. 2007): los jets profundos (clase A) son prefrontales,
con presión en descenso por la profundización de la baja del noroeste argentino y la llegada de un frente frío
después; los someros (clase B) responderían a un forzante más débil. Con las observaciones horarias de superficie de
cada aeropuerto (NOAA ISD: presión reducida al nivel del mar (sólo SLP, sin reglaje altimétrico), temperatura y punto de rocío a
12 UTC ± 1 h), compuestos de anomalías (respecto de la mediana del mes calendario) de los días −2 a +3 respecto del
sondeo de 12 UTC de cada clase (oct-mar, 1998-2025).
Paso frontal en las 48 h siguientes: subida de presión ≥ 4 hPa y descenso de temperatura ≥ 3 °C en 24 h (12 UTC a
12 UTC). Diferencias entre clases con IC 90 % por bootstrap de temporadas completas.
Salida: analisis/jet/16_resumen.txt y 16_compuestos.parquet (para la figura).
"""
import glob
import os

import numpy as np
import pandas as pd

EST = {"ARM00087155": ("87155099999", "Resistencia"), "ARM00087344": ("87344099999", "Córdoba"),
       "ARM00087623": ("87623099999", "Santa Rosa"), "ARM00087576": ("87576099999", "Ezeiza")}
rng = np.random.default_rng(7)


def superficie(usaf):
    partes = []
    for f in sorted(glob.glob(f"data/isd_full/{usaf}_*.csv")):
        if int(f[-8:-4]) < 1997:
            continue
        d = pd.read_csv(f, usecols=lambda c: c in ("DATE", "TMP", "DEW", "SLP", "MA1"), dtype=str)
        t = d.TMP.str.split(",", expand=True)
        td = d.DEW.str.split(",", expand=True)
        slp = pd.to_numeric(d.SLP.str.split(",", expand=True)[0], errors="coerce") if "SLP" in d else np.nan
        qnh = pd.to_numeric(d.MA1.str.split(",", expand=True)[0], errors="coerce") if "MA1" in d else np.nan
        x = pd.DataFrame({"T": pd.to_numeric(t[0], errors="coerce").values / 10,
                          "Td": pd.to_numeric(td[0], errors="coerce").values / 10,
                          "p": pd.Series(slp).where(slp < 99999).values / 10}, index=pd.to_datetime(d.DATE).values)
        partes.append(x)
    x = pd.concat(partes)
    x = x[(x["T"].abs() < 60) & (x.Td.abs() < 60)].copy()
    x.loc[(x.p < 950) | (x.p > 1060), "p"] = np.nan
    x = x[(x.index.hour >= 11) & (x.index.hour <= 13)]
    x["dist"] = np.abs(x.index.hour * 60 + x.index.minute - 720)
    x = x.sort_values("dist")
    x.index = x.index.normalize()
    x = x.groupby(level=0).first()[["T", "Td", "p"]].sort_index()
    return x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D"))


def clases(sid):
    s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
    s = s[(s.index.hour == 12) & (np.where(s.index.month >= 10, s.index.year + 1, s.index.year) >= 1998) & s.index.month.isin([10, 11, 12, 1, 2, 3])].copy()
    s.index = s.index.normalize()
    s = s[~s.index.duplicated()]
    w8, w7 = np.hypot(s.u850, s.v850), np.hypot(s.u700, s.v700)
    dr = (np.degrees(np.arctan2(-s.u850, -s.v850)) + 360) % 360
    f = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).where(w8.notna() & w7.notna())
    c = pd.Series(np.where(f == 1, "A", np.where(s.jet_norte == True, "B", "C")), index=s.index)  # noqa: E712
    return c.where(f.notna() & s.jet_norte.notna()).dropna()


def temporada(idx):
    return np.where(idx.month >= 10, idx.year + 1, idx.year)


def ic(a, b=None, n=1000):
    """Media de a (o diferencia de medias a − b) con IC 90 % por bootstrap de TEMPORADAS completas (los jets y los frentes se
    agrupan en días consecutivos y en temporadas; remuestrear días daría intervalos demasiado estrechos). a, b: Series con fecha."""
    ta = pd.DataFrame({"s": a.groupby(temporada(a.index)).sum(), "n": a.groupby(temporada(a.index)).count()})
    tb = None if b is None else pd.DataFrame({"s": b.groupby(temporada(b.index)).sum(), "n": b.groupby(temporada(b.index)).count()})
    temps = ta.index if tb is None else ta.index.union(tb.index)
    ta = ta.reindex(temps, fill_value=0)
    tb = None if tb is None else tb.reindex(temps, fill_value=0)

    def est(i):
        m = ta.s.values[i].sum() / max(ta.n.values[i].sum(), 1)
        return m if tb is None else m - tb.s.values[i].sum() / max(tb.n.values[i].sum(), 1)

    todo = np.arange(len(temps))
    x = np.array([est(rng.integers(0, len(temps), len(temps))) for _ in range(n)])
    return est(todo), np.percentile(x, 5), np.percentile(x, 95)


def main():
    L = ["Evolución en superficie (12 UTC) alrededor de cada clase de jet, oct-mar 1998-2025; anomalías respecto de la",
         "mediana del mes. Días −2…+3. Frente = Δp24 ≥ +4 hPa y ΔT24 ≤ −3 °C en las 48 h siguientes."]
    comp, largo = [], []
    for sid, (usaf, nom) in EST.items():
        x = superficie(usaf)
        for v in ("T", "Td", "p"):
            x[v + "a"] = x[v] - x.groupby(x.index.month)[v].transform("median")
        dp, dT = x.p.diff(), x["T"].diff()
        x["frente"] = ((dp >= 4) & (dT <= -3)).astype(float).where(dp.notna() & dT.notna())
        c = clases(sid)
        c = c[c.index.isin(x.index)]
        L.append(f"\n=== {nom}  (A {int((c == 'A').sum())}, B {int((c == 'B').sum())}, C {int((c == 'C').sum())})")
        for k in ("A", "B", "C"):
            dias = c.index[c == k]
            for lag in range(-2, 4):
                y = x.reindex(dias + pd.Timedelta(days=lag))
                for v in ("pa", "Ta", "Tda"):
                    largo.append(pd.DataFrame({"clase": k, "lag": lag, "var": v, "temp": temporada(dias), "val": y[v].values}).dropna())
                comp.append({"sid": sid, "nombre": nom, "clase": k, "lag": lag, "n": int(y.pa.notna().sum()),
                             "pa": y.pa.mean(), "Ta": y.Ta.mean(), "Tda": y.Tda.mean()})
        fr = {}
        for k in "ABC":
            ix = c.index[c == k]
            v = pd.concat([x.frente.reindex(ix + pd.Timedelta(days=d)).set_axis(ix) for d in (1, 2)], axis=1, sort=True).max(axis=1)
            fr[k] = v.dropna()
        tend = {}
        for k in "ABC":
            ix = c.index[c == k]
            tend[k] = (x.p.reindex(ix).values - x.p.reindex(ix - pd.Timedelta(days=1)).values)
            tend[k] = pd.Series(tend[k], index=ix).dropna()
        for k in "ABC":
            m, lo, hi = ic(100 * fr[k])
            mt, lt, ht = ic(tend[k])
            L.append(f"  {k}: frente en 48 h {m:4.1f} % [{lo:4.1f},{hi:4.1f}]   Δp 24 h previas {mt:+.1f} hPa [{lt:+.1f},{ht:+.1f}]")
        for a, b in (("A", "C"), ("B", "C"), ("A", "B")):
            m, lo, hi = ic(100 * fr[a], 100 * fr[b])
            mt, lt, ht = ic(tend[a], tend[b])
            L.append(f"  {a}−{b}: frente {m:+5.1f} pp [{lo:+.1f},{hi:+.1f}]   Δp previa {mt:+.1f} [{lt:+.1f},{ht:+.1f}]")
    C = pd.DataFrame(comp)
    C.to_parquet("analisis/jet/16_compuestos.parquet")
    # IC 90 % de los compuestos agrupados (días de las 4 estaciones) por bootstrap de temporadas completas
    LG = pd.concat(largo)
    ics = []
    for (k, lag, v), g in LG.groupby(["clase", "lag", "var"]):
        t = g.groupby("temp").val.agg(["sum", "count"])
        ids = [rng.integers(0, len(t), len(t)) for _ in range(500)]
        bs = [t["sum"].values[i].sum() / t["count"].values[i].sum() for i in ids]
        ics.append({"clase": k, "lag": lag, "var": v, "media": g.val.mean(), "lo": np.percentile(bs, 5), "hi": np.percentile(bs, 95)})
    pd.DataFrame(ics).to_parquet("analisis/jet/16_ic.parquet")
    L.append("\nCompuestos (media de las 4 estaciones) de anomalía de presión (hPa) / temperatura (°C) / rocío (°C):")
    t = C.groupby(["clase", "lag"])[["pa", "Ta", "Tda"]].mean().round(1)
    L.append(t.unstack("clase").to_string())
    open("analisis/jet/16_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
