"""Paper 2, paso 25: robustez estadística (respuesta a la revisión de la ronda 2, comentario M6).

(a) Multiplicidad: tendencias por estación del paso 3/Fig. 1 (22 estaciones, 2 criterios) con control de la tasa de falso
    descubrimiento (Benjamini-Hochberg, q = 0.10; Wilks 2016, con α_FDR = 2 α_global como recomienda).
(b) Colinealidad entre la tendencia y los efectos de fuente: R² de la regresión del año sobre efectos de mes, estación y fuente
    (VIF) y número de temporadas con ≥ 20 sondeos de dos fuentes a la vez.
(c) Modos climáticos: correlación del índice de jet fijo (media de anomalías de 7 estaciones, series sin tendencia lineal) con ONI, PDO
    y SAM en oct-dic (el análisis previo) y en la temporada completa oct-mar (y DJF para el ONI), con corrección FDR sobre las
    8 pruebas.
(d) Modelo logístico agrupado (en lugar de pendientes de Theil-Sen sobre frecuencias de eventos raros): razón de chances por década
    de jet fijo con efectos de mes, estación y fuente; IC 90 % por bootstrap de temporadas; 1980-2025 y 1993-2025.
Salida: analisis/jet/25_resumen.txt
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
p20 = importlib.import_module("20_habilidad_temporal")
rng = np.random.default_rng(25)
MODOS7 = ["ARM00087155", "ARM00087344", "ARM00087623", "ARM00087576", "BRM00083827", "BRM00083928", "BRM00083612"]


def bh(p, q=0.10):
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    ok = p[o] <= q * (np.arange(1, m + 1) / m)
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    sig = np.zeros(m, bool)
    sig[o[:k]] = True
    return sig


def tabla_mensual(archivo, **kw):
    t = pd.read_csv(archivo, **kw)
    t.columns = ["anio"] + list(range(1, 13))
    return t.set_index("anio").replace(99.99, np.nan)


def media_oct_mar(t):
    """Media oct-mar de la temporada a+1 (oct-dic de a, ene-mar de a+1)."""
    a = t[[10, 11, 12]].mean(axis=1)
    a.index = a.index + 1
    b = t[[1, 2, 3]].mean(axis=1)
    return pd.concat([a, b], axis=1).mean(axis=1)


def main():
    L = []
    # ---- (a) FDR sobre las tendencias por estación
    f = pd.read_csv("analisis/jet/03b_tendencias.csv").dropna(subset=["ta", "tb"])
    L.append(f"(a) Tendencias por estación (Fig. 1, {len(f)} estaciones): significativas con p < 0.05 sin corregir y con FDR (q = 0.10)")
    for c, lab in (("pa", "perfil completo"), ("pb", "niveles fijos")):
        sig = bh(f[c].values)
        L.append(f"  {lab:16s} sin corregir {(f[c] < 0.05).sum():2d}; con FDR {sig.sum():2d} (p máx. incluido {f[c][sig].max() if sig.any() else float('nan'):.3f})")
    # ---- (b) colinealidad
    D = pd.read_parquet("analisis/jet/24_perfil.parquet")
    X = np.hstack([pd.get_dummies(D.mes).values.astype(float), pd.get_dummies(D.sid).values.astype(float)[:, 1:],
                   pd.get_dummies(D.fuente).values.astype(float)[:, 1:]])
    y = D.temp.values.astype(float)
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - res.var() / y.var()
    L.append(f"\n(b) Colinealidad año ~ mes + estación + fuente (10 estaciones, {len(D)} sondeos): R² = {r2:.3f}, VIF = {1 / (1 - r2):.1f}")
    L.append("    Temporadas por estación con ≥ 20 sondeos de dos fuentes a la vez (superposición):")
    for sid, g in D.groupby("sid"):
        c = g.groupby(["temp", "fuente"]).size().unstack(fill_value=0) >= 20
        L.append(f"      {sid}: {int((c.sum(axis=1) >= 2).sum())} temporadas superpuestas de {g.temp.nunique()}; fuentes {dict(g.fuente.value_counts().head(4))}")
    # ---- (c) modos climáticos
    F = pd.read_parquet("analisis/jet/22_fijos.parquet")
    F = F[F.sid.isin(MODOS7)]
    PF = pd.read_parquet("analisis/jet/24_perfil.parquet")
    PF = PF[PF.sid.isin(MODOS7)]
    pdo = tabla_mensual("data/indices/pdo_ersst.dat", sep=r"\s+", skiprows=1)
    sam = pd.read_csv("data/indices/sam_marshall.txt", sep=r"\s+", skiprows=2, header=None, names=["anio"] + list(range(1, 13))).set_index("anio")
    oni = pd.read_csv("data/indices/oni.txt", sep=r"\s+")
    onid = oni[oni.SEAS == "DJF"].set_index("YR").ANOM
    onio = oni[oni.SEAS == "OND"].set_index("YR").ANOM
    onio.index = onio.index + 1
    modos = {"oct-mar": {"ONI (NDJ-JFM medio)": pd.concat([onid, onio], axis=1).mean(axis=1), "PDO": media_oct_mar(pdo), "SAM": media_oct_mar(sam)},
             "oct-dic": {"ONI": onio, "PDO": pdo[[10, 11, 12]].mean(axis=1).set_axis(pdo.index + 1), "SAM": sam[[10, 11, 12]].mean(axis=1).set_axis(sam.index + 1)}}
    pruebas = []
    L.append("\n(c) Índices de jet (media de anomalías de 7 estaciones; series sin tendencia lineal) vs modos climáticos; p con tamaño efectivo de muestra"
             " N_eff = N (1 − r1x r1y) / (1 + r1x r1y) (r1 = autocorrelación de orden 1)")
    for indice, base, col in (("niveles fijos", F, "fijos"), ("perfil completo", PF, "crudo")):
        for temporada, mod in (("oct-dic", base[base.mes.isin([10, 11, 12])]), ("oct-mar", base)):
            g = mod.groupby(["sid", "temp"])[col].agg(["mean", "count"])
            g = g[g["count"] >= (30 if temporada == "oct-dic" else 60)]["mean"] * 100
            idx = (g - g.groupby("sid").transform("mean")).groupby("temp").mean().loc[1980:2025]
            for nom, m in modos[temporada].items():
                a = pd.concat([idx, m], axis=1).dropna().loc[1980:2025]
                x_, y_ = signal.detrend(a.iloc[:, 0].values), signal.detrend(a.iloc[:, 1].values)
                r, p = stats.pearsonr(x_, y_)
                r1x, r1y = np.corrcoef(x_[:-1], x_[1:])[0, 1], np.corrcoef(y_[:-1], y_[1:])[0, 1]
                neff = min(len(a), len(a) * (1 - r1x * r1y) / (1 + r1x * r1y))   # nunca mayor que N
                t_ = r * np.sqrt((neff - 2) / (1 - r ** 2))
                pe = 2 * stats.t.sf(abs(t_), neff - 2)
                pruebas.append((indice, temporada, nom, r, p, pe, len(a), neff))
    sig = bh([x[5] for x in pruebas if x[0] == "niveles fijos"])
    k = 0
    for (ind, t, n, r, p, pe, kk, neff) in pruebas:
        marca = ""
        if ind == "niveles fijos":
            marca = "  significativa con FDR (6 pruebas, p efectivo)" if sig[k] else "  no significativa con FDR"
            k += 1
        L.append(f"  {ind:15s} {t:8s} {n:22s} r = {r:+.2f} (p {p:.3f}; N {kk}, N_eff {neff:.0f}, p efectivo {pe:.3f}){marca}")
    # ---- (d) modelo logístico agrupado
    L.append("\n(d) Modelo logístico agrupado de jet con niveles fijos (efectos de mes, estación y fuente): razón de chances por década [IC 90 %]")
    F = F.dropna(subset=["fijos"])
    F0 = pd.read_parquet("analisis/jet/22_fijos.parquet").dropna(subset=["fijos"])

    def odds(d, y):
        X = np.column_stack([np.ones(len(d)), ((d.temp.values - 2000) / 10.0)] + [pd.get_dummies(d.mes, drop_first=True).values.astype(float)]
                            + [pd.get_dummies(d.sid, drop_first=True).values.astype(float), pd.get_dummies(d.fuente, drop_first=True).values.astype(float)])
        return np.exp(p20.logit(X, d[y].values.astype(float))[1])

    for ini in (1980, 1993):
        for y, lab in (("fijos", "crudo"), ("fijos_adj", "ajustado con ERA5")):
            d = F0[F0.temp >= ini].dropna(subset=[y])
            temps = d.temp.unique()
            g = {t: d[d.temp == t] for t in temps}
            bs = [odds(pd.concat([g[t] for t in rng.choice(temps, len(temps))]), y) for _ in range(100)]
            L.append(f"  desde {ini} {lab:18s} OR/déc = {odds(d, y):.2f} [{np.percentile(bs, 5):.2f},{np.percentile(bs, 95):.2f}]")
    open("analisis/jet/25_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
