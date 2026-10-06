"""Funciones comunes de los pasos del paper 2 (serie estacional de frecuencia y tendencia), sin dependencias gráficas."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from estadistica import mk_hamed_rao, sen  # noqa: E402


def temporada(idx):
    return np.where(idx.month >= 10, idx.year + 1, idx.year)


def series(sid):
    d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
    d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3]) & (d.index.year >= 1979)]
    out = {}
    if {"u850", "v850", "u700", "v700"} <= set(d):
        w8, w7 = np.hypot(d.u850, d.v850), np.hypot(d.u700, d.v700)
        dr = (np.degrees(np.arctan2(-d.u850, -d.v850)) + 360) % 360
        out["fijos"] = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).where(w8.notna() & w7.notna())
    out["completo"] = d.jet_norte.astype(float)
    out["estandar"] = d.jet_norte_std.astype(float)
    t = temporada(d.index)
    res = {}
    for k, x in out.items():
        g = x.astype(float).groupby(t)
        res[k] = (g.mean()[g.count() >= 60] * 100).loc[1980:2025]
    return pd.DataFrame(res)


def tendencia(s):
    s = s.dropna()
    if len(s) < 20:
        return np.nan, np.nan
    x = s.index.values.astype(float)
    return sen(x, s.values) * 10, mk_hamed_rao(x, s.values)[1]
