"""Paper 2, paso 7: ¿qué jets "aparecen" con la alta resolución de los sondeos modernos?

En Resistencia, Córdoba y Santa Rosa (12 UTC, oct-mar, 1998-2025, sondeos con 925 hPa) se separan los días en:
  A) jet homogéneo: detectado con los niveles fijos 850/700 (Bonner), medible igual en todo el período;
  B) jet "solo alta resolución": detectado con el perfil completo pero no con 850/700;
  C) sin jet.
Para cada clase: núcleo (velocidad y altura), y compuestos de ERA5 a 12 UTC (archivos del paper 1): viento a 850 hPa
y transporte meridional de humedad a 850 hPa (−q·v) en la región, como anomalía respecto de los días sin jet.
Salidas: analisis/jet/07_resumen.txt, analisis/jet/07_compuestos.nc
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import era5io  # noqa: E402

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa"}
# caja de salida del jet de cada estación (lat S, lon): Santa Rosa (36.6°S) queda fuera de la caja de las otras dos
CAJAS = {"ARM00087155": ((-20, -32), (-65, -55)), "ARM00087344": ((-20, -32), (-65, -55)), "ARM00087623": ((-30, -40), (-68, -58))}


def clases(sid):
    d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
    d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3]) & (np.where(d.index.month >= 10, d.index.year + 1, d.index.year) >= 1998) & d.u925.notna()]
    w8, w7 = np.hypot(d.u850, d.v850), np.hypot(d.u700, d.v700)
    dr = (np.degrees(np.arctan2(-d.u850, -d.v850)) + 360) % 360
    fijo = (w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))
    full = d.jet_norte == True
    c = pd.Series("C", index=d.index)
    c[full & ~fijo] = "B"
    c[fijo] = "A"
    return d.assign(clase=c.values)


def main():
    pl = era5io.abrir_pl()
    pl = pl.sel(time=pl.time.dt.hour == 12).sel(level=850)
    L, comp = [], {}
    for sid, nom in EST.items():
        d = clases(sid)
        n = d.clase.value_counts()
        L.append(f"\n{nom}: A (fijos 850/700) {n.get('A', 0)}, B (solo alta resolución) {n.get('B', 0)}, C {n.get('C', 0)} días")
        for k in ("A", "B"):
            x = d[d.clase == k]
            L.append(f"  {k}: núcleo {x.nucleo_v.median():.1f} m/s a {x.nucleo_dp.median():.0f} hPa sobre la superficie; "
                     f"v850 sondeo {(-x.v850).median():.1f} m/s (sur→norte positivo invertido); q850 {x.q850.median():.1f} g/kg")
        fechas = {k: pd.DatetimeIndex(d.index[d.clase == k]) for k in "ABC"}
        sub = {k: pl.sel(time=pl.time.isin(f.values)) for k, f in fechas.items()}
        flujo = {k: (-(s.q * 1000) * s.v).mean("time") for k, s in sub.items()}
        for k in ("A", "B"):
            an = (flujo[k] - flujo["C"]).compute()
            (la0, la1), (lo0, lo1) = CAJAS[sid]
            caja = an.sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1)).mean().item()
            L.append(f"  {k}: anomalía del transporte −qv a 850 hPa ({-la0}-{-la1}°S, {-lo0}-{-lo1}°W) {caja:+.1f} g kg⁻¹ m s⁻¹")
            comp[f"{sid}_{k}"] = an
    xr.Dataset(comp).to_netcdf("analisis/jet/07_compuestos.nc")
    open("analisis/jet/07_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
