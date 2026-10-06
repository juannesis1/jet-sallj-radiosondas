"""Paper 2, paso 2: perfiles verticales de ERA5 (1000-700 hPa, 12 niveles) en las estaciones del recorrido del jet.

Para aplicar a ERA5 el mismo criterio de perfil completo que a los sondeos (paso 1) hacen falta muchos niveles.
Se pide un año calendario por vez, solo ene-mar y oct-dic (temporada cálida), 00 y 12 UTC, u y v en 12 niveles de
1000 a 700 hPa (costo ≈ 53 000 < 60 000 del CDS) sobre la caja que
contiene las estaciones, se extrae el punto de grilla más cercano a cada estación y se borra el archivo grande.
Salida: data/era5_perfiles/<bloque>.parquet (estación, fecha, nivel, u, v). Reanudable. Log: analisis/jet/02_log.txt
"""
import os
import sys
import time

import cdsapi
import pandas as pd
import xarray as xr

ESTACIONES = {"ARM00087155": (-27.44, -59.05), "ARM00087344": (-31.30, -64.21), "ARM00087576": (-34.82, -58.54),
              "ARM00087623": (-36.59, -64.28), "BRM00083827": (-25.60, -54.48), "BRM00083928": (-29.78, -57.03),
              "BRM00083612": (-20.47, -54.67), "BRM00083362": (-15.65, -56.10), "BRM00083208": (-12.70, -60.10),
              "BRM00082824": (-8.77, -63.92), "BRM00083768": (-23.33, -51.13)}
NIVELES = ["1000", "975", "950", "925", "900", "875", "850", "825", "800", "775", "750", "700"]
AREA = [-8, -65, -37, -51]
SAL = "data/era5_perfiles"


def bloque(anios):
    dest = f"{SAL}/{anios[0]}-{anios[-1]}.parquet"
    if os.path.exists(dest):
        return
    nc = f"{SAL}/tmp_{anios[0]}.nc"
    req = {"product_type": ["reanalysis"], "variable": ["u_component_of_wind", "v_component_of_wind"],
           "pressure_level": NIVELES, "year": [str(y) for y in anios], "month": ["01", "02", "03", "10", "11", "12"],
           "day": [f"{d:02d}" for d in range(1, 32)], "time": ["00:00", "12:00"], "area": AREA,
           "data_format": "netcdf", "download_format": "unarchived"}
    c = cdsapi.Client(quiet=True)
    while True:
        try:
            c.retrieve("reanalysis-era5-pressure-levels", req, nc)
            break
        except Exception as e:
            print(time.strftime("%d/%m %H:%M"), anios, "reintento:", str(e)[:120], flush=True)
            time.sleep(300)
    ds = xr.open_dataset(nc).rename({"valid_time": "fecha", "pressure_level": "nivel"})
    partes = []
    for sid, (la, lo) in ESTACIONES.items():
        p = ds[["u", "v"]].sel(latitude=la, longitude=lo, method="nearest").to_dataframe().reset_index()
        partes.append(p[["fecha", "nivel", "u", "v"]].assign(sid=sid))
    pd.concat(partes).to_parquet(dest)
    ds.close()
    os.remove(nc)
    print(time.strftime("%d/%m %H:%M"), "ok", dest, flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    os.makedirs(SAL, exist_ok=True)
    bloques = [[y] for y in range(1979, 2026)]
    if len(sys.argv) > 1:                       # repartir entre procesos: python 02_... i n
        i, n = int(sys.argv[1]), int(sys.argv[2])
        bloques = bloques[i::n]
    for b in bloques:
        bloque(b)
