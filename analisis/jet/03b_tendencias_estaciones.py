"""Paper 2, paso 3b: tendencia por estación (Theil-Sen, Mann-Kendall con la corrección de Hamed y Rao) del perfil completo y de los niveles fijos,
1980-2025, temporadas con ≥ 60 sondeos y estaciones con ≥ 20 temporadas. Alimenta la Fig. 1, el control de multiplicidad del paso 25 y la
Tabla 1 (paso 28). Salida: analisis/jet/03b_tendencias.csv
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from comun import series, tendencia  # noqa: E402


def main():
    est = pd.read_fwf("data/igra/igra2-station-list.txt", colspecs=[(0, 11), (12, 20), (21, 30)], header=None, names=["sid", "lat", "lon"]).set_index("sid")
    red = set(pd.read_csv("data/igra_sa/estaciones.txt", sep=r"\s+", header=None, usecols=[0])[0])
    filas = []
    for f in sorted(glob.glob("data/igra_sa/perfiles/*.parquet")):
        sid = os.path.basename(f)[:11]
        if sid not in red:
            continue
        s = series(sid)
        if "fijos" not in s:
            continue
        ta, pa = tendencia(s.completo)
        tb, pb = tendencia(s.fijos)
        filas.append({"sid": sid, "lat": est.lat[sid], "lon": est.lon[sid], "ta": ta, "pa": pa, "tb": tb, "pb": pb})
    r = pd.DataFrame(filas).dropna(subset=["ta", "tb"])
    r.to_csv("analisis/jet/03b_tendencias.csv", index=False)
    sa, sb = r[r.pa < 0.05], r[r.pb < 0.05]
    open("analisis/jet/fig1_resumen.txt", "w").write(
        f"Estaciones con tendencia: {len(r)}\nPerfil completo, p<0.05: {len(sa)} ({(sa.ta > 0).sum()} positivas)\n"
        f"Niveles fijos, p<0.05: {len(sb)}: " + ", ".join(f"{i} {t:+.2f}" for i, t in zip(sb.sid, sb.tb)) + "\n"
        f"Perfil completo, máx: " + ", ".join(f"{i} {t:+.2f}" for i, t in r.nlargest(4, "ta")[["sid", "ta"]].values) + "\n")
    print(open("analisis/jet/fig1_resumen.txt").read())


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
