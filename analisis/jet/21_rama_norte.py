"""Paper 2, paso 21: ¿se pueden verificar con sondeos las tendencias publicadas del jet fuera de la cuenca del Plata?

(a) Sitios de Montini et al. (2019): estaciones de IGRA a menos de 1.5° de Santa Cruz de la Sierra (17.8°S, 63.2°W) y
    Mariscal Estigarribia (22.0°S, 60.6°W), con su número de sondeos.
(b) Rama norte (Jones 2019): sondeos de 12 UTC por década en Maracay, San Antonio del Táchira, Las Gaviotas, Riohacha,
    Curaçao, Trinidad y Boa Vista; último año con datos; frecuencia de jet con niveles fijos antes y después de 1990 y
    tendencia 1980-2025 (Sen, Mann-Kendall Hamed-Rao, temporadas oct-mar con ≥ 60 sondeos). En la rama norte el jet es
    del norte o del este según el tramo, así que se usa la rapidez sin restricción de dirección.
Salida: analisis/jet/21_resumen.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from estadistica import mk_hamed_rao, sen  # noqa: E402

NORTE = {"VEM00080413": "Maracay", "VEM00080447": "San Antonio del Táchira", "COM00080241": "Las Gaviotas",
         "COM00080035": "Riohacha", "UCM00078988": "Curaçao", "TDM00078970": "Trinidad", "BRM00082022": "Boa Vista"}
SITIOS = {"Santa Cruz de la Sierra": (-17.8, -63.2), "Mariscal Estigarribia": (-22.0, -60.6)}


def main():
    L = []
    lst = pd.read_fwf("data/igra/igra2-station-list.txt", colspecs=[(0, 11), (12, 20), (21, 30), (41, 71), (72, 76),
                                                                     (77, 81), (82, 88)], header=None,
                      names=["sid", "lat", "lon", "nombre", "ini", "fin", "n"])
    L.append("(a) Estaciones IGRA a menos de 1.5° de los sitios de Montini et al. (2019)")
    for nom, (la, lo) in SITIOS.items():
        c = lst[(abs(lst.lat - la) < 1.5) & (abs(lst.lon - lo) < 1.5)]
        L.append(f"  {nom}: " + ("; ".join(f"{r.nombre.strip()} {r.ini}-{r.fin} n={r.n}" for r in c.itertuples())
                                   or "ninguna"))
    L.append("\n(b) Rama norte: sondeos de 12 UTC por década, último año, jet con niveles fijos (sin restricción de dirección)")
    for sid, nom in NORTE.items():
        d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        d = d[d.index.hour == 12]
        dec = d.groupby(d.index.year // 10 * 10).size()
        w8, w7 = np.hypot(d.u850, d.v850), np.hypot(d.u700, d.v700)
        f = ((w8 >= 12) & (w8 - w7 >= 6)).astype(float).where(w8.notna() & w7.notna())
        f = f[f.index.month.isin([10, 11, 12, 1, 2, 3])]
        temp = np.where(f.index.month >= 10, f.index.year + 1, f.index.year)
        g = f.groupby(temp)
        t = (100 * g.mean())[g.count() >= 60]
        t = t[(t.index >= 1980) & (t.index <= 2025)]
        tend = (f"{10 * sen(t.index.values.astype(float), t.values):+.2f} pp/déc (p "
                f"{mk_hamed_rao(t.index.values.astype(float), t.values)[1]:.2f}, {len(t)} temporadas)") if len(t) >= 15 \
            else f"sin tendencia ({len(t)} temporadas)"
        L.append(f"  {nom:24s} por década {dec.to_dict()}; último año {d.index.year.max()}; jet antes/después de 1990 "
                 f"{100 * f[f.index.year < 1990].mean():.1f}/{100 * f[f.index.year >= 1990].mean():.1f} %; {tend}")
    open("analisis/jet/21_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
