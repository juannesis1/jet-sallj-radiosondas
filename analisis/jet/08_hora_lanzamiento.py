"""Paper 2, paso 8: ¿cambió la hora real de lanzamiento de los sondeos de 12 UTC?

El jet nocturno se debilita después del amanecer (≈ 09-10 UTC en verano); un lanzamiento más temprano lo muestra más
intenso. Del encabezado IGRA v2 se toma RELTIME (columnas 28-31, HHMM UTC; 9999 = desconocido) de los sondeos
nominales de 12 UTC, oct-mar. Se reporta la mediana por quinquenio y la frecuencia de jet (perfil completo) según la hora.
Salida: analisis/jet/08_resumen.txt
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande", "BRM00083928": "Uruguaiana"}


def horas(sid):
    filas = []
    with zipfile.ZipFile(f"data/igra_sa/{sid}-data.txt.zip") as z, z.open(z.namelist()[0]) as fh:
        for line in io.TextIOWrapper(fh, encoding="ascii", errors="ignore"):
            if line.startswith("#"):
                y, m, d, h, r = int(line[13:17]), int(line[18:20]), int(line[21:23]), int(line[24:26]), line[27:31]
                if y >= 1980 and h == 12 and m in (10, 11, 12, 1, 2, 3) and r.strip() not in ("9999", "") and r[2:] != "99":
                    filas.append((pd.Timestamp(y, m, d, 12), int(r[:2]) + int(r[2:]) / 60))
    return pd.Series(dict(filas))


def main():
    L = ["Hora real de lanzamiento (UTC) de los sondeos nominales de 12 UTC, oct-mar: mediana y % antes de 11:30, por quinquenio"]
    for sid, nom in EST.items():
        h = horas(sid)
        if h.empty:
            continue
        g = h.groupby((h.index.year // 5) * 5)
        L.append(f"\n{nom} ({h.size} con hora): " + " ".join(f"{k}:{v:.2f}h/{100 * p:.0f}%" for (k, v), p in
                                                            zip(g.median().items(), g.apply(lambda x: (x < 11.5).mean()))))
        d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        d = d[(d.index.hour == 12)].dropna(subset=["jet_norte"])
        j = d.jet_norte.astype(float).reindex(h.index)
        temprano, tarde = j[h < 11.5].mean(), j[(h >= 11.5)].mean()
        L.append(f"   jet (perfil completo) si se lanzó antes de 11:30: {100 * temprano:.1f}% | a partir de 11:30: {100 * tarde:.1f}%")
    open("analisis/jet/08_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
