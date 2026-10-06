"""Paper 2, paso 14: ¿qué física distingue a los jets profundos (clase A) de los someros (clase B)?

Clases como en el paso 7 (1998-2025, oct-mar, 12 UTC): A = jet con niveles fijos 850/700 hPa; B = jet sólo con el
perfil completo; C = sin jet. Dos hipótesis clásicas:
  1) Jet nocturno de capa límite (Blackadar 1957; oscilación inercial): los someros necesitan una capa nocturna estable.
     Estabilidad: θ(850 hPa) − θ(superficie), ambos niveles reportados en todas las épocas.
  2) Jet sinóptico prefrontal (Salio et al. 2002; Saulo et al. 2007): los profundos preceden a la llegada de un frente.
     Frente: viento del sur a 850 hPa (v850 > 2 m/s) en alguno de los dos sondeos de 12 UTC siguientes.
Además: rapidez a 850 hPa y presión de superficie (anomalía respecto de la media del mes de la estación).
Salida: analisis/jet/14_resumen.txt
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande"}


def temperaturas(sid):
    """θ en superficie y 850 hPa de los sondeos de 12 UTC desde 1998."""
    filas, cab, ts, t8 = [], None, np.nan, np.nan
    with zipfile.ZipFile(f"data/igra_sa/{sid}-data.txt.zip") as z, z.open(z.namelist()[0]) as fh:
        for line in io.TextIOWrapper(fh, encoding="ascii", errors="ignore"):
            if line.startswith("#"):
                if cab is not None:
                    filas.append((cab, *ts, t8))
                y, h = int(line[13:17]), int(line[24:26])
                cab = pd.Timestamp(y, int(line[18:20]), int(line[21:23])) if (y >= 1998 or (y == 1997 and int(line[18:20]) >= 10)) and h == 12 else None
                ts, t8 = (np.nan, np.nan), np.nan
                continue
            if cab is None:
                continue
            p, t = line[9:15].strip(), line[22:27].strip()
            if p.startswith("-") or t.startswith("-"):
                continue
            p, t = float(p) / 100, float(t) / 10 + 273.15
            if line[1] == "1":
                ts = (p, t * (1000 / p) ** 0.286)
            elif p == 850.0:
                t8 = t * (1000 / 850) ** 0.286
    if cab is not None:
        filas.append((cab, *ts, t8))
    d = pd.DataFrame(filas, columns=["fecha", "p_sfc", "th_sfc", "th850"]).set_index("fecha")
    return d[~d.index.duplicated()]


def main():
    L = ["Clases de jet (oct-mar, 12 UTC, 1998-2025): A profundo (850/700), B somero (sólo perfil), C sin jet",
         "estab = θ850 − θsup (K); frente = viento del sur a 850 hPa en las 48 h siguientes (%); "
         "Δps = anomalía de presión de superficie (hPa)"]
    for sid, nom in EST.items():
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[s.index.hour == 12].copy()
        s.index = s.index.normalize()
        s = s[~s.index.duplicated()]
        w8, w7 = np.hypot(s.u850, s.v850), np.hypot(s.u700, s.v700)
        dr = (np.degrees(np.arctan2(-s.u850, -s.v850)) + 360) % 360
        fijos = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).where(w8.notna() & w7.notna())
        sur = (s.v850 > 2).astype(float).where(s.v850.notna())
        s["frente"] = pd.concat([sur.shift(-1, freq="D"), sur.shift(-2, freq="D")], axis=1, sort=True).reindex(s.index).max(axis=1)
        s["clase"] = np.where(fijos == 1, "A", np.where(s.jet_norte == True, "B", "C"))  # noqa: E712
        s.loc[fijos.isna() | s.jet_norte.isna(), "clase"] = None
        t = temperaturas(sid)
        s = s.join(t, how="inner")
        s = s[s.index.month.isin([10, 11, 12, 1, 2, 3])].dropna(subset=["clase"])
        s["estab"] = s.th850 - s.th_sfc
        s["dps"] = s.p_sfc - s.groupby(s.index.month).p_sfc.transform("median")
        s["w850"] = w8.reindex(s.index)
        g = s.groupby("clase")
        r = pd.DataFrame({"n": g.size(), "estab": g.estab.median(), "estab>6K %": g.estab.apply(lambda x: 100 * (x > 6).mean()),
                          "frente %": 100 * g.frente.mean(), "Δps": g.dps.median(), "w850": g.w850.median(),
                          "núcleo hPa": g.nucleo_dp.median()})
        L.append(f"\n=== {nom}\n" + r.round(1).to_string())
    open("analisis/jet/14_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
