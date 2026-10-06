"""Paper 2, paso 15: ciclo diario del jet en las estaciones brasileñas (sondeos de 00 y 12 UTC).

Las estaciones brasileñas lanzan a 00 UTC (≈ 20 h local, capa de mezcla recién colapsada) y a 12 UTC (≈ 08 h local, al
final de la noche). Si los jets someros (clase B) son jets nocturnos de capa límite (Blackadar 1957; oscilación
inercial tras el desacople de la fricción), deben:
  1) ser mucho más frecuentes a 12 UTC que a 00 UTC, más que los jets profundos (clase A);
  2) acelerarse durante la noche en las capas bajas (925 hPa) más que en 850 hPa;
  3) en el hemisferio sur, el viento ageostrófico gira en sentido antihorario: el viento a 925 hPa debe rotar
     antihorario (dirección decreciente) entre 00 y 12 UTC en las noches que terminan en jet somero.
  4) la rotación nocturna debe crecer con el parámetro de Coriolis |f| (latitud): se testea con Spearman entre
     estaciones (8.8°S a 29.8°S).
Pares nocturnos: 00 UTC y 12 UTC del mismo día (oct-mar, 1998-2025). Se compara con las noches sin jet (clase C).
Salida: analisis/jet/15_resumen.txt
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

EST = {"BRM00083827": "Foz", "BRM00083612": "Campo Grande", "BRM00083362": "Cuiabá", "BRM00083928": "Uruguaiana",
       "BRM00083768": "Londrina", "BRM00083208": "Vilhena", "BRM00082824": "Porto Velho"}
LAT = {"BRM00083827": -25.5, "BRM00083612": -20.5, "BRM00083362": -15.6, "BRM00083928": -29.8, "BRM00083768": -23.3,
       "BRM00083208": -12.7, "BRM00082824": -8.8}
rng = np.random.default_rng(3)


def clases(s):
    w8, w7 = np.hypot(s.u850, s.v850), np.hypot(s.u700, s.v700)
    dr = (np.degrees(np.arctan2(-s.u850, -s.v850)) + 360) % 360
    f = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).where(w8.notna() & w7.notna())
    c = pd.Series(np.where(f == 1, "A", np.where(s.jet_norte == True, "B", "C")), index=s.index)  # noqa: E712
    return c.where(f.notna() & s.jet_norte.notna())


def main():
    L = ["Ciclo diario (oct-mar, 1998-2025): frecuencia de clases a 00 y 12 UTC; cambio nocturno 00→12 UTC del viento",
         "Δw = cambio de rapidez (m/s); rot = cambio de dirección (°, negativo = antihorario, lo esperado en el HS)"]
    tot = []
    for sid, nom in EST.items():
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[(np.where(s.index.month >= 10, s.index.year + 1, s.index.year) >= 1998) & s.index.month.isin([10, 11, 12, 1, 2, 3]) & s.index.hour.isin([0, 12])].copy()
        s["clase"] = clases(s)
        h0, h12 = s[s.index.hour == 0], s[s.index.hour == 12]
        h0.index, h12.index = h0.index.normalize(), h12.index.normalize()
        h0, h12 = h0[~h0.index.duplicated()], h12[~h12.index.duplicated()]
        L.append(f"\n=== {nom}")
        for k in ("A", "B"):
            f0, f12 = 100 * (h0.clase == k)[h0.clase.notna()].mean(), 100 * (h12.clase == k)[h12.clase.notna()].mean()
            L.append(f"  clase {k}: 00 UTC {f0:5.1f} %  12 UTC {f12:5.1f} %  (razón {f12 / max(f0, 0.1):.1f})  "
                     f"núcleo mediano 00/12 UTC {h0.nucleo_dp[h0.clase == k].median():.0f}/{h12.nucleo_dp[h12.clase == k].median():.0f} hPa")
        p = h0.join(h12, lsuffix="_0", rsuffix="_12", how="inner")
        for niv in (925, 850):
            u0, v0, u1, v1 = p[f"u{niv}_0"], p[f"v{niv}_0"], p[f"u{niv}_12"], p[f"v{niv}_12"]
            p[f"dw{niv}"] = np.hypot(u1, v1) - np.hypot(u0, v0)
            rot = np.degrees(np.arctan2(u0 * v1 - v0 * u1, u0 * u1 + v0 * v1))  # positivo = antihorario
            p[f"rot{niv}"] = -rot  # en grados de dirección meteorológica: negativo = antihorario
        p["sid"] = sid
        tot.append(p)
        g = p[p.clase_12.isin(["A", "B", "C"])].groupby("clase_12")
        L.append("  noches que terminan en cada clase (12 UTC):\n" + pd.DataFrame(
            {"n": g.size(), "Δw925": g.dw925.median(), "Δw850": g.dw850.median(),
             "rot925": g.rot925.median(), "rot850": g.rot850.median()}).round(1).to_string())
    P = pd.concat(tot)
    P.to_parquet("analisis/jet/15_pares.parquet")
    P = P[P.clase_12.isin(["A", "B", "C"])]
    L.append("\n=== Todas las estaciones juntas (media e IC 90 % por bootstrap de temporadas completas; las medianas de datos discretizados"
             " en nudos y en 5° de rumbo dan intervalos degenerados)")
    rb = np.random.default_rng(15)
    for k in ("A", "B", "C"):
        x = P[P.clase_12 == k]
        fila = [f"  clase {k} (n {len(x)}):"]
        for v in ("dw925", "dw850", "rot925", "rot850"):
            z = x[v].dropna()
            tt = np.where(z.index.month >= 10, z.index.year + 1, z.index.year)
            g = z.groupby(tt).agg(["sum", "count"])
            bs = []
            for _ in range(500):
                ii = rb.integers(0, len(g), len(g))
                bs.append(g["sum"].values[ii].sum() / g["count"].values[ii].sum())
            fila.append(f"{v} {z.mean():+.1f} [{np.percentile(bs, 5):+.1f},{np.percentile(bs, 95):+.1f}]")
        L.append("  ".join(fila))
    f = {sid: 2 * 7.292e-5 * abs(np.sin(np.radians(la))) for sid, la in LAT.items()}
    L.append("\nRotación nocturna a 925 hPa vs |f| entre estaciones (Spearman); rotación inercial teórica en 12 h = f·12 h:")
    for k in ("A", "B", "C"):
        m = P[P.clase_12 == k].groupby("sid").rot925.median()
        r, pv = stats.spearmanr([f[i] for i in m.index], m.values)
        L.append(f"  clase {k}: rho {r:+.2f} (p {pv:.3f}); " + ", ".join(
            f"{EST[i]} {m[i]:+.0f}° (teórica {np.degrees(f[i] * 43200):.0f}°)" for i in m.index))
    open("analisis/jet/15_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
