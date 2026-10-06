"""Paper 2, paso 33: ¿depende de la velocidad del viento que un sondeo reporte el nivel de 925 hPa? (respuesta a la revisión de la ronda 3, M5)

En el archivo USAF sólo 31-38 % de los sondeos reportan 925 hPa. Si ese reporte fuera un nivel significativo (elegido porque el viento
tiene un rasgo en él), la serie de diferencias sondeo − ERA5 a 925 hPa y sus saltos de 1997-2000 podrían ser artefactos de muestreo. Test: dentro
de la misma estación y temporada, comparar la rapidez a 850 y 700 hPa (niveles que se reportan en ambos grupos, así que no dependen de la
decisión de reportar 925) y la frecuencia de jets de niveles fijos entre sondeos con y sin 925 hPa reportado (archivo USAF, 12 UTC oct-mar).
Diferencia media entre temporadas (ponderada por el menor de los dos tamaños) con IC 90 % por bootstrap de temporadas.
Salida: analisis/jet/33_resumen.txt
"""
import os

import numpy as np
import pandas as pd

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande", "BRM00083928": "Uruguaiana", "BRM00083362": "Cuiabá",
       "BRM00083208": "Vilhena", "BRM00082824": "Porto Velho"}
rng = np.random.default_rng(33)


def main():
    filas = []
    for sid, nom in EST.items():
        d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3]) & (d.fuente == "usaf-ds3") & (d.index.year >= 1980)].copy()
        d["temp"] = np.where(d.index.month >= 10, d.index.year + 1, d.index.year)
        d["w850"], d["w700"] = np.hypot(d.u850, d.v850), np.hypot(d.u700, d.v700)
        dr = (np.degrees(np.arctan2(-d.u850, -d.v850)) + 360) % 360
        d["fijo"] = ((d.w850 >= 12) & (d.w850 - d.w700 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).astype(float).where(d.w850.notna() & d.w700.notna())
        d["con925"] = d.u925.notna()
        for t, q in d.groupby("temp"):
            a, b = q[q.con925], q[~q.con925]
            if len(a) >= 15 and len(b) >= 15:
                for v in ("w850", "w700", "fijo"):
                    filas.append({"sid": sid, "temp": t, "var": v, "dif": a[v].mean() - b[v].mean(), "peso": min(len(a), len(b))})
    R = pd.DataFrame(filas)
    L = ["Sondeos USAF con y sin 925 hPa reportado, dentro de la misma estación y temporada: (con − sin)"]
    if R.empty:
        L.append("  sin temporadas con ≥ 15 sondeos de cada grupo")
    else:
        L.append(f"  {R.groupby('var').size().min()} estaciones-temporada comparables; {R.sid.nunique()} estaciones")
        for v, lab, esc in (("w850", "rapidez a 850 hPa (m/s)", 1), ("w700", "rapidez a 700 hPa (m/s)", 1), ("fijo", "frecuencia de jet de niveles fijos (pp)", 100)):
            q = R[R["var"] == v]
            m = np.average(q.dif, weights=q.peso)
            temps = q.temp.unique()
            bs = []
            for _ in range(500):
                ts = rng.choice(temps, len(temps))
                z = pd.concat([q[q.temp == t] for t in ts])
                bs.append(np.average(z.dif, weights=z.peso))
            L.append(f"  {lab:42s} {esc * m:+.2f} [{esc * np.percentile(bs, 5):+.2f},{esc * np.percentile(bs, 95):+.2f}]  ({len(q)} estaciones-temporada, {int(q.temp.min())}–{int(q.temp.max())})")
    open("analisis/jet/33_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
