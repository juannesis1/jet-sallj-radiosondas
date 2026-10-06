"""Paper 2, paso 13: diferencias sondeo − ERA5 (obs-minus-analysis) en el viento de 925 y 850 hPa.

ERA5 es una referencia común a todas las estaciones con resolución vertical fija en el tiempo, así que un cambio en el
instrumento o en el procesamiento del sondeo aparece como un escalón en la serie de diferencias (es el enfoque de
RAOBCORE/RICH, Haimberger 2007; Haimberger et al. 2012, con el fondo del reanálisis en lugar de los vecinos). Salvedad:
ERA5 asimila estos mismos sondeos, así que las diferencias de análisis subestiman el escalón real (lo amortiguan).
Para cada estación: rapidez del viento a 12 UTC, oct-mar, sondeo menos punto de grilla de ERA5 (mismo día), media
mensual (≥ 10 pares); SNHT sobre la serie mensual; diferencia media antes/después de los saltos contra vecinos del paso 5
y por fuente de archivo. Para separar el escalón del sondeo de una deriva del propio ERA5, cada estación argentina se
compara además contra la media de las anomalías de estaciones brasileñas sin quiebres (referencia común, paso 22), con p por
permutación de años completos.
Salida: analisis/jet/13_resumen.txt y 13_mensual.parquet (para la figura).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

NOMBRES = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087576": "Ezeiza", "ARM00087623": "Santa Rosa",
           "BRM00083827": "Foz", "BRM00083928": "Uruguaiana", "BRM00083612": "Campo Grande", "BRM00083362": "Cuiabá",
           "BRM00083208": "Vilhena", "BRM00082824": "Porto Velho", "BRM00083768": "Londrina"}
# referencias brasileñas sin quiebres significativos contra ERA5 (selección iterativa del paso 22)

SALTOS = {"ARM00087623": "2000-02", "ARM00087344": "2002-11", "BRM00083827": "2009-10", "BRM00082824": "1992-10"}


def snht(x):
    z = (x - x.mean()) / x.std()
    n = len(z)
    t = np.array([k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2 for k in range(12, n - 12)])
    return t.max(), int(np.argmax(t)) + 12


def main():
    e = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob("data/era5_perfiles/*.parquet")))
    e["fecha"] = pd.to_datetime(e.fecha)
    e = e[(e.fecha.dt.hour == 12) & e.nivel.isin([925.0, 850.0])]
    e["w"] = np.hypot(e.u, e.v)
    e["fecha"] = e.fecha.dt.normalize()
    e = e.pivot_table(index=["sid", "fecha"], columns="nivel", values="w")
    anios = sorted({int(os.path.basename(f)[:4]) for f in glob.glob("data/era5_perfiles/*.parquet")})
    L = [f"Sondeo − ERA5, rapidez del viento (m/s), 12 UTC oct-mar; años ERA5: {anios[0]}-{anios[-1]} ({len(anios)})"]
    mens = []
    for sid, nom in NOMBRES.items():
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[(s.index.hour == 12) & s.index.month.isin([10, 11, 12, 1, 2, 3])].copy()
        s.index = s.index.normalize()
        s = s[~s.index.duplicated()]
        r = e.loc[sid].reindex(s.index)
        d = pd.DataFrame({"d925": np.hypot(s.u925, s.v925) - r[925.0], "d850": np.hypot(s.u850, s.v850) - r[850.0],
                          "fuente": s.fuente}, index=s.index).dropna(subset=["d925", "d850"], how="all")
        L.append(f"\n=== {nom}  (pares 925: {d.d925.notna().sum()}, 850: {d.d850.notna().sum()})")
        for niv in ("d925", "d850"):
            g = d[niv].groupby(d.index.to_period("M"))
            m = g.mean()[g.count() >= 10]
            mens.append(pd.DataFrame({"sid": sid, "nivel": niv, "mes": m.index.to_timestamp(), "dif": m.values}))
            if len(m) < 40:
                L.append(f"  {niv}: serie corta ({len(m)} meses)")
                continue
            t, k = snht(m.values)
            antes, despues = m.values[:k].mean(), m.values[k:].mean()
            L.append(f"  {niv}: media {m.mean():+.2f}; SNHT T={t:.0f} en {m.index[k]} ({antes:+.2f} → {despues:+.2f}, "
                     f"escalón {despues - antes:+.2f})")
            if sid in SALTOS and niv == "d925":
                c = pd.Period(SALTOS[sid], "M")
                a, b = m[(m.index < c) & (m.index >= c - 96)], m[(m.index >= c) & (m.index < c + 96)]
                if len(a) >= 10 and len(b) >= 10:
                    L.append(f"  salto vs vecinos {SALTOS[sid]}: d925 8 años antes {a.mean():+.2f} → después {b.mean():+.2f} "
                             f"(escalón {b.mean() - a.mean():+.2f}; meses {len(a)}/{len(b)})")
        f = d.groupby("fuente").agg(n=("d925", "size"), d925=("d925", "mean"), d850=("d850", "mean"))
        L.append("  por fuente:\n" + f[f.n >= 100].round(2).to_string())
    M = pd.concat(mens)
    M.to_parquet("analisis/jet/13_mensual.parquet")
    q = pd.read_csv("analisis/jet/22_quiebres.csv")
    L.append("\nQuiebres de sondeo − ERA5 por estación y su control regional (paso 22; control = cambio medio en las estaciones del otro país en los seis"
             " años anteriores y posteriores):")
    for r in q.itertuples():
        L.append(f"  {r.estacion:13s} {r.nivel} hPa {r.fecha}: paso {r.paso_m_s:+.2f} m/s, control {r.control_m_s:+.2f} (n {r.n_control}) → "
                 f"{'regional' if r.regional else 'propio de la estación'}")
    open("analisis/jet/13_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
