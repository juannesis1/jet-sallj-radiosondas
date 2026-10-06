"""Paper 2, paso 9: efecto de la fuente de archivo (IGRA) sobre la frecuencia del jet observada.

IGRA v2 arma cada serie con sondeos de distintos archivos (p. ej. ncdc6301/6310/6314/6322, usaf-ds3, ncdc-gts), con
distinta cantidad de niveles. Para separar el efecto de la fuente del clima, se comparan fuentes DENTRO de la misma
temporada (oct-mar, 12 UTC): diferencia de frecuencia de jet entre sondeos de distintas fuentes en temporadas donde
ambas aportan ≥ 20 sondeos (efecto fijo de temporada). También: niveles medios en 0-250 hPa sobre la superficie,
disponibilidad de viento a 925/850/700 hPa y frecuencia de jet por fuente y criterio.
Salida: analisis/jet/09_resumen.txt
"""
import os

import numpy as np
import pandas as pd

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande"}


def preparar(sid):
    d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
    d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3]) & (d.index.year >= 1980)].copy()
    w8, w7 = np.hypot(d.u850, d.v850), np.hypot(d.u700, d.v700)
    dr = (np.degrees(np.arctan2(-d.u850, -d.v850)) + 360) % 360
    d["fijos"] = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).where(w8.notna() & w7.notna())
    d["completo"] = d.jet_norte.astype(float)
    d["temp"] = np.where(d.index.month >= 10, d.index.year + 1, d.index.year)
    return d


def main():
    L = []
    for sid, nom in EST.items():
        d = preparar(sid)
        L.append(f"\n=== {nom}")
        g = d.groupby("fuente")
        t = pd.DataFrame({"sondeos": g.size(), "años": g.temp.agg(lambda x: f"{x.min()}-{x.max()}"),
                          "niveles_0-250": g.n_niveles.median(), "%925": g.u925.apply(lambda x: 100 * x.notna().mean()),
                          "%850/700": g.apply(lambda x: 100 * (x.u850.notna() & x.u700.notna()).mean(), include_groups=False),
                          "jet_completo%": 100 * g.completo.mean(), "jet_fijos%": 100 * g.fijos.mean()})
        L.append(t[t.sondeos >= 100].round(1).to_string())
        # comparación dentro de la misma temporada
        pares = []
        for temp, x in d.groupby("temp"):
            c = x.fuente.value_counts()
            c = c[c >= 20]
            if len(c) >= 2:
                f = x[x.fuente.isin(c.index)].groupby("fuente")[["completo", "fijos", "n_niveles"]].mean()
                for a in f.index:
                    for b in f.index:
                        if a < b:
                            pares.append({"temp": temp, "a": a, "b": b, "nmin": int(min(c[a], c[b])), "dif_completo": 100 * (f.completo[a] - f.completo[b]),
                                          "dif_fijos": 100 * (f.fijos[a] - f.fijos[b]), "dif_niveles": f.n_niveles[a] - f.n_niveles[b]})
        if pares:
            P = pd.DataFrame(pares)
            r_ = np.random.default_rng(9)
            filas = []
            for (a, b), q in P.groupby(["a", "b"]):
                fila = {"a": a, "b": b, "temporadas": len(q), "nmin": int(q.nmin.min()), "nmed": int(q.nmin.median())}
                for c in ("dif_completo", "dif_fijos", "dif_niveles"):
                    bs = [q[c].values[r_.integers(0, len(q), len(q))].mean() for _ in range(1000)]
                    fila[c] = q[c].mean()
                    if c != "dif_niveles":
                        fila[c + "_IC90"] = f"[{np.percentile(bs, 5):+.1f},{np.percentile(bs, 95):+.1f}]" if len(q) >= 3 else "[n<3]"
                filas.append(fila)
            L.append("  Misma temporada, fuente a − fuente b (pp; niveles): temporadas, sondeos por celda (mín./mediana del menor grupo), IC 90 % entre temporadas (sólo con ≥ 3):")
            L.append(pd.DataFrame(filas).set_index(["a", "b"]).round(1).to_string())
    open("analisis/jet/09_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
