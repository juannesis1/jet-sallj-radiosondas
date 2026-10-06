"""Paper 2, paso 35: evaluabilidad y selección de la muestra del perfil completo (respuesta a las rondas 4 y 5, comentario M3).

El criterio de perfil completo exige al menos 3 niveles en los 250 hPa inferiores y un nivel sobre el núcleo; los demás sondeos no son evaluables
(NaN) y no entran en la frecuencia. Si la evaluabilidad cambia con el tiempo (archivos de 1-4 niveles en los 80-90) y si los sondeos que sí son
evaluables son justamente los que tienen niveles insertados por un rasgo del viento, la frecuencia y su tendencia dependen de esa selección.
Se informa: (a) fracción evaluable por período, fuente y estación; (b) tendencia agrupada (efectos de mes y estación, sin y con fuente) con los
no evaluables excluidos (lo que hacen los pasos 10 y 24), con los no evaluables contados como "sin jet" (NaN → 0, cota inferior de lo que se
pierde) y con los no evaluables imputados con la frecuencia de jet de los evaluables del mismo archivo-estación-mes (cota intermedia);
(c) jets con núcleo bajo: fracción de los jets con núcleo ≤ 20 hPa sobre la superficie por período y tendencia de los jets con núcleo > 50 hPa.
Salida: analisis/jet/35_resumen.txt y 35_evaluable.csv (columna de la Tabla 1)
"""
import os

import numpy as np
import importlib
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
p19 = importlib.import_module("19_emulacion_resolucion")

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande", "BRM00083928": "Uruguaiana", "BRM00083362": "Cuiabá",
       "BRM00083208": "Vilhena", "BRM00082824": "Porto Velho"}


def tend(d, y, fuente):
    x = d.dropna(subset=[y])
    X = [pd.get_dummies(x.mes).values.astype(float), ((x.temp.values - 2000) / 10.0)[:, None], pd.get_dummies(x.sid, drop_first=True).values.astype(float)]
    if fuente:
        X.append(pd.get_dummies(x.fuente, drop_first=True).values.astype(float))
    return 100 * np.linalg.lstsq(np.hstack(X), x[y].values.astype(float), rcond=None)[0][len(np.unique(x.mes))]


def boot(d, y, fuente, n=1000):
    r = np.random.default_rng(35)
    temps = d.temp.unique()
    g = {t: d[d.temp == t] for t in temps}
    b = [tend(pd.concat([g[t] for t in r.choice(temps, len(temps))]), y, fuente) for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def main():
    D = pd.read_parquet("analisis/jet/24_perfil.parquet")
    nuc = []
    for sid in EST:
        p = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")[["nucleo_dp", "jet_norte"]]
        p = p[p.index.hour == 12]
        p.index = p.index.normalize()
        p = p[~p.index.duplicated()]
        p["sid"] = sid
        nuc.append(p)
    P = pd.concat(nuc)
    P.index.name = "fecha"
    P = P.reset_index()
    P = P.drop_duplicates(["sid", "fecha"])
    D = D.merge(P[["sid", "fecha", "nucleo_dp"]], on=["sid", "fecha"], how="left")
    D["evaluable"] = D.crudo.notna().astype(float)
    D["per"] = pd.cut(D.temp, [1979, 1992, 2000, 2005, 2025], labels=["1980-92", "1993-2000", "2001-05", "2006-25"])
    L = ["(a) Fracción evaluable del perfil completo (%), 12 UTC oct-mar, 10 estaciones"]
    t = (100 * D.groupby("per", observed=True).evaluable.mean()).round(0)
    L.append("  por período: " + ", ".join(f"{k} {v:.0f} %" for k, v in t.items()))
    f = (100 * D.groupby("fuente").evaluable.agg(["mean", "size"]).query("size >= 300")["mean"]).round(0)
    L.append("  por archivo (≥ 300 sondeos): " + ", ".join(f"{k} {v:.0f} %" for k, v in f.items()))
    ev = (100 * D.groupby(["sid", "per"], observed=True).evaluable.mean()).unstack().round(0)
    L.append("  por estación y período (%):\n" + ev.rename(index=EST).to_string())
    ev.rename(index=EST).to_csv("analisis/jet/35_evaluable.csv")
    L.append("\n(b) Tendencia agrupada del perfil completo (pp/década) según el tratamiento de los no evaluables [IC 95 %, 1000 réplicas]")
    D["y_excl"] = D.crudo
    D["y_cero"] = D.crudo.fillna(0.0)
    m = D.groupby(["sid", "fuente", "mes"]).crudo.transform("mean")
    D["y_imput"] = D.crudo.fillna(m).fillna(D.crudo.mean())
    for y, lab in (("y_excl", "no evaluables excluidos (pasos 10 y 24)"), ("y_imput", "no evaluables imputados con la frecuencia de su archivo-estación-mes"),
                   ("y_cero", "no evaluables contados como sin jet (NaN → 0)")):
        for fu in (False, True):
            b = tend(D, y, fu)
            lo, hi = boot(D, y, fu)
            L.append(f"  {lab:75s} {'con fuente' if fu else 'sin fuente '}: {b:+.2f} [{lo:+.2f},{hi:+.2f}]  (media {100 * D[y].mean():.1f} %)")
    L.append("\n(c) Altura del núcleo de los jets de perfil completo (hPa sobre la superficie)")
    J = D[D.crudo == 1]
    for per, q in J.groupby("per", observed=True):
        L.append(f"  {per}: {len(q)} jets; núcleo ≤ 20 hPa {100 * (q.nucleo_dp <= 20).mean():.0f} %; ≤ 50 hPa {100 * (q.nucleo_dp <= 50).mean():.0f} %; mediana {q.nucleo_dp.median():.0f}")
    D["y_alto"] = np.where(D.crudo.isna(), np.nan, ((D.crudo == 1) & (D.nucleo_dp > 50)).astype(float))
    for fu in (False, True):
        b = tend(D, "y_alto", fu)
        lo, hi = boot(D, "y_alto", fu)
        L.append(f"  Tendencia de los jets con núcleo > 50 hPa (evaluables): {'con fuente' if fu else 'sin fuente '} {b:+.2f} [{lo:+.2f},{hi:+.2f}]")
    D["y_bajo"] = np.where(D.crudo.isna(), np.nan, ((D.crudo == 1) & (D.nucleo_dp <= 50)).astype(float))
    for fu in (False, True):
        b = tend(D, "y_bajo", fu)
        lo, hi = boot(D, "y_bajo", fu)
        L.append(f"  Tendencia de los jets con núcleo ≤ 50 hPa (evaluables): {'con fuente' if fu else 'sin fuente '} {b:+.2f} [{lo:+.2f},{hi:+.2f}]"
                 f"  (frecuencia media {100 * D.y_bajo.mean():.1f} % frente a {100 * D.y_alto.mean():.1f} % de los de núcleo > 50 hPa)")

    # (d) ¿el nivel de superficie se reporta igual en todos los archivos? (si falta, la superficie es la mayor presión reportada)
    L.append("\n(d) Fracción de sondeos que reportan el nivel de superficie (indicador 1 de IGRA) por archivo y por período")
    sup, jets2 = [], []
    for sid in EST:
        perf = p19.leer(sid)
        for k, (ps, a, fu) in perf.items():
            sup.append({"sid": sid, "fecha": k, "sup": float(p19.SUP.get((sid, k), False))})
            jets2.append({"sid": sid, "fecha": k, "y_sup10": p19.jet(ps, a, 10.0), "y_sup30": p19.jet(ps, a, 30.0)})
    D = D.merge(pd.DataFrame(sup), on=["sid", "fecha"], how="left").merge(pd.DataFrame(jets2), on=["sid", "fecha"], how="left")
    f = (100 * D.groupby("fuente").sup.agg(["mean", "size"]).query("size >= 300")["mean"]).round(0)
    L.append("  por archivo (≥ 300 sondeos): " + ", ".join(f"{k} {v:.0f} %" for k, v in f.items()))
    f = (100 * D.groupby("per", observed=True).sup.mean()).round(0)
    L.append("  por período: " + ", ".join(f"{k} {v:.0f} %" for k, v in f.items()))
    # (e) jets cuyo núcleo está estrictamente sobre la superficie
    L.append("\n(e) Perfil completo con el núcleo al menos 10 o 30 hPa por encima del nivel de superficie (excluye máximos del nivel de superficie), pp/década [IC 95 %]")
    for col, lab in (("crudo", "criterio original (núcleo desde la superficie)"), ("y_sup10", "núcleo ≥ 10 hPa sobre la superficie"), ("y_sup30", "núcleo ≥ 30 hPa sobre la superficie")):
        for fu in (False, True):
            b = tend(D, col, fu)
            lo, hi = boot(D, col, fu)
            L.append(f"  {lab:50s} {'con fuente' if fu else 'sin fuente '}: {b:+.2f} [{lo:+.2f},{hi:+.2f}]  (media {100 * D[col].mean():.1f} %, evaluables {100 * D[col].notna().mean():.0f} %)")
    open("analisis/jet/35_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
