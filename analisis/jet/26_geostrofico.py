"""Paper 2, paso 26: control independiente de superficie con el viento geostrófico (respuesta a la revisión de la ronda 2, M8).

El viento de superficie de un aeropuerto depende del anemómetro y del desacople nocturno, así que es un control débil. El gradiente
horizontal de presión a nivel del mar entre aeropuertos cercanos, en cambio, no depende del anemómetro ni de la fricción: un plano
ajustado por mínimos cuadrados a la presión (SLP) de Resistencia, Córdoba, Santa Rosa y Ezeiza a las 12 UTC (± 1 h) da el viento
geostrófico (u_g = −(1/ρf) ∂p/∂y, v_g = (1/ρf) ∂p/∂x; f < 0 en el hemisferio sur). Si el jet se hubiera vuelto el doble de frecuente, el
flujo geostrófico del norte debería aumentar. Sólo se usa SLP (el reglaje altimétrico no es comparable con la SLP).
Se informa: disponibilidad de SLP por quinquenio; frecuencia de días con flujo geostrófico del norte ≥ 4 m/s y su media por temporada;
tendencia (Sen, MK-HR, 1980-2025 y 1993-2025); diferencia antes/después (8 temporadas) de los saltos de los sondeos; correlación diaria
con el jet de niveles fijos observado en los sondeos (sanidad del diagnóstico).
Salida: analisis/jet/26_resumen.txt
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from estadistica import mk_hamed_rao, sen  # noqa: E402

EST = {"87155099999": ("Resistencia", -27.45, -59.05), "87344099999": ("Córdoba", -31.32, -64.22),
       "87623099999": ("Santa Rosa", -36.57, -64.27), "87576099999": ("Ezeiza", -34.82, -58.54)}
R, OMEGA, RHO = 6371.0, 7.292e-5, 1.2


def slp(usaf):
    partes = []
    for f in sorted(glob.glob(f"data/isd_full/{usaf}_*.csv")):
        d = pd.read_csv(f, usecols=["DATE", "SLP"], dtype=str)
        p = pd.to_numeric(d.SLP.str.split(",", expand=True)[0], errors="coerce").values / 10
        q = d.SLP.str.split(",", expand=True)[1].values
        x = pd.DataFrame({"p": p, "q": q}, index=pd.to_datetime(d.DATE).values)
        x = x[(x.p > 950) & (x.p < 1060) & x.q.isin(["1", "5", "0", "4", "9"])]
        x = x[(x.index.hour >= 11) & (x.index.hour <= 13)]
        x["dist"] = np.abs(x.index.hour * 60 + x.index.minute - 720)
        x = x.sort_values("dist")
        x.index = x.index.normalize()
        partes.append(x.groupby(level=0).first()[["p"]])
    return pd.concat(partes).p


def main():
    P = pd.concat({nom: slp(u) for u, (nom, la, lo) in EST.items()}, axis=1)
    P = P[~P.index.duplicated()].sort_index()
    P = P[P.index.month.isin([10, 11, 12, 1, 2, 3]) & (P.index.year >= 1980)]
    lat0 = np.mean([v[1] for v in EST.values()])
    xy = np.array([[R * np.cos(np.radians(lat0)) * np.radians(lo), R * np.radians(la)] for (_, la, lo) in EST.values()])
    A = np.column_stack([np.ones(4), xy - xy.mean(axis=0)])
    f = 2 * OMEGA * np.sin(np.radians(lat0))      # < 0
    L = ["Viento geostrófico del plano de SLP de 4 aeropuertos, 12 UTC oct-mar"]
    disp = P.notna().sum(axis=1)
    L.append("  Días con las 4 estaciones: " + ", ".join(f"{a}-{a + 4}: {100 * (disp[(disp.index.year >= a) & (disp.index.year <= a + 4)] == 4).mean():.0f} %"
                                                       for a in range(1980, 2025, 5)))
    P4 = P[disp == 4]
    coef = np.linalg.lstsq(A, (P4.values * 100).T, rcond=None)[0]       # Pa; pendientes en Pa/km
    dpdx, dpdy = coef[1] / 1000, coef[2] / 1000                          # Pa/m
    vg, ug = dpdx / (RHO * f), -dpdy / (RHO * f)
    G = pd.DataFrame({"norte": -vg, "oeste": ug}, index=P4.index)        # norte > 0: flujo desde el norte
    G["temp"] = np.where(G.index.month >= 10, G.index.year + 1, G.index.year)
    G["fuerte"] = (G.norte >= 4).astype(float)
    g = G.groupby("temp")
    S = pd.DataFrame({"fuerte": 100 * g.fuerte.mean(), "media": g.norte.mean(), "n": g.size()})
    S = S[S.n >= 60]
    L.append(f"  Frecuencia media de flujo geostrófico del norte ≥ 4 m/s: {100 * G.fuerte.mean():.1f} % de los días; media {G.norte.mean():+.2f} m/s")
    L.append("  Tendencia (pp/década y m/s/década):")
    for ini in (1980, 1993):
        z = S[S.index >= ini]
        for c, un in (("fuerte", "pp"), ("media", "m/s")):
            L.append(f"    desde {ini} {c:7s}: {10 * sen(z.index.values.astype(float), z[c].values):+.2f} {un}/déc "
                     f"(p MK-HR {mk_hamed_rao(z.index.values.astype(float), z[c].values)[1]:.2f}, {len(z)} temporadas)")
    L.append("  Antes → después de los saltos de los sondeos (8 temporadas a cada lado), % de días con flujo del norte ≥ 4 m/s:")
    for nom, c in (("Resistencia / Ezeiza (dic 1997)", 1998), ("Santa Rosa (feb 2000)", 2000), ("Córdoba (nov 2002)", 2003)):
        a, d = S.loc[c - 8:c - 1, "fuerte"].mean(), S.loc[c + 1:c + 8, "fuerte"].mean()
        L.append(f"    {nom:34s} {a:.1f} → {d:.1f}")
    # sanidad: relación con el jet de niveles fijos observado en los sondeos (Resistencia, Córdoba, Santa Rosa)
    F = pd.read_parquet("analisis/jet/22_fijos.parquet")
    F = F[F.sid.isin(["ARM00087155", "ARM00087344", "ARM00087623"])]
    F["dia"] = F.index.normalize()
    j = F.groupby("dia").fijos.max()
    x = pd.concat([G.norte, j], axis=1).dropna()
    L.append(f"  Sanidad: flujo geostrófico del norte medio en días con jet fijo observado en algún sondeo {x[x.fijos == 1].norte.mean():+.1f} m/s"
             f" (n {int((x.fijos == 1).sum())}) frente a {x[x.fijos == 0].norte.mean():+.1f} m/s sin jet (n {int((x.fijos == 0).sum())})")
    open("analisis/jet/26_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
