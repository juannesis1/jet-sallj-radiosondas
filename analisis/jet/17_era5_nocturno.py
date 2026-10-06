"""Paper 2, paso 17: ¿reproduce ERA5 la aceleración y el giro nocturnos del jet? (explicación física de los jets someros
que ERA5 no detecta).

Mismas noches que el paso 15 (estaciones brasileñas, 00 y 12 UTC del mismo día, oct-mar desde 1998), en el punto de
grilla de ERA5: cambio nocturno de la rapidez a 925 y 850 hPa y giro a 925 hPa, por clase observada a 12 UTC. Si el
esquema de capa límite de ERA5 mezcla de más en condiciones estables (Sandu et al. 2013), la aceleración nocturna cerca
del suelo y el giro inercial deberían quedar subestimados, sobre todo en las noches que terminan en jet somero.
Requiere 15_pares.parquet (paso 15) y los perfiles de ERA5 (paso 2).
Salida: analisis/jet/17_resumen.txt
"""
import glob
import os

import numpy as np
import pandas as pd

EST = {"BRM00083827": "Foz", "BRM00083612": "Campo Grande", "BRM00083362": "Cuiabá", "BRM00083928": "Uruguaiana",
       "BRM00083768": "Londrina", "BRM00083208": "Vilhena", "BRM00082824": "Porto Velho"}
rng = np.random.default_rng(11)


def giro(u0, v0, u1, v1):
    """Cambio de dirección meteorológica (°); negativo = antihorario."""
    return -np.degrees(np.arctan2(u0 * v1 - v0 * u1, u0 * u1 + v0 * v1))


def med(z, dia=None):
    """Media (no mediana: los datos son discretos) con IC 90 % por bootstrap de temporadas completas; z: Series con MultiIndex (sid, dia)."""
    z = z.dropna()
    dias = z.index.get_level_values("dia")
    tt = np.where(dias.month >= 10, dias.year + 1, dias.year)
    g = z.groupby(tt).agg(["sum", "count"])
    b = []
    for _ in range(500):
        ii = rng.integers(0, len(g), len(g))
        b.append(g["sum"].values[ii].sum() / g["count"].values[ii].sum())
    return f"{z.mean():+.1f} [{np.percentile(b, 5):+.1f},{np.percentile(b, 95):+.1f}]"


def main():
    e = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob("data/era5_perfiles/*.parquet")))
    e = e[e.sid.isin(EST) & e.nivel.isin([925.0, 850.0])]
    e["fecha"] = pd.to_datetime(e.fecha)
    e["h"] = e.fecha.dt.hour
    e["dia"] = e.fecha.dt.normalize()
    e = e.pivot_table(index=["sid", "dia"], columns=["h", "nivel"], values=["u", "v"])
    e.columns = [f"e_{c}_{h}_{int(n)}" for c, h, n in e.columns]
    P = pd.read_parquet("analisis/jet/15_pares.parquet")
    P = P[P.clase_12.isin(["A", "B", "C"])].copy()
    P.index.name = "dia"
    P = P.reset_index().set_index(["sid", "dia"])
    P = P.join(e, how="inner")
    for niv in (925.0, 850.0):
        u0, v0, u1, v1 = (P[f"e_{c}_{h}_{int(niv)}"] for h in (0, 12) for c in ("u", "v"))
        P[f"e_dw{int(niv)}"] = np.hypot(u1, v1) - np.hypot(u0, v0)
        P[f"e_rot{int(niv)}"] = giro(u0, v0, u1, v1)
        P[f"e_w12_{int(niv)}"] = np.hypot(u1, v1)
    años = sorted(P.index.get_level_values("dia").year.unique())
    L = [f"Noches 00→12 UTC, estaciones brasileñas, sondeo vs ERA5 (mismas noches; años ERA5 {años[0]}-{años[-1]})",
         "Media [IC 90 % por bootstrap de temporadas]: Δw (m/s), giro a 925 hPa (°, negativo = antihorario)"]
    for k in ("A", "B", "C"):
        x = P[P.clase_12 == k]
        L.append(f"\nclase {k} (n {len(x)})")
        L.append(f"  Δw925  sondeo {med(x.dw925)}   ERA5 {med(x.e_dw925)}")
        L.append(f"  Δw850  sondeo {med(x.dw850)}   ERA5 {med(x.e_dw850)}")
        L.append(f"  giro925 sondeo {med(x.rot925)}   ERA5 {med(x.e_rot925)}")
    L.append("\nPor estación, noches que terminan en jet somero (B): Δw925 sondeo / ERA5; giro925 sondeo / ERA5")
    for sid, nom in EST.items():
        if sid not in P.index.get_level_values("sid"):
            continue
        x = P.loc[sid]
        x = x[x.clase_12 == "B"]
        if len(x) >= 20:
            L.append(f"  {nom:13s} n {len(x):3d}: {x.dw925.median():+.1f} / {x.e_dw925.median():+.1f}   "
                     f"{x.rot925.median():+.0f} / {x.e_rot925.median():+.0f}")
    open("analisis/jet/17_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
