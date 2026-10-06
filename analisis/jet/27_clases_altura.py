"""Paper 2, paso 27: ¿los jets someros son un fenómeno físico distinto del profundo, sin depender de la definición por niveles?

Las clases A (jet detectado con 850/700 hPa) y B (sólo con el perfil) se definen por el método de detección, así que comparar la
velocidad en 850 hPa entre ellas es casi tautológico (revisión, ronda 2, M5). Acá se clasifican los jets de perfil completo (norte, 12 UTC,
oct-mar, 2006-2025, alta resolución en todos los sondeos) por la ALTURA de su núcleo sobre el suelo, que no depende de los niveles fijos:
bajos (núcleo ≤ 60 hPa sobre la superficie, unos 500 m) y altos (> 60 hPa).
(a) Perfiles compuestos de rapidez en función de la altura sobre el suelo, para las clases A, B y C (Resistencia, Córdoba, Santa Rosa,
    Foz; 2006-2025), con mediana y rango intercuartil.
(b) Cociente de frecuencias 12 UTC / 00 UTC (estaciones brasileñas, 2006-2025) para jets bajos y altos.
(c) Fracción de cada clase detectada por ERA5 (perfil) y rapidez de los núcleos.
(d) Giro nocturno a 925 hPa de las noches que terminan en jets bajos vs sin jet, normalizado por la rotación inercial f·12 h (el giro
    del viento total es sólo una fracción: se reporta la fracción y se compara con las noches sin jet).
Salida: analisis/jet/27_resumen.txt y 27_perfiles.parquet (figura 4a)
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
p19 = importlib.import_module("19_emulacion_resolucion")
EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "BRM00083827": "Foz"}
BR = {"BRM00083827": -25.6, "BRM00083612": -20.5, "BRM00083362": -15.6, "BRM00083928": -29.8, "BRM00083768": -23.3,
      "BRM00083208": -12.7, "BRM00082824": -8.8}
GRID = np.arange(0, 301, 10.0)


def main():
    L = ["(a) Perfiles compuestos de rapidez (m/s) por altura sobre la superficie (hPa), 2006-2025, 12 UTC oct-mar: mediana [Q1,Q3]"]
    filas = []
    for sid, nom in EST.items():
        perf = p19.leer(sid)
        pf = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        pf = pf[(pf.index.hour == 12) & (pf.index.year >= 2006)]
        pf.index = pf.index.normalize()
        pf = pf[~pf.index.duplicated()]
        for k, (ps, a, fu) in perf.items():
            if k.year < 2006 or k not in pf.index or len(a) < 4:
                continue
            q = pf.loc[k]
            w8, w7 = np.hypot(q.u850, q.v850), np.hypot(q.u700, q.v700)
            if not (np.isfinite(w8) and np.isfinite(w7)):
                continue
            dr = (np.degrees(np.arctan2(-q.u850, -q.v850)) + 360) % 360
            fijo = (w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))
            cl = "A" if fijo else ("B" if q.jet_norte == True else "C")  # noqa: E712
            rel = ps - a[:, 0]
            o = np.argsort(rel)
            perfil = np.interp(GRID, rel[o], a[o, 2], right=np.nan)
            filas.append({"sid": sid, "clase": cl, **{f"r{int(g)}": v for g, v in zip(GRID, perfil)}})
    P = pd.DataFrame(filas)
    out = []
    for cl, g in P.groupby("clase"):
        for gi in GRID:
            v = g[f"r{int(gi)}"].dropna()
            out.append({"clase": cl, "altura": gi, "q1": v.quantile(0.25), "med": v.median(), "q3": v.quantile(0.75), "n": len(v)})
    O = pd.DataFrame(out)
    O.to_parquet("analisis/jet/27_perfiles.parquet")
    for cl, g in O.groupby("clase"):
        sel = g[g.altura.isin([0, 30, 60, 100, 150, 200, 300])]
        L.append(f"  clase {cl} (n {int(g.n.max())}): " + "; ".join(f"{r.altura:.0f} hPa: {r.med:.1f} [{r.q1:.1f},{r.q3:.1f}]" for r in sel.itertuples()))

    # ---- (b), (c), (d) por altura del núcleo en las estaciones brasileñas
    L.append("\n(b) Jets de perfil completo (norte) según la altura del núcleo, estaciones brasileñas, 2006-2025: frecuencia (%) a 00 y 12 UTC")
    tot = {"bajo": [0, 0], "alto": [0, 0]}
    n0 = n12 = 0
    for sid in BR:
        d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        d = d[(d.index.year >= 2006) & d.index.month.isin([10, 11, 12, 1, 2, 3]) & d.index.hour.isin([0, 12])]
        d = d[d.jet_norte.notna()]
        h12, h0 = d[d.index.hour == 12], d[d.index.hour == 0]
        n0, n12 = n0 + len(h0), n12 + len(h12)
        for nom, f in (("bajo", lambda x: (x.jet_norte == True) & (x.nucleo_dp <= 60)), ("alto", lambda x: (x.jet_norte == True) & (x.nucleo_dp > 60))):  # noqa: E712
            tot[nom][0] += f(h0).sum()
            tot[nom][1] += f(h12).sum()
    for nom, (a0, a12) in tot.items():
        L.append(f"  núcleo {nom:4s}: 00 UTC {100 * a0 / n0:.1f} %  12 UTC {100 * a12 / n12:.1f} %  cociente {a12 / n12 / max(a0 / n0, 1e-9):.1f}")

    L.append("\n(c) Fracción de los jets observados (12 UTC, 2006-2025) que ERA5 detecta con el perfil, por altura del núcleo (todas las estaciones con ERA5)")
    r = pd.read_parquet("analisis/jet/12_diario.parquet")
    r = r[r.temp >= 2006]
    r["fecha"] = pd.to_datetime(r.fecha)
    acum = []
    for sid in r.sid.unique():
        d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        d = d[(d.index.hour == 12) & (d.index.year >= 2006)]
        d.index = d.index.normalize()
        d = d[~d.index.duplicated()]
        x = r[r.sid == sid].set_index("fecha").join(d[["jet_norte", "nucleo_dp", "nucleo_v"]], how="inner")
        acum.append(x[x.jet_norte == True])  # noqa: E712
    J = pd.concat(acum)
    J["era_perfil"] = J.era_perfil.astype(float)
    for nom, q in (("bajo (≤ 60 hPa)", J[J.nucleo_dp <= 60]), ("alto (> 60 hPa)", J[J.nucleo_dp > 60])):
        L.append(f"  núcleo {nom:16s}: n {len(q):5d}, rapidez mediana {q.nucleo_v.median():.1f} m/s, ERA5 detecta {100 * q.era_perfil.mean():.0f} %")
    for lo, hi in ((0, 30), (30, 60), (60, 100), (100, 150), (150, 250)):
        q = J[(J.nucleo_dp > lo) & (J.nucleo_dp <= hi)]
        L.append(f"    núcleo {lo:3d}-{hi:3d} hPa sobre el suelo: n {len(q):5d}, rapidez {q.nucleo_v.median():.1f} m/s, ERA5 detecta {100 * q.era_perfil.mean():.0f} %")
    # fracción del giro inercial en noches con jet bajo vs sin jet
    L.append("\n(d) Giro nocturno a 925 hPa como fracción de f·12 h (mediana por estación; teórico = 100 %), noches que terminan en jet vs sin jet")
    P15 = pd.read_parquet("analisis/jet/15_pares.parquet")
    P15 = P15[P15.clase_12.isin(["A", "B", "C"])]
    for cl in ("A", "B", "C"):
        fr = []
        for sid, lat in BR.items():
            q = P15[(P15.sid == sid) & (P15.clase_12 == cl)]
            if len(q) >= 20:
                fr.append(100 * abs(q.rot925.median()) / np.degrees(2 * 7.292e-5 * abs(np.sin(np.radians(lat))) * 43200))
        L.append(f"  clase {cl}: {np.median(fr):.0f} % (rango entre estaciones {min(fr):.0f}–{max(fr):.0f} %)")
    # ---- (e) sensibilidad al umbral de velocidad del núcleo (los jets someros están pegados al umbral de 12 m/s)
    L.append("\n(e) Sensibilidad al umbral de velocidad (11, 12 y 13 m/s; caída ≥ 6 m/s), sondeos 2006-2025 de Resistencia, Córdoba, Santa Rosa y Foz do Iguaçu")
    nivel = lambda a, p0: (a[np.abs(a[:, 0] - p0) < 2][0] if (np.abs(a[:, 0] - p0) < 2).any() else None)  # noqa: E731
    perfs = {sid: p19.leer(sid) for sid in EST}
    for th in (11.0, 12.0, 13.0):
        p19.MIN_NUCLEO = th
        nA = nB = nT = 0
        cerca = 0
        vB = []
        for sid, perf in perfs.items():
            for k, (ps, a, fu) in perf.items():
                if k.year < 2006 or len(a) < 4:
                    continue
                a8, a7 = nivel(a, 850.0), nivel(a, 700.0)
                if a8 is None or a7 is None:
                    continue
                j = p19.jet(ps, a)
                if not np.isfinite(j):
                    continue
                fijo = a8[2] >= th and a8[2] - a7[2] >= 6 and (a8[1] >= 292.5 or a8[1] <= 67.5)
                nT += 1
                nA += bool(fijo)
                if j == 1 and not fijo:
                    nB += 1
                    dp = ps - a[:, 0]
                    bajo = (dp >= 0) & (dp <= 250)
                    v = a[bajo, 2].max()
                    vB.append(v)
                    cerca += v < th + 1
        L.append(f"  umbral {th:.0f} m/s: deep {100 * nA / nT:.1f} %, shallow {100 * nB / nT:.1f} % de los sondeos; cociente B/A {nB / max(nA, 1):.1f};"
                 f" núcleo mediano de B {np.median(vB):.1f} m/s; {100 * cerca / max(nB, 1):.0f} % de B a menos de 1 m/s del umbral")
    p19.MIN_NUCLEO = 12.0
    open("analisis/jet/27_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
