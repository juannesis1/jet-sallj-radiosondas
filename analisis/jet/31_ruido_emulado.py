"""Paper 2, paso 31: ¿puede el ruido de los sondeos antiguos esconder un aumento real del jet? Sensibilidad al ruido, al redondeo y a la resolución.

El rms de la diferencia sondeo − ERA5 del viento a 850 hPa baja de 3.5 a 1.7 m/s entre 1980-95 y 2011-25 (paso 23; en días con jet de 7.3 a 2.7 m/s):
podría reflejar sondeos menos ruidosos, menos error de ERA5, o ambas cosas; en los días con jet, además, la diferencia incluye regresión a la media y error de
representatividad, de modo que NO se traduce en una desviación de ruido del sondeo ni da una cota. Los jets se definen con umbrales (≥ 12 m/s y caída ≥ 6 m/s)
sobre una cola rara y el ruido infla la frecuencia de excedencias. Este paso mide cuánto sube la frecuencia con ruido gaussiano de desvío σ por
componente y nivel, independiente (variante A) o correlacionado en la vertical con escala de 0.15 en ln p (variante B), con el redondeo de los archivos
antiguos (nudo entero, rumbo a 10°), y combinado con la degradación de resolución (plantillas de sondeos de 1980-89 del mismo mes, emulador de posiciones
fijas y adaptativo, paso 19) para el perfil completo. Sondeos modernos (2006-2025) de seis estaciones; 6 sorteos por sondeo y escenario; IC 95 % por bootstrap de
temporadas completas. El sesgo de tendencia es la diferencia de frecuencia dividida por 2.65 décadas (años medios de 1980-92 y 2001-25): es lo que restaría a la
tendencia observada si el índice de los años 80 estuviera inflado por ese ruido.
Salida: analisis/jet/31_resumen.txt
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
p19 = importlib.import_module("19_emulacion_resolucion")
EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande"}
DECADAS = 2.65
NUDO = 0.514444
NSORT = 6
rng = np.random.default_rng(31)


def uv(a):
    d = np.radians(a[:, 1])
    return -a[:, 2] * np.sin(d), -a[:, 2] * np.cos(d)


def a_polar(u, v, p):
    w = np.hypot(u, v)
    d = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    return np.column_stack([p, d, w])


def nivel(a, p0):
    k = np.where(np.abs(a[:, 0] - p0) < 2)[0]
    return a[k[0]] if len(k) else None


def fijo(a):
    a8, a7 = nivel(a, 850.0), nivel(a, 700.0)
    if a8 is None or a7 is None:
        return np.nan
    return float(a8[2] >= 12 and a8[2] - a7[2] >= 6 and (a8[1] >= 292.5 or a8[1] <= 67.5))


def ruido(n_niv, lnp, sigma, esc):
    """Ruido gaussiano (n_niv,): independiente (esc = 0) o correlacionado en la vertical con escala esc en ln p."""
    if esc <= 0:
        return rng.normal(0, sigma, n_niv)
    C = np.exp(-np.abs(lnp[:, None] - lnp[None, :]) / esc)
    return np.linalg.cholesky(C + 1e-9 * np.eye(n_niv)) @ rng.normal(0, sigma, n_niv)


def perturbar(a, sigma=0.0, esc=0.0, redondeo=False):
    u, v = uv(a)
    if sigma:
        lnp = np.log(a[:, 0])
        u, v = u + ruido(len(u), lnp, sigma, esc), v + ruido(len(v), lnp, sigma, esc)
    b = a_polar(u, v, a[:, 0])
    if redondeo:
        b[:, 2] = np.round(b[:, 2] / NUDO) * NUDO
        b[:, 1] = (np.round(b[:, 1] / 10) * 10) % 360
    return b


def main():
    esc = [("limpio", 0.0, 0.0, False, None), ("redondeo antiguo (nudos, 10°)", 0.0, 0.0, True, None)]
    for sg in (1.0, 2.0, 3.0, 5.0):
        esc.append((f"σ = {sg:.0f} m/s, independiente", sg, 0.0, False, None))
        esc.append((f"σ = {sg:.0f} m/s, correlacionado en la vertical", sg, 0.15, False, None))
    esc.append(("σ = 2 m/s + redondeo", 2.0, 0.0, True, None))
    esc.append(("resolución antigua (posiciones fijas) + σ = 2 m/s", 2.0, 0.0, False, "na"))
    esc.append(("resolución antigua (adaptativa) + σ = 2 m/s", 2.0, 0.0, False, "ad"))
    filas = []
    for sid, nom in EST.items():
        perf = p19.leer(sid)
        pl80 = p19.plantillas(perf, 1989)
        for k, (ps, a, fu) in perf.items():
            if k.year < 2006 or len(a) < 4 or nivel(a, 850.0) is None or nivel(a, 700.0) is None:
                continue
            temp = k.year + 1 if k.month >= 10 else k.year
            for nombre, sg, es, rd, deg in esc:
                n = 1 if (sg == 0) else NSORT
                f, p = [], []
                for _ in range(n):
                    base = a
                    if deg:
                        if k.month not in pl80:
                            continue
                        t = pl80[k.month][rng.integers(0, len(pl80[k.month]))]
                        base = (p19.degradar_adaptativo if deg == "ad" else p19.degradar)(ps, a, t)
                        if len(base) < 3:
                            continue
                    b = perturbar(base, sg, es, rd)
                    f.append(fijo(perturbar(a, sg, es, rd)))
                    p.append(p19.jet(ps, b))
                if f:
                    filas.append({"sid": sid, "temp": temp, "esc": nombre, "fijo": np.nanmean(f), "perfil": np.nanmean(p) if np.isfinite(p).any() else np.nan})
    R = pd.DataFrame(filas)
    L = ["Ruido, redondeo y resolución en sondeos modernos (2006-2025, seis estaciones): frecuencia de jet (%) y diferencia respecto de la versión limpia;"
         " IC 95 % por bootstrap de temporadas"]
    limpio = R[R.esc == "limpio"].set_index(["sid", "temp"])
    rb = np.random.default_rng(310)
    for nombre, *_ in esc:
        q = R[R.esc == nombre].set_index(["sid", "temp"])
        j = q.join(limpio, rsuffix="_l", how="inner")
        fila = []
        for c, lab in (("fijo", "niveles fijos"), ("perfil", "perfil completo")):
            d = (j[c] - j[c + "_l"]).dropna()
            g = d.groupby(level="temp").agg(["sum", "count"])
            bs = []
            for _ in range(500):
                i = rb.integers(0, len(g), len(g))
                bs.append(g["sum"].values[i].sum() / g["count"].values[i].sum())
            m = d.mean()
            fila.append(f"{lab}: {100 * j[c].mean():5.2f} % (Δ {100 * m:+.2f} pp [{100 * np.percentile(bs, 2.5):+.2f},{100 * np.percentile(bs, 97.5):+.2f}]; "
                        f"sesgo de tendencia {100 * m / DECADAS:+.2f} pp/déc)")
        L.append(f"  {nombre:52s} " + " | ".join(fila))
    L.append("  Nota: el rms con ERA5 en días con jet (7.3 → 2.7 m/s) no equivale a un σ de ruido (incluye regresión a la media y representatividad); los valores"
             " de σ de 1-5 m/s son escenarios de sensibilidad, no cotas.")
    open("analisis/jet/31_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
