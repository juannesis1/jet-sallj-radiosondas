"""Paper 2, paso 10: tendencia del jet corregida por fuente de archivo.

Modelo de probabilidad lineal por estación (sondeos de 12 UTC, oct-mar, 1980-2025):
    jet_i = Σ_m μ_m [mes] + Σ_f α_f [fuente] + β · (año − 2000)/10 + ε_i
Sin y con los efectos de fuente α_f. β (pp/década) es la tendencia. Intervalos del 95 % por bootstrap de temporadas
completas (respeta la autocorrelación dentro de cada temporada). Para el perfil completo y para niveles fijos 850/700.
Además, el modelo agrupado de las estaciones del recorrido del jet con efectos fijos de estación.
Sensibilidad: el modelo agrupado desde 1993, después del escalón de ~1990-92 del viento a 850 hPa en las estaciones
argentinas (paso 13), que sesga hacia abajo la tendencia con niveles fijos desde 1980.
Control de resolución: el modelo agrupado del perfil completo con efectos fijos del número de niveles reportados en los
250 hPa inferiores (hasta 9), solo y junto con la fuente.
Salida: analisis/jet/10_resumen.txt y 10_tendencias.csv (para la figura 3).
"""
import os

import numpy as np
import pandas as pd

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande", "BRM00083928": "Uruguaiana", "BRM00083362": "Cuiabá",
       "BRM00083208": "Vilhena", "BRM00082824": "Porto Velho"}
rng = np.random.default_rng(5)


def datos(sid):
    d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
    d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3])].copy()
    d["temp"] = np.where(d.index.month >= 10, d.index.year + 1, d.index.year)
    d = d[(d.temp >= 1980) & (d.temp <= 2025)]
    w8, w7 = np.hypot(d.u850, d.v850), np.hypot(d.u700, d.v700)
    dr = (np.degrees(np.arctan2(-d.u850, -d.v850)) + 360) % 360
    d["fijos"] = ((w8 >= 12) & (w8 - w7 >= 6) & ((dr >= 292.5) | (dr <= 67.5))).where(w8.notna() & w7.notna())
    d["completo"] = d.jet_norte.astype(float)
    d["mes"] = d.index.month
    d["sid"] = sid
    return d


def beta(d, y, fuente, estacion=False):
    x = d.dropna(subset=[y])
    X = [pd.get_dummies(x.mes).values.astype(float), ((x.temp.values - 2000) / 10.0)[:, None]]
    if fuente:
        X.append(pd.get_dummies(x.fuente, drop_first=True).values.astype(float))
    if estacion:
        X.append(pd.get_dummies(x.sid, drop_first=True).values.astype(float))
    return 100 * np.linalg.lstsq(np.hstack(X), x[y].values.astype(float), rcond=None)[0][len(np.unique(x.mes))]


def boot(d, y, fuente, estacion=False, n=1000):
    r = np.random.default_rng(123)               # generador fijo: mismo estimando => mismo intervalo que en el paso 22
    temps = d.temp.unique()
    g = {t: d[d.temp == t] for t in temps}
    out = [beta(pd.concat([g[t] for t in r.choice(temps, len(temps))]), y, fuente, estacion) for _ in range(n)]
    return np.percentile(out, [2.5, 97.5])


def main():
    filas, L = [], ["Tendencia de la frecuencia de jet del norte (pp/década, oct-mar 12 UTC, 1980-2025), IC 95 %"]
    todos = []
    for sid, nom in EST.items():
        d = datos(sid)
        if d.fuente.nunique() < 1 or len(d) < 2000:
            continue
        todos.append(d)
        for y in ("completo", "fijos"):
            for fu in (False, True):
                b = beta(d, y, fu)
                lo, hi = boot(d, y, fu)
                filas.append({"estacion": nom, "criterio": y, "corrige_fuente": fu, "beta": b, "lo": lo, "hi": hi})
        r = pd.DataFrame(filas)
        q = r[r.estacion == nom].set_index(["criterio", "corrige_fuente"])
        L.append(f"{nom:13s} perfil completo: crudo {q.beta['completo', False]:+.2f} [{q.lo['completo', False]:+.2f},{q.hi['completo', False]:+.2f}]"
                 f" → corregido {q.beta['completo', True]:+.2f} [{q.lo['completo', True]:+.2f},{q.hi['completo', True]:+.2f}] | "
                 f"fijos: crudo {q.beta['fijos', False]:+.2f} [{q.lo['fijos', False]:+.2f},{q.hi['fijos', False]:+.2f}]"
                 f" → corregido {q.beta['fijos', True]:+.2f} [{q.lo['fijos', True]:+.2f},{q.hi['fijos', True]:+.2f}]")
    D = pd.concat(todos)
    for y in ("completo", "fijos"):
        for fu in (False, True):
            b = beta(D, y, fu, True)
            lo, hi = boot(D, y, fu, True, n=1000)
            filas.append({"estacion": "Agrupado", "criterio": y, "corrige_fuente": fu, "beta": b, "lo": lo, "hi": hi})
            L.append(f"AGRUPADO ({D.sid.nunique()} est.) {y:9s} {'corregido' if fu else 'crudo    '}: {b:+.2f} [{lo:+.2f},{hi:+.2f}]")
    for y in ("completo", "fijos"):
        for fu in (False, True):
            x = D[D.temp >= 1993]
            b, (lo, hi) = beta(x, y, fu, True), boot(x, y, fu, True, n=1000)
            L.append(f"AGRUPADO desde 1993 {y:9s} {'corregido' if fu else 'crudo    '}: {b:+.2f} [{lo:+.2f},{hi:+.2f}]")
    # sensibilidad: sin Porto Velho (registro inhomogéneo, sección 4.4)
    D9 = D[D.sid != "BRM00082824"]
    for y in ("completo", "fijos"):
        for fu in (False, True):
            b = beta(D9, y, fu, True)
            lo, hi = boot(D9, y, fu, True, n=1000)
            L.append(f"AGRUPADO sin Porto Velho ({D9.sid.nunique()} est.) {y:9s} {'corregido' if fu else 'crudo    '}: {b:+.2f} [{lo:+.2f},{hi:+.2f}]")
            filas.append({"estacion": "Agrupado sin Porto Velho", "criterio": y, "corrige_fuente": fu, "beta": b, "lo": lo, "hi": hi})
    D["nb"] = D.n_niveles.clip(upper=9).fillna(-1).astype(int).astype(str)
    for extra in (["nb"], ["fuente", "nb"]):
        x = D.dropna(subset=["completo"])
        X = [pd.get_dummies(x.mes).values.astype(float), ((x.temp.values - 2000) / 10.0)[:, None],
             pd.get_dummies(x.sid, drop_first=True).values.astype(float)]
        X += [pd.get_dummies(x[e], drop_first=True).values.astype(float) for e in extra]
        b = 100 * np.linalg.lstsq(np.hstack(X), x.completo.values.astype(float), rcond=None)[0][len(np.unique(x.mes))]
        L.append(f"AGRUPADO completo con efectos de {' + '.join(extra)}: {b:+.2f}")
    pd.DataFrame(filas).to_csv("analisis/jet/10_tendencias.csv", index=False)
    open("analisis/jet/10_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
