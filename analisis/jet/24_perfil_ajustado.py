"""Paper 2, paso 24: ¿cuánto del aumento del perfil completo se debe a los saltos del viento de los sondeos?

La validación del paso 19 muestra que, en las mismas temporadas, el archivo USAF (5-8 niveles) y el GTS (7-8 niveles) dan
frecuencias de jet de perfil completo casi iguales; la resolución de los archivos de los 80 no alcanza para explicar el aumento hasta
la era GTS. El paso 22 identifica saltos del viento del sondeo respecto de ERA5 y de una referencia sin quiebres a 925, 850 y 700 hPa.
Acá se ajusta el perfil completo: a cada nivel se le resta el ajuste del tramo correspondiente (interpolado en log p entre 925, 850 y 700
hPa, constante fuera de ese rango) y se vuelve a aplicar el criterio de perfil completo. Se compara la tendencia agrupada (10 estaciones)
con y sin ajuste, con efectos de mes y estación, con efectos de fuente y con el número de niveles, para 1980-2025 y 1993-2025.
Salida: analisis/jet/24_resumen.txt y 24_perfil.parquet
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
p19 = importlib.import_module("19_emulacion_resolucion")
p22 = importlib.import_module("22_homogeneizar_fijos")
EST = p22.EST
rng = np.random.default_rng(24)
NIVELES = np.log([925.0, 850.0, 700.0])


def ajuste_perfil(ajustes, sid, periodo, p):
    """Ajuste (m/s a restar) en cada nivel p del perfil, interpolado en log p entre los tres niveles ajustados."""
    v = []
    for niv in (925, 850, 700):
        a = ajustes.get((sid, niv))
        v.append(float(a.get(periodo, 0.0)) if a is not None else 0.0)
    # np.interp requiere abscisas crecientes: log p decrece de 925 a 700
    return np.interp(np.log(p), NIVELES[::-1], v[::-1])


def beta(d, y, extra):
    x = d.dropna(subset=[y])
    X = [pd.get_dummies(x.mes).values.astype(float), ((x.temp.values - 2000) / 10.0)[:, None],
         pd.get_dummies(x.sid, drop_first=True).values.astype(float)]
    for e in extra:
        X.append(pd.get_dummies(x[e], drop_first=True).values.astype(float))
    return 100 * np.linalg.lstsq(np.hstack(X), x[y].values.astype(float), rcond=None)[0][len(np.unique(x.mes))]


def boot(d, y, extra, n=1000):
    r = np.random.default_rng(123)               # generador fijo
    temps = d.temp.unique()
    g = {t: d[d.temp == t] for t in temps}
    out = [beta(pd.concat([g[t] for t in r.choice(temps, len(temps))]), y, extra) for _ in range(n)]
    return np.percentile(out, [2.5, 97.5])


MANDATORIOS = np.array([1000.0, 925.0, 850.0, 700.0, 500.0])


def horas(sid):
    """Hora de lanzamiento (h UTC) de cada sondeo de 12 UTC oct-mar, del encabezado de IGRA (RELTIME); nan si falta."""
    import io
    import zipfile
    out = {}
    with zipfile.ZipFile(f"data/igra_sa/{sid}-data.txt.zip") as z, z.open(z.namelist()[0]) as fh:
        for line in io.TextIOWrapper(fh, encoding="ascii", errors="ignore"):
            if line.startswith("#"):
                y, m, d, h = int(line[13:17]), int(line[18:20]), int(line[21:23]), int(line[24:26])
                if y >= 1980 and h == 12 and m in (10, 11, 12, 1, 2, 3):
                    r = line[27:31].strip()
                    out[pd.Timestamp(y, m, d)] = (int(r[:2]) + int(r[2:]) / 60) if r.isdigit() and int(r) < 2400 else np.nan
    return out


def main():
    ajustes = pd.read_pickle("analisis/jet/22_ajustes.pkl")
    rango = pd.period_range("1979-01", "2026-12", freq="M")   # meses sin pares suficientes toman el ajuste del tramo vecino
    ajustes = {k: v.reindex(rango).ffill().bfill() for k, v in ajustes.items()}
    filas = []
    PERF = {}
    for sid in EST:
        perf = p19.leer(sid)
        PERF[sid] = perf
        hr = horas(sid)
        for k, (ps, a, fu) in perf.items():
            temp = k.year + 1 if k.month >= 10 else k.year
            if temp < 1980 or temp > 2025 or len(a) == 0:
                continue
            per = pd.Period(k, "M")
            b = a.copy()
            b[:, 2] = np.maximum(a[:, 2] - ajuste_perfil(ajustes, sid, per, a[:, 0]), 0)
            rel = ps - a[:, 0]
            en = (rel >= 0) & (rel <= 250)
            nm = int(np.isin(np.round(a[en, 0]), MANDATORIOS).sum())
            filas.append({"sid": sid, "fecha": k, "temp": temp, "mes": k.month, "fuente": fu, "nb": str(min(int(en.sum()), 9)),
                          "nm": str(min(nm, 5)), "hora": hr.get(k, np.nan), "crudo": p19.jet(ps, a), "ajustado": p19.jet(ps, b)})
    D = pd.DataFrame(filas)
    D["tardio"] = (D.hora >= 11.5).astype(float).where(D.hora.notna())
    D.to_parquet("analisis/jet/24_perfil.parquet")
    L = ["Perfil completo: jets crudos y con el viento ajustado por los saltos respecto de ERA5/referencia (paso 22)",
         f"  frecuencia media (%), 1980-2025: crudo {100 * D.crudo.mean():.1f}, ajustado {100 * D.ajustado.mean():.1f}",
         "  Control del número de niveles: 'nm' = niveles MANDATORIOS (1000, 925, 850, 700, 500 hPa) reportados en los 250 hPa inferiores, fijos"
         " por reglamento pero NO exógenos al viento (reportar 925 hPa depende del viento de 850 hPa, paso 33): no son un control válido; 'nb' = todos los niveles reportados (incluye los significativos, que se insertan donde el"
         " perfil tiene rasgos y por eso es endógeno al jet: se muestra sólo para comparar)."]
    L.append("\nTendencia agrupada (10 estaciones), pp/década [IC 95 % por bootstrap de temporadas, 1000 réplicas]")
    casc = []
    controles = ((([], "sólo mes y estación"), (["fuente"], "+ fuente"), (["fuente", "nm"], "+ fuente + niveles mandatorios"),
                  (["fuente", "nb"], "+ fuente + nº de niveles (endógeno)")))
    for ini in (1980, 1993):
        x = D[D.temp >= ini]
        for y in ("crudo", "ajustado"):
            for extra, lab in controles:
                b = beta(x, y, extra)
                lo, hi = boot(x, y, extra)
                L.append(f"  desde {ini} {y:9s} {lab:36s} {b:+.2f} [{lo:+.2f},{hi:+.2f}]  ({100 * b / (100 * x[y].mean()):+.0f} %/déc)")
                casc.append({"desde": ini, "viento": y, "control": lab, "beta": b, "lo": lo, "hi": hi})
    # muestras restringidas: ¿queda el aumento cuando se comparan archivos de resolución comparable?
    L.append("\nMuestras restringidas (1980-2025), pp/década [IC 95 %]:")
    for nom, x in (("sólo sondeos con ≥ 5 niveles en 0-250 hPa", D[D.nb.astype(int) >= 5]),
                   ("sólo archivos USAF + GTS (excluye NCDC 6314/6316/6322)", D[D.fuente.isin(["usaf-ds3", "ncdc-gts"])])):
        for y in ("crudo", "ajustado"):
            for extra, lab in (([], "sólo mes y estación"), (["fuente"], "+ fuente")):
                b = beta(x, y, extra)
                lo, hi = boot(x, y, extra)
                L.append(f"  {nom:55s} {y:9s} {lab:20s} {b:+.2f} [{lo:+.2f},{hi:+.2f}]  (n {len(x)}, {100 * b / (100 * x[y].mean()):+.0f} %/déc)")
                casc.append({"desde": 1980, "viento": y, "control": nom + " " + lab, "beta": b, "lo": lo, "hi": hi})
    # incertidumbre del ajuste propagada al perfil completo: en cada réplica cada corte se desplaza U(-12, +12) meses y el ajuste de cada tramo
    # recibe un error N(0, se); se recalcula el criterio de perfil completo y la tendencia (+ fuente). Se combina en cuadratura con el bootstrap.
    seg = pd.read_csv("analisis/jet/22_ajustes.csv")
    rp = np.random.default_rng(2400)
    NREP = 60
    ests = {1980: [], 1993: []}
    for r_ in range(NREP):
        aj = {}
        for (sid, niv), g in seg.groupby(["sid", "nivel"]):
            g = g.sort_values("desde")
            ini = [pd.Period(x, "M") for x in g.desde]
            fin = pd.Period(g.hasta.iloc[-1], "M")
            cortes = [ini[i] + int(rp.integers(-12, 13)) for i in range(1, len(ini))]
            lim = [ini[0]] + sorted(cortes) + [fin + 1]
            v = pd.Series(0.0, index=rango)
            for i in range(len(g)):
                val = g.ajuste_m_s.iloc[i] + rp.normal(0, g.se_m_s.iloc[i])
                v[(v.index >= lim[i]) & (v.index < lim[i + 1])] = val
            v[v.index < lim[0]] = v[lim[0]] if lim[0] in v.index else 0.0
            aj[(sid, niv)] = v
        pert = []
        for sid in EST:
            for k, (ps, a, fu) in PERF[sid].items():
                temp = k.year + 1 if k.month >= 10 else k.year
                if temp < 1980 or temp > 2025 or len(a) == 0:
                    continue
                b = a.copy()
                b[:, 2] = np.maximum(a[:, 2] - ajuste_perfil(aj, sid, pd.Period(k, "M"), a[:, 0]), 0)
                pert.append(p19.jet(ps, b))
        Dp = D.copy()
        Dp["ajustado"] = pert
        for ini_ in ests:
            ests[ini_].append(beta(Dp[Dp.temp >= ini_], "ajustado", ["fuente"]))
    L.append(f"\nIncertidumbre del ajuste propagada al perfil completo ({NREP} réplicas; cortes ±12 meses, error de la media de cada tramo), + fuente:")
    for ini_, v in ests.items():
        row = next(c for c in casc if c["desde"] == ini_ and c["viento"] == "ajustado" and c["control"] == "+ fuente")
        sd = float(np.std(v, ddof=1))
        lo_t = row["beta"] - np.hypot(row["beta"] - row["lo"], 1.96 * sd)
        hi_t = row["beta"] + np.hypot(row["hi"] - row["beta"], 1.96 * sd)
        L.append(f"  desde {ini_}: DE por perturbación {sd:.2f} pp/déc (rango {min(v):+.2f} a {max(v):+.2f}); IC 95 % total [{lo_t:+.2f},{hi_t:+.2f}]")
        casc.append({"desde": ini_, "viento": "ajustado", "control": "+ fuente, incertidumbre del ajuste", "beta": row["beta"], "lo": lo_t, "hi": hi_t})
    pd.DataFrame(casc).to_csv("analisis/jet/24_cascada.csv", index=False)
    # dejar una estación afuera
    L.append("\nDejando una estación afuera (1980-2025, + fuente, viento ajustado), pp/década [IC 95 %, 200 réplicas]:")
    rl = []
    for sid, nom in EST.items():
        x = D[D.sid != sid]
        b = beta(x, "ajustado", ["fuente"])
        lo, hi = boot(x, "ajustado", ["fuente"], n=200)
        rl.append(b)
        L.append(f"  sin {nom:13s} {b:+.2f} [{lo:+.2f},{hi:+.2f}]")
    L.append(f"  rango de las estimaciones puntuales: {min(rl):+.2f} a {max(rl):+.2f}")
    # hora de lanzamiento (disponible desde 2000)
    x = D[(D.temp >= 2001) & D.tardio.notna()]
    L.append(f"\nControl de la hora de lanzamiento (2001-2025, {len(x)} sondeos con hora): tendencia agrupada del perfil completo ajustado + fuente")
    for extra, lab in ((["fuente"], "sin control de hora"), (["fuente", "tardio"], "con control de lanzamiento tardío (≥ 11:30 UTC)")):
        b = beta(x, "ajustado", extra)
        lo, hi = boot(x, "ajustado", extra, n=200)
        L.append(f"  {lab:50s} {b:+.2f} [{lo:+.2f},{hi:+.2f}]")
    L.append("\nPor estación (1980-2025, sólo mes): crudo → ajustado, pp/década")
    for sid, nom in EST.items():
        x = D[D.sid == sid]
        L.append(f"  {nom:13s} {beta(x, 'crudo', []):+.2f} → {beta(x, 'ajustado', []):+.2f}   frecuencia {100 * x.crudo.mean():.1f} → {100 * x.ajustado.mean():.1f} %")
    x = D[(D.temp >= 2006) & D.fuente.isin(["ncdc-gts"])]
    lo, hi = boot(x, "crudo", [])
    L.append(f"\nEra GTS de alta resolución (2006-2025, fuente ncdc-gts): crudo {beta(x, 'crudo', []):+.2f} [{lo:+.2f},{hi:+.2f}], ajustado {beta(x, 'ajustado', []):+.2f}")
    open("analisis/jet/24_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
