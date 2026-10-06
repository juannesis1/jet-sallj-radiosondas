"""Paper 2, paso 19: experimento de emulación. ¿Cuánto jet verían los sondeos modernos con la resolución vertical antigua?

Para cada sondeo moderno (12 UTC, oct-mar 2005-2025, casi todos con 925 hPa y 7-8 niveles en los 250 hPa inferiores)
se sortea un sondeo antiguo de la misma estación y el mismo mes (1980-1995) y se conserva del perfil moderno sólo el
viento en las presiones que reportó ese sondeo antiguo (interpolación lineal en log p del perfil moderno, sin extrapolar).
Se aplica el criterio de perfil completo al perfil degradado. Si la frecuencia del jet "degradada" se parece a la
frecuencia observada en 1980-1995, la diferencia entre épocas se explica por la resolución, no por el clima.
Se repite con 10 sorteos por sondeo. Además, la variante "sin 925 hPa" (perfil moderno quitando sólo ese nivel).
Segunda parte: cambio climático a igual resolución. Degradar también los sondeos antiguos los degradaría dos veces, así
que se comparan dos épocas: observado 1980-1989 (archivo con más niveles de la época antigua, antes del bache de los
noventa) contra 2005-2025 degradado con plantillas de 1980-1989 del mismo mes. Diferencia con IC 90 % por bootstrap de
temporadas, por estación y agrupada (media de estaciones). Sensibilidad: los vientos de 1980-1989 de las estaciones
argentinas eran ~0.5 m/s más fuertes respecto de ERA5 que después del escalón de ~1990-1992 a 850 hPa (paso 13); se
repite restando 0.5 m/s a todos los vientos de 1980-1989, y además sumando 1 m/s a los modernos (escalón de 925 hPa).
Revisión (ronda 2): (i) el emulador no es adaptativo, pero los observadores eligen los niveles significativos donde el
viento tiene rasgos; se agrega una variante 'adaptativa' (cota superior: se agregan los niveles reales del máximo y del
mínimo superior) y el resultado real queda entre ambas; (ii) validación contra la diferencia observada entre archivos
(USAF vs GTS) dentro de las mismas temporadas; (iii) línea de base sólo con la fuente dominante de los años 80;
(iv) composición de los archivos de baja resolución.
Salida: analisis/jet/19_resumen.txt
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande"}
CAPA_NUCLEO, CAPA_CAIDA, MIN_NUCLEO, MIN_CAIDA = 250.0, 300.0, 12.0, 6.0
rng = np.random.default_rng(19)


VMAX_PLAUSIBLE = 80.0   # m/s; mismo filtro que el paso 1
SUP = {}                # (sid, fecha) → ¿el sondeo reporta el nivel de superficie?


def leer(sid):
    """Perfiles de viento (p, dir, rapidez), 12 UTC oct-mar 1980-2025; superficie = nivel de superficie o, si falta,
    la mayor presión reportada (como el paso 1)."""
    out, cab, niv, psup, pmax, fuente = {}, None, [], np.nan, np.nan, ""
    with zipfile.ZipFile(f"data/igra_sa/{sid}-data.txt.zip") as z, z.open(z.namelist()[0]) as fh:
        for line in io.TextIOWrapper(fh, encoding="ascii", errors="ignore"):
            if line.startswith("#"):
                if cab is not None and niv:
                    out.setdefault(cab, (psup if np.isfinite(psup) else pmax, np.array(niv), fuente))
                    SUP.setdefault((sid, cab), bool(np.isfinite(psup)))
                y, m, d, h = int(line[13:17]), int(line[18:20]), int(line[21:23]), int(line[24:26])
                ok = 1980 <= y <= 2025 and h == 12 and m in (10, 11, 12, 1, 2, 3)
                cab = pd.Timestamp(y, m, d) if ok else None
                niv, psup, pmax, fuente = [], np.nan, np.nan, line[37:45].strip()
                continue
            if cab is None:
                continue
            p, wd, ws = line[9:15].strip(), line[40:45].strip(), line[46:51].strip()
            if p.startswith("-"):
                continue
            p = float(p) / 100
            pmax = np.nanmax([pmax, p])
            if line[1] == "1":
                psup = p
            if wd.startswith("-") or ws.startswith("-"):
                continue
            if float(ws) / 10 > VMAX_PLAUSIBLE:      # QC: viento físicamente implausible
                continue
            niv.append((p, float(wd), float(ws) / 10))
    if cab is not None and niv:
        out.setdefault(cab, (psup if np.isfinite(psup) else pmax, np.array(niv), fuente))
        SUP.setdefault((sid, cab), bool(np.isfinite(psup)))
    return out


def jet(psup, a, dp_min=0.0):
    """Criterio de perfil completo (como el paso 1) sobre a = [[p, dir, rapidez], ...]; dp_min > 0 exige que el núcleo esté al menos dp_min hPa
    por encima de la superficie (excluye máximos del nivel de superficie)."""
    if not np.isfinite(psup) or len(a) == 0:
        return np.nan
    dp = psup - a[:, 0]
    bajo = (dp >= dp_min) & (dp <= CAPA_NUCLEO)
    if (dp >= 0).sum() == 0 or ((dp >= 0) & (dp <= CAPA_NUCLEO)).sum() < 3 or bajo.sum() < 1:
        return np.nan
    i = np.where(bajo)[0][np.argmax(a[bajo, 2])]
    arriba = (dp > dp[i]) & (dp <= CAPA_CAIDA)
    if not arriba.any():
        return np.nan
    d = a[i, 1]
    return float(a[i, 2] >= MIN_NUCLEO and a[i, 2] - a[arriba, 2].min() >= MIN_CAIDA and (d >= 292.5 or d <= 67.5))


def degradar(psup, a, plantilla):
    """Interpola el perfil moderno a los niveles relativos (psup − p) del sondeo antiguo."""
    rel_viejo = plantilla
    rel = psup - a[:, 0]
    o = np.argsort(rel)
    rel, a = rel[o], a[o]
    sel = rel_viejo[(rel_viejo >= rel.min()) & (rel_viejo <= rel.max())]
    if len(sel) == 0:
        return np.empty((0, 3))
    u, v = -a[:, 2] * np.sin(np.radians(a[:, 1])), -a[:, 2] * np.cos(np.radians(a[:, 1]))
    lp = np.log(psup - rel)
    lps = np.log(psup - sel)
    ui, vi = np.interp(-lps, -lp, u), np.interp(-lps, -lp, v)  # np.interp necesita abscisas crecientes
    w = np.hypot(ui, vi)
    di = (np.degrees(np.arctan2(-ui, -vi)) + 360) % 360
    return np.column_stack([psup - sel, di, w])


def degradar_adaptativo(psup, a, plantilla):
    """Cota superior de la adaptatividad de los niveles significativos: al perfil degradado se le agregan los niveles
    reales del máximo de viento en 0-250 hPa sobre la superficie y del mínimo superior hasta 300 hPa (los extremos que un
    observador elige como niveles significativos). El resultado real de un archivo antiguo está entre degradar() y esto."""
    b = degradar(psup, a, plantilla)
    dp = psup - a[:, 0]
    bajo = (dp >= 0) & (dp <= CAPA_NUCLEO)
    if bajo.sum() < 1:
        return b
    i = np.where(bajo)[0][np.argmax(a[bajo, 2])]
    arriba = (dp > dp[i]) & (dp <= CAPA_CAIDA)
    extra = [a[i]]
    if arriba.any():
        extra.append(a[np.where(arriba)[0][np.argmin(a[arriba, 2])]])
    return np.vstack([b] + [np.array([x[0], x[1], x[2]]) for x in extra]) if len(b) else b


def plantillas(perf, anio_max, fuente=None):
    """Niveles relativos a la superficie (psup − p, hPa) de los sondeos antiguos, por mes calendario."""
    pl = {}
    for k, (ps, a, fu) in perf.items():
        if k.year <= anio_max and np.isfinite(ps) and len(a) and (fuente is None or fu == fuente):
            rel = ps - a[:, 0]
            rel = rel[(rel >= 0) & (rel <= CAPA_CAIDA + 20)]
            if len(rel) >= 2:
                pl.setdefault(k.month, []).append(rel)
    return pl


def emular(ps, a, pl, k, adaptativo=False, n=20, dv=0.0):
    """Media de n sorteos de plantillas antiguas del mismo mes; dv se resta a la rapidez (sensibilidad)."""
    if k.month not in pl:
        return np.nan
    if dv:
        a = a.copy()
        a[:, 2] = np.maximum(a[:, 2] - dv, 0)
    f = degradar_adaptativo if adaptativo else degradar
    r = [jet(ps, f(ps, a, pl[k.month][t])) for t in rng.integers(0, len(pl[k.month]), n)]
    r = [x for x in r if np.isfinite(x)]
    return float(np.mean(r)) if r else np.nan


def diferencia(g80, gmod):
    dif = 100 * (np.concatenate(gmod).mean() - np.concatenate(g80).mean())
    bs = [100 * (np.concatenate([gmod[i] for i in rng.integers(0, len(gmod), len(gmod))]).mean()
                 - np.concatenate([g80[i] for i in rng.integers(0, len(g80), len(g80))]).mean()) for _ in range(1000)]
    return dif, np.array(bs)


def main():
    L = ["Emulación de la resolución vertical (criterio de perfil completo, 12 UTC oct-mar)",
         "Variantes del emulador: 'no adaptativo' = niveles de un sondeo antiguo del mismo mes (cota inferior de lo que habría "
         "detectado el archivo antiguo); 'adaptativo' = además los niveles reales del máximo y del mínimo superior (cota superior)."]
    CSV1, CSV2, CSV3 = [], [], []
    A, B, C = [], [], []   # diferencias bootstrap: no adaptativo, adaptativo, no adaptativo con sensibilidad de vientos
    L2 = []
    for sid, nom in EST.items():
        perf = leer(sid)
        f_viejo = np.nanmean([jet(ps, a) for k, (ps, a, fu) in perf.items() if k.year <= 1995])
        pl95 = plantillas(perf, 1995)
        filas = []
        for k, (ps, a, fu) in perf.items():
            if k.year >= 2005:
                filas.append({"j": jet(ps, a), "na": emular(ps, a, pl95, k), "ad": emular(ps, a, pl95, k, True)})
        d = pd.DataFrame(filas).dropna(subset=["j"])
        CSV1.append({"estacion": nom, "obs_80_95": 100 * f_viejo, "moderno": 100 * d.j.mean(), "no_adaptativo": 100 * d.na.mean(),
                     "adaptativo": 100 * d.ad.mean()})
        L.append(f"\n=== {nom}\n  1980-1995 observado {100 * f_viejo:5.1f} %  | 2005-2025: moderno {100 * d.j.mean():5.1f}  "
                 f"no adaptativo {100 * d.na.mean():5.1f}  adaptativo {100 * d.ad.mean():5.1f}  "
                 f"→ fracción del aumento explicada por la resolución: {100 * (d.j.mean() - d.na.mean()) / (d.j.mean() - f_viejo):.0f} % "
                 f"(no adaptativo), {100 * (d.j.mean() - d.ad.mean()) / (d.j.mean() - f_viejo):.0f} % (adaptativo)")
        # cambio a igual resolución: 1980-1989 observado vs 2005-2025 emulado con plantillas de 1980-1989
        pl80 = plantillas(perf, 1989)
        filas = []
        for k, (ps, a, fu) in perf.items():
            temp = k.year + 1 if k.month >= 10 else k.year
            if temp <= 1989:
                b = a.copy()
                b[:, 2] = np.maximum(b[:, 2] - 0.5, 0)
                filas.append({"temp": temp, "ep": "80", "fu": fu, "na": jet(ps, a), "ad": jet(ps, a), "sens": jet(ps, b)})
            elif temp >= 2005 and k.month in pl80:
                filas.append({"temp": temp, "ep": "mod", "fu": fu, "na": emular(ps, a, pl80, k),
                              "ad": emular(ps, a, pl80, k, True), "sens": emular(ps, a, pl80, k, dv=1.0)})
        d = pd.DataFrame(filas, columns=["temp", "ep", "fu", "na", "ad", "sens"])
        if (d.ep == "80").sum() < 300:
            L2.append(f"  {nom:13s} sin datos suficientes en 1980-1989")
            continue
        dd = d.dropna(subset=["na"])
        fuentes = dd[dd.ep == "80"].fu.value_counts()
        res = {}
        for v in ("na", "ad", "sens"):
            x = d.dropna(subset=[v])
            g80 = [q[v].values for _, q in x[x.ep == "80"].groupby("temp")]
            gm = [q[v].values for _, q in x[x.ep == "mod"].groupby("temp")]
            res[v] = diferencia(g80, gm)
        A.append(res["na"][1]); B.append(res["ad"][1]); C.append(res["sens"][1])
        CSV2.append({"estacion": nom, "dif_na": res["na"][0], "na_lo": np.percentile(res["na"][1], 5), "na_hi": np.percentile(res["na"][1], 95),
                     "dif_ad": res["ad"][0], "ad_lo": np.percentile(res["ad"][1], 5), "ad_hi": np.percentile(res["ad"][1], 95)})
        # sólo con la fuente dominante de los 80 (para que la línea de base no mezcle archivos)
        dom = fuentes.index[0]
        x = dd[(dd.ep == "mod") | (dd.fu == dom)]
        g80 = [q.na.values for _, q in x[x.ep == "80"].groupby("temp")]
        gm = [q.na.values for _, q in x[x.ep == "mod"].groupby("temp")]
        dsol = diferencia(g80, gm)[0]
        L2.append(f"  {nom:13s} 1980-89 obs {100 * d[d.ep == '80'].na.mean():5.1f} (fuentes {dict(fuentes.head(3))}) | 2005-25 emulado: "
                  f"no adaptativo {100 * d[d.ep == 'mod'].na.mean():5.1f}, adaptativo {100 * d[d.ep == 'mod'].ad.mean():5.1f}\n"
                  f"      diferencia no adaptativo {res['na'][0]:+.1f} [{np.percentile(res['na'][1], 5):+.1f},{np.percentile(res['na'][1], 95):+.1f}]"
                  f"  adaptativo {res['ad'][0]:+.1f} [{np.percentile(res['ad'][1], 5):+.1f},{np.percentile(res['ad'][1], 95):+.1f}]"
                  f"  con sensibilidad de vientos {res['sens'][0]:+.1f}  sólo fuente {dom} {dsol:+.1f}")
    L += ["\n--- Cambio a igual resolución: 1980-1989 observado vs 2005-2025 emulado con niveles de 1980-1989 ---"] + L2
    for nombre, X in (("no adaptativo", A), ("adaptativo (cota superior)", B), ("no adaptativo + sensibilidad de vientos", C)):
        m = np.mean(X, axis=0)
        L.append(f"  Agrupado ({len(X)} estaciones), {nombre}: {np.mean(m):+.1f} pp [{np.percentile(m, 5):+.1f},{np.percentile(m, 95):+.1f}]")

    # ---- validación con fuentes reales: dentro de las mismas temporadas, USAF (5-8 niveles) vs GTS
    L.append("\n--- Validación: ¿el emulador reproduce la diferencia observada entre archivos dentro de la misma temporada? ---")
    L.append("    diferencia (USAF − GTS), pp: observada | predicha por emulación con plantillas USAF aplicadas a los sondeos GTS "
             "(no adaptativo / adaptativo)")
    tot = {"obs": [], "na": [], "ad": [], "n": []}
    for sid, nom in EST.items():
        perf = leer(sid)
        plu = plantillas(perf, 2100, "usaf-ds3")
        filas = []
        for k, (ps, a, fu) in perf.items():
            temp = k.year + 1 if k.month >= 10 else k.year
            if fu in ("usaf-ds3", "ncdc-gts"):
                f = {"temp": temp, "fu": fu, "j": jet(ps, a), "na": np.nan, "ad": np.nan}
                if fu == "ncdc-gts":
                    f["na"], f["ad"] = emular(ps, a, plu, k, n=10), emular(ps, a, plu, k, True, n=10)
                filas.append(f)
        d = pd.DataFrame(filas)
        o, na, ad = [], [], []
        for t, q in d.groupby("temp"):
            u, g = q[q.fu == "usaf-ds3"], q[q.fu == "ncdc-gts"]
            if u.j.notna().sum() >= 20 and g.j.notna().sum() >= 20:
                tot["n"].append((int(u.j.notna().sum()), int(g.j.notna().sum())))
                o.append(100 * (u.j.mean() - g.j.mean()))
                na.append(100 * (g.na.mean() - g.j.mean()))
                ad.append(100 * (g.ad.mean() - g.j.mean()))
        if len(o) >= 2:
            ci_ = lambda v: (np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(500)], 5), np.percentile([np.mean(rng.choice(v, len(v))) for _ in range(500)], 95)) if len(v) >= 3 else (np.nan, np.nan)  # noqa: E731
            CSV3.append({"estacion": nom, "obs": np.mean(o), "na": np.mean(na), "ad": np.mean(ad), "n": len(o),
                         "obs_lo": ci_(o)[0], "obs_hi": ci_(o)[1], "na_lo": ci_(na)[0], "na_hi": ci_(na)[1], "ad_lo": ci_(ad)[0], "ad_hi": ci_(ad)[1]})
            L.append(f"  {nom:13s} {len(o):2d} temporadas: observada {np.mean(o):+5.1f} | predicha {np.mean(na):+5.1f} / {np.mean(ad):+5.1f}")
            tot["obs"] += o; tot["na"] += na; tot["ad"] += ad
    if tot["obs"]:
        arr = {k: np.array(tot[k]) for k in ("obs", "na", "ad")}
        ic = {}
        for k, v in arr.items():
            bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            ic[k] = (np.percentile(bs, 5), np.percentile(bs, 95))
        n = np.array(tot["n"])
        L.append(f"  TODAS {len(tot['obs'])} estaciones-temporadas: observada {np.mean(tot['obs']):+.1f} [{ic['obs'][0]:+.1f},{ic['obs'][1]:+.1f}] | predicha "
                 f"{np.mean(tot['na']):+.1f} [{ic['na'][0]:+.1f},{ic['na'][1]:+.1f}] / {np.mean(tot['ad']):+.1f} [{ic['ad'][0]:+.1f},{ic['ad'][1]:+.1f}]"
                 f" (IC 90 % por bootstrap de estaciones-temporada; sondeos por celda USAF/GTS: mediana {int(np.median(n[:, 0]))}/{int(np.median(n[:, 1]))},"
                 f" mínimo {n[:, 0].min()}/{n[:, 1].min()})")

    # ---- composición de los archivos de baja resolución (¿por qué NCDC 6316 da más jets?)
    L.append("\n--- Archivos de baja resolución: nº de sondeos, niveles en 0-250 hPa, % con 850 y 700 hPa, jets (perfil completo, %) ---")
    for sid, nom in (("ARM00087344", "Córdoba"), ("ARM00087623", "Santa Rosa"), ("ARM00087155", "Resistencia")):
        perf = leer(sid)
        filas = []
        for k, (ps, a, fu) in perf.items():
            rel = ps - a[:, 0]
            filas.append({"fu": fu, "anio": k.year, "niv": ((rel >= 0) & (rel <= 250)).sum(), "j": jet(ps, a),
                          "p850": np.isclose(a[:, 0], 850).any() and np.isclose(a[:, 0], 700).any()})
        d = pd.DataFrame(filas)
        t = d.groupby("fu").agg(n=("j", "size"), años=("anio", lambda x: f"{x.min()}-{x.max()}"), niveles=("niv", "median"),
                                pct_850_700=("p850", lambda x: 100 * x.mean()), jet=("j", lambda x: 100 * x.mean()))
        L.append(f"  {nom}\n" + t[t.n >= 100].round(1).to_string())
    pd.DataFrame(CSV1).to_csv("analisis/jet/19_emulacion.csv", index=False)
    pd.DataFrame(CSV2).to_csv("analisis/jet/19_cambio.csv", index=False)
    pd.DataFrame(CSV3).to_csv("analisis/jet/19_validacion.csv", index=False)
    open("analisis/jet/19_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
