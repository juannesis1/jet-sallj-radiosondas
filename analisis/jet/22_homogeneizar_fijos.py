"""Paper 2, paso 22: homogeneidad del índice con niveles fijos (850/700 hPa) y de los vientos de 925/850/700 hPa, con ERA5 como referencia.

  (a) Para cada estación y nivel (925, 850, 700): serie mensual de la diferencia sondeo − ERA5 (rapidez, mismos días, 12 UTC, oct-mar),
      anomalía respecto de su media mensual.
  (b) Segmentación binaria con SNHT; significancia por permutación de bloques de 3 años (preserva la persistencia interanual);
      la familia de pruebas son TODOS los nodos probados de las 30 series (incluidos los que no pasan), con Benjamini-Hochberg (q = 0.10,
      tope p ≤ 0.05); se descartan los cortes con estabilidad < 80 % (5 semillas, 300 permutaciones). Mínimo de 18 meses por tramo.
  (c) Cada corte se compara con el cambio medio de sondeo − ERA5 en las estaciones del otro país (±6 años); si se mueve en el mismo
      sentido al menos la mitad, es regional y no se ajusta. Los tramos se llevan al nivel del último (aditivo; la incertidumbre del
      ajuste se perturba con el error de la media de cada tramo, no incluye el error de la fecha).
  (d) Tendencia de jets con niveles fijos (modelo de probabilidad lineal, efectos de mes, estación y fuente; bootstrap de temporadas)
      para 1980-2025, 1993-2025 y 2000-2025; fracción de sondeos con 850 y 700 reportados y frecuencia de jet fijo en ERA5 con y sin esos niveles.
Salida: analisis/jet/22_resumen.txt, 22_quiebres.csv, 22_ajustes.csv y 22_fijos.parquet
"""
import glob
import json
import os

import numpy as np
import pandas as pd

EST = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087623": "Santa Rosa", "ARM00087576": "Ezeiza",
       "BRM00083827": "Foz", "BRM00083612": "Campo Grande", "BRM00083928": "Uruguaiana", "BRM00083362": "Cuiabá",
       "BRM00083208": "Vilhena", "BRM00082824": "Porto Velho"}


PERMUTACIONES = 1000
SEMILLA = 22


def snht(x):
    z = (x - x.mean()) / x.std()
    n = len(z)
    S = np.cumsum(z)
    k = np.arange(12, n - 12)
    Sk = S[k - 1]
    t = Sk ** 2 / k + (S[-1] - Sk) ** 2 / (n - k)
    return t.max(), int(np.argmax(t)) + 12


BLOQUE = 3   # años consecutivos que se permutan juntos (preserva la persistencia interanual)


def pval(x, años, n=None, semilla=None):
    """p por permutación de bloques de BLOQUE años consecutivos de la T del SNHT (la persistencia interanual de la serie se conserva dentro
    de cada bloque); generador local y determinista para cada serie."""
    n = PERMUTACIONES if n is None else n
    t, _ = snht(x)
    ua = np.unique(años)
    g = [x[(años >= ua[i]) & (años <= ua[min(i + BLOQUE - 1, len(ua) - 1)])] for i in range(0, len(ua), BLOQUE)]
    r = np.random.default_rng([SEMILLA if semilla is None else semilla, len(x), int(abs(x[:5]).sum() * 1e4) % 100003])
    nulo = np.array([snht(np.concatenate([g[i] for i in r.permutation(len(g))]))[0] for _ in range(n)])
    return t, float((1 + np.sum(nulo >= t)) / (1 + n))


def segmentar(d, minimo=18, n=None, semilla=None):
    """SNHT binario sobre una serie mensual (Series indexada por Period). Devuelve (cortes, probados): cortes = [(posición, p, nodo, padre)] con
    p < 0.05 (nodos de la recursión) y probados = lista de los p de TODOS los nodos probados (la familia para el control de la tasa de falsos
    descubrimientos, incluidos los que no pasan)."""
    nodos = []

    def rec(a, b, padre):
        x = d.values[a:b]
        if len(x) < 2 * minimo or x.std() == 0:
            return
        años = d.index[a:b].year.values
        t, p = pval(x, años, n, semilla)
        k = snht(x)[1]
        if k < minimo or len(x) - k < minimo:
            return
        idx = len(nodos)
        nodos.append((a + k, p, idx, padre))
        if p >= 0.05:
            return
        rec(a, a + k, idx)
        rec(a + k, b, idx)

    rec(0, len(d), -1)
    return nodos


def departures(e):
    """Diferencia mensual sondeo − ERA5 (rapidez) por estación y nivel, 12 UTC oct-mar."""
    out = {}
    for sid in EST:
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[(s.index.hour == 12) & s.index.month.isin([10, 11, 12, 1, 2, 3])].copy()
        s.index = s.index.normalize()
        s = s[~s.index.duplicated()]
        r = e.loc[sid].reindex(s.index)
        for niv in (925, 850, 700):
            d = np.hypot(s[f"u{niv}"], s[f"v{niv}"]) - r[float(niv)]
            g = d.groupby(d.index.to_period("M"))
            m = g.mean()[g.count() >= 10]
            out[(sid, niv)] = m - m.groupby(m.index.month).transform("mean")
    return out


def main():
    e = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob("data/era5_perfiles/*.parquet")))
    e["fecha"] = pd.to_datetime(e.fecha)
    e = e[(e.fecha.dt.hour == 12) & e.nivel.isin([925.0, 850.0, 700.0])]
    e["w"] = np.hypot(e.u, e.v)
    e["fecha"] = e.fecha.dt.normalize()
    e = e.pivot_table(index=["sid", "fecha"], columns="nivel", values="w")
    D = departures(e)
    L = ["Homogeneización de los vientos de 925, 850 y 700 hPa con ERA5 (sondeo − ERA5, anomalías mensuales)"]
    ajustes, filas_aj, quiebres = {}, [], []
    ARG = [i for i in EST if i.startswith("ARM")]
    BRA = [i for i in EST if i.startswith("BRM")]
    L.append("Quiebres de la serie sondeo − ERA5 de cada estación (sin referencia: una referencia de estaciones sin quiebres sólo existe desde ~1996"
             " y vuelve el método ciego a los años 80). Cada quiebre se contrasta con el cambio medio del sondeo − ERA5 en las estaciones del OTRO país"
             " (Brasil para las argentinas y viceversa) en los seis años anteriores y posteriores: si ese control se mueve en el mismo sentido y"
             " al menos la mitad, el quiebre se considera regional (deriva de ERA5 o clima) y no se ajusta.")

    def cambio(z, fecha):
        a = z[(z.index >= fecha - 72) & (z.index < fecha)]
        d_ = z[(z.index >= fecha) & (z.index < fecha + 72)]
        return (d_.mean() - a.mean()) if (len(a) >= 12 and len(d_) >= 12) else np.nan

    # fase 1: todos los nodos probados de todas las series (familia completa), BH (q = 0.10, techo 0.05) y estabilidad con otras semillas
    NODOS, OTROS, ACEPTADOS = {}, {}, {}
    for niv in (925, 850, 700):
        for sid in EST:
            z = D[(sid, niv)].dropna()
            if len(z) < 72:
                continue
            NODOS[(sid, niv)] = segmentar(z)
            OTROS[(sid, niv)] = [[k for k, *_ in segmentar(z, n=300, semilla=100 + sd) if _[0] < 0.05] for sd in range(5)]
    familia = np.array([p for nod in NODOS.values() for _, p, _, _ in nod])
    o = np.argsort(familia)
    ok = familia[o] <= 0.10 * (np.arange(1, len(familia) + 1) / len(familia))
    pstar = min(0.05, familia[o][np.max(np.where(ok)[0])] if ok.any() else 0.0)
    for clave, nod in NODOS.items():
        acept = set()
        for k, p, i, pa in nod:
            if p <= pstar and (pa == -1 or pa in acept):
                acept.add(i)
        ACEPTADOS[clave] = acept
    n_cand = int((familia < 0.05).sum())
    L.append(f"  Familia de {len(familia)} nodos probados (todas las series y niveles, con los que no pasan); {n_cand} con p < 0.05; "
             f"umbral de Benjamini-Hochberg (q = 0.10, tope 0.05) p ≤ {pstar:.4f}; permutación de bloques de {BLOQUE} años.")

    ajustes_se = {}
    for niv in (925, 850, 700):
        for sid in EST:
            z = D[(sid, niv)].dropna()
            if len(z) < 72:
                continue
            nodos = NODOS[(sid, niv)]
            cp = [(k, p) for k, p, i, pa in nodos if i in ACEPTADOS[(sid, niv)]]
            otros = OTROS[(sid, niv)]
            mantener = []
            for k, pk in cp:
                fecha = z.index[k]
                paso = cambio(z, fecha)
                otro = BRA if sid in ARG else ARG
                cs = [cambio(D[(o, niv)].dropna(), fecha) for o in otro if o != sid]
                cs = [c for c in cs if np.isfinite(c)]
                ctrl = float(np.mean(cs)) if cs else np.nan
                regional = bool(cs) and np.isfinite(paso) and np.sign(ctrl) == np.sign(paso) and abs(ctrl) >= 0.5 * abs(paso)
                estab = np.mean([any(abs(kk - k) <= 3 for kk in o) for o in otros])
                if estab < 0.8:      # cortes que no reaparecen con otras semillas se descartan
                    continue
                quiebres.append({"sid": sid, "estacion": EST[sid], "nivel": niv, "fecha": str(fecha), "paso_m_s": paso, "control_m_s": ctrl,
                                 "n_control": len(cs), "regional": regional, "p_permutacion": pk, "estabilidad_5_semillas": estab, "BH_q10": True})
                if not regional:
                    mantener.append(k)
            lim = [0] + mantener + [len(z)]
            medias = [z.values[lim[i]:lim[i + 1]].mean() for i in range(len(lim) - 1)]
            varz = [z.values[lim[i]:lim[i + 1]].var(ddof=1) / max(lim[i + 1] - lim[i], 2) for i in range(len(lim) - 1)]
            ajuste = pd.Series(0.0, index=z.index)
            se = pd.Series(0.0, index=z.index)
            segid = pd.Series(0, index=z.index)
            for i in range(len(lim) - 1):
                ajuste.iloc[lim[i]:lim[i + 1]] = medias[i] - medias[-1]
                se.iloc[lim[i]:lim[i + 1]] = np.sqrt(varz[i] + varz[-1]) if i < len(lim) - 2 else 0.0
                segid.iloc[lim[i]:lim[i + 1]] = i
            ajustes[(sid, niv)] = ajuste
            ajustes_se[(sid, niv)] = (se, segid)
            for i in range(len(lim) - 1):
                filas_aj.append({"sid": sid, "nivel": niv, "desde": str(z.index[lim[i]]), "hasta": str(z.index[lim[i + 1] - 1]),
                                 "media": medias[i], "ajuste_m_s": medias[i] - medias[-1], "se_m_s": float(se.iloc[lim[i]])})
    Q = pd.DataFrame(quiebres)
    L.append(f"  {len(Q)} cortes aceptados por BH sobre la familia completa; estabilidad media con 5 semillas y 300 permutaciones: "
             f"{100 * Q.estabilidad_5_semillas.mean():.0f} %; cortes con estabilidad ≥ 80 %: {int((Q.estabilidad_5_semillas >= 0.8).sum())}")
    for r_ in quiebres:
        L.append(f"  {r_['estacion']:13s} {r_['nivel']} hPa {r_['fecha']}: paso {r_['paso_m_s']:+.2f} m/s; control {r_['control_m_s']:+.2f} (n {r_['n_control']}) → "
                 f"{'regional, no se ajusta' if r_['regional'] else 'se ajusta'}; p {r_['p_permutacion']:.3f}; estabilidad {100 * r_['estabilidad_5_semillas']:.0f} %"
                 + (f"; BH {'sí' if r_.get('BH_q10') else 'no'}" if 'BH_q10' in r_ else ""))
    pd.DataFrame(quiebres).to_csv("analisis/jet/22_quiebres.csv", index=False)
    pd.DataFrame(filas_aj).to_csv("analisis/jet/22_ajustes.csv", index=False)
    pd.to_pickle({k: v for k, v in ajustes.items()}, "analisis/jet/22_ajustes.pkl")

    # ---- (c) jets con niveles fijos con viento crudo y ajustado (los meses sin pares toman el ajuste del tramo vecino, igual que en el paso 24)
    rango = pd.period_range("1979-01", "2026-12", freq="M")
    base = {}
    for sid, nom in EST.items():
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[(s.index.hour == 12) & s.index.month.isin([10, 11, 12, 1, 2, 3])].copy()
        s["temp"] = np.where(s.index.month >= 10, s.index.year + 1, s.index.year)
        s = s[(s.temp >= 1980) & (s.temp <= 2025)]
        dr = (np.degrees(np.arctan2(-s.u850, -s.v850)) + 360) % 360
        w0 = {niv: np.hypot(s[f"u{niv}"], s[f"v{niv}"]) for niv in (850, 700)}
        base[sid] = dict(s=s, per=s.index.to_period("M"), w0=w0, dirok=(dr >= 292.5) | (dr <= 67.5), disp=w0[850].notna() & w0[700].notna())

    def calcular(ajs):
        filas = []
        for sid in EST:
            b_ = base[sid]
            s = b_["s"].copy()
            w = {}
            for niv in (850, 700):
                adj = ajs.get((sid, niv))
                a = (pd.Series(b_["per"].map(adj.reindex(rango).ffill().bfill()), index=s.index).astype(float) if adj is not None else 0.0)
                w[niv] = np.maximum(b_["w0"][niv] - a, 0)
            s["fijos"] = ((b_["w0"][850] >= 12) & (b_["w0"][850] - b_["w0"][700] >= 6) & b_["dirok"]).astype(float).where(b_["disp"])
            s["fijos_adj"] = ((w[850] >= 12) & (w[850] - w[700] >= 6) & b_["dirok"]).astype(float).where(b_["disp"])
            s["mes"], s["sid"] = s.index.month, sid
            filas.append(s[["sid", "temp", "mes", "fuente", "fijos", "fijos_adj"]])
        return pd.concat(filas)

    D2 = calcular(ajustes)
    D2.to_parquet("analisis/jet/22_fijos.parquet")
    L.append("\nFrecuencia media de jet con niveles fijos (%, 12 UTC oct-mar 1980-2025): " +
             ", ".join(f"{EST[i]} {100 * v:.1f}" for i, v in D2.groupby("sid").fijos.mean().items()))

    def beta(d, y, fuente=True):
        x = d.dropna(subset=[y])
        X = [pd.get_dummies(x.mes).values.astype(float), ((x.temp.values - 2000) / 10.0)[:, None],
             pd.get_dummies(x.sid, drop_first=True).values.astype(float)]
        if fuente:
            X.append(pd.get_dummies(x.fuente, drop_first=True).values.astype(float))
        return 100 * np.linalg.lstsq(np.hstack(X), x[y].values.astype(float), rcond=None)[0][len(np.unique(x.mes))]

    def boot(d, y, fuente=True, n=1000):
        r = np.random.default_rng(123)           # generador fijo: el mismo estimando da el mismo intervalo en todos los pasos
        temps = d.temp.unique()
        g = {t: d[d.temp == t] for t in temps}
        out = [beta(pd.concat([g[t] for t in r.choice(temps, len(temps))]), y, fuente) for _ in range(n)]
        return np.percentile(out, [5, 95]), np.percentile(out, [2.5, 97.5])

    csvt = []
    L.append("\nTendencia agrupada (10 estaciones) de jets con niveles fijos, pp/década [IC 90 % por bootstrap de temporadas];"
             " entre paréntesis, % relativo por década respecto de la media del período")
    for ini in (1980, 1993):
        x = D2[D2.temp >= ini]
        for y, lab in (("fijos", "crudo"), ("fijos_adj", "ajustado con ERA5")):
            for fu in (False, True):
                b = beta(x, y, fu)
                (lo, hi), (lo95, hi95) = boot(x, y, fu)
                media = 100 * x[y].mean()
                csvt.append({"desde": ini, "serie": lab, "con_fuente": fu, "beta": b, "lo": lo, "hi": hi, "lo95": lo95, "hi95": hi95, "media": media})
                L.append(f"  desde {ini} {lab:18s} {'+ fuente' if fu else '        '}: {b:+.2f} [{lo:+.2f},{hi:+.2f}]"
                         f"  ({100 * b / media:+.0f} %/déc; media {media:.1f} %)")
    # incertidumbre del ajuste: se perturba cada ajuste por tramo con su error estándar (200 réplicas) y se recalcula la tendencia agrupada
    rp = np.random.default_rng(2200)
    sd_aj = {}
    for ini in (1980, 1993):
        v = []
        for _ in range(200):
            ajs = {}
            for k, aj in ajustes.items():
                se, segid = ajustes_se[k]
                dr_ = rp.normal(0, 1, int(segid.max()) + 1)
                ajs[k] = aj + pd.Series(dr_[segid.values], index=aj.index) * se
            v.append(beta(calcular(ajs)[lambda d: d.temp >= ini], "fijos_adj", True))
        sd_aj[ini] = float(np.std(v))
        for r_ in csvt:
            if r_["desde"] == ini and r_["serie"] == "ajustado con ERA5" and r_["con_fuente"]:
                se_b = (r_["hi"] - r_["lo"]) / (2 * 1.645)
                tot = float(np.sqrt(se_b ** 2 + sd_aj[ini] ** 2))
                r_["lo_total"], r_["hi_total"] = r_["beta"] - 1.645 * tot, r_["beta"] + 1.645 * tot
                r_["lo95_total"], r_["hi95_total"] = r_["beta"] - 1.96 * tot, r_["beta"] + 1.96 * tot
                L.append(f"  desde {ini}, ajustado + fuente, con la incertidumbre del ajuste: DE por perturbación de los ajustes {sd_aj[ini]:.3f} pp/déc;"
                         f" IC 90 % total [{r_['lo_total']:+.2f},{r_['hi_total']:+.2f}]; IC 95 % total [{r_['lo95_total']:+.2f},{r_['hi95_total']:+.2f}]")
    pd.DataFrame(csvt).to_csv("analisis/jet/22_tendencias.csv", index=False)
    # tendencia por estación (modelo de probabilidad lineal por estación, sin fuente)
    L.append("\nPor estación (1980-2025), pp/década sin/con ajuste:")
    for sid, nom in EST.items():
        x = D2[D2.sid == sid]
        L.append(f"  {nom:13s} crudo {beta(x, 'fijos', False):+.2f}  ajustado {beta(x, 'fijos_adj', False):+.2f}")

    # ---- (d) bache de los noventa en el índice fijo
    L.append("\n(d) Fracción de sondeos con 850 y 700 hPa reportados y frecuencia de jet fijo en ERA5 con y sin esos niveles en el sondeo")
    dd = []
    for sid, nom in EST.items():
        s = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        s = s[(s.index.hour == 12) & s.index.month.isin([10, 11, 12, 1, 2, 3])]
        s.index = s.index.normalize()
        s = s[~s.index.duplicated()]
        disp = (s.u850.notna() & s.u700.notna())
        dd.append(pd.DataFrame({"sid": sid, "fecha": s.index, "disp": disp.values}))
    dd = pd.concat(dd)
    r12 = pd.read_parquet("analisis/jet/12_diario.parquet")[["sid", "fecha", "era_fijos"]]
    r12["era_fijos"] = r12.era_fijos.astype(float)
    dd = dd.merge(r12, on=["sid", "fecha"])
    dd["bloque"] = pd.cut(dd.fecha.dt.year, [1979, 1989, 1999, 2009, 2019, 2026], labels=["80s", "90s", "00s", "10s", "20s"])
    t = dd.groupby("bloque", observed=True).apply(lambda q: pd.Series({
        "%_con_850/700": 100 * q.disp.mean(), "ERA5_jet_%_con": 100 * q[q.disp].era_fijos.mean(),
        "ERA5_jet_%_sin": 100 * q[~q.disp].era_fijos.mean(), "n_sin": int((~q.disp).sum())}), include_groups=False)
    L.append(t.round(1).to_string())
    # ---- (f) ¿el escalón de 850 hPa de ~nov 1991 es del sondeo o de ERA5? sondeo − ERA5 en todas las estaciones
    L.append("\n(f) Sondeo − ERA5 a 850 hPa (anomalía mensual media, m/s): seis años antes y después de nov 1991, por estación")
    for sid, nom in EST.items():
        z = D[(sid, 850)]
        a, d_ = z[(z.index >= pd.Period("1985-11", "M")) & (z.index <= pd.Period("1991-10", "M"))], z[(z.index >= pd.Period("1991-11", "M")) & (z.index <= pd.Period("1997-10", "M"))]
        if len(a) >= 12 and len(d_) >= 12:
            L.append(f"  {nom:13s} antes {a.mean():+.2f} (n {len(a)}), después {d_.mean():+.2f} (n {len(d_)}), cambio {d_.mean() - a.mean():+.2f}")
        else:
            L.append(f"  {nom:13s} sin datos suficientes en ambos lados (antes {len(a)}, después {len(d_)} meses)")
    # ---- (e) el bache de los 90 del índice fijo: sondeo vs ERA5 en los mismos días, y tendencia sin 1993-2000
    L.append("\n(e) Frecuencia de jet fijo (%) por período, mismos días con sondeo: sondeo | ERA5 (10 estaciones, media de estaciones)")
    r12 = pd.read_parquet("analisis/jet/12_diario.parquet")
    r12 = r12[r12.sid.isin(EST)].dropna(subset=["obs_fijos"])
    r12["per"] = pd.cut(r12.temp, [1979, 1992, 2000, 2025], labels=["1980-92", "1993-2000", "2001-25"])
    t = r12.assign(o=r12.obs_fijos.astype(float), e=r12.era_fijos.astype(float)).groupby(["sid", "per"], observed=True)[["o", "e"]].mean() * 100
    t = t.groupby("per", observed=True).mean()
    L.append(t.round(2).to_string())
    x = D2[(D2.temp <= 1992) | (D2.temp >= 2001)]
    (lo, hi), _ = boot(x, "fijos", True)
    L.append(f"  Tendencia agrupada sin las temporadas 1993-2000, crudo + fuente: {beta(x, 'fijos', True):+.2f} [{lo:+.2f},{hi:+.2f}]")
    open("analisis/jet/22_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
