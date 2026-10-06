"""Paper 2: figuras (mismo estilo que el paper 1: Arial, paletas de Crameri, mapa base con provincias).

La numeración sigue el orden de cita en el texto.
Fig. 1  Mapas de tendencia de la frecuencia del jet del norte (oct-mar, 12 UTC, 1980-2025): (a) perfil completo; (b) niveles fijos 850/700 hPa.
Fig. 2  Series en Santa Rosa, Córdoba y Resistencia: tres criterios, ERA5 en los mismos días, archivo de origen y quiebres del viento.
Fig. 3  Tendencias por estación y agrupadas: crudas y corregidas por fuente de archivo (paso 10).
Fig. 4  Resultado principal: tendencia agrupada del perfil completo y de los jets profundos con controles sucesivos (pasos 22 y 24).
Fig. 5  Resolución: emulación y validación contra archivos reales (paso 19).
Fig. 6  Física de los jets: perfiles, presión, punto de rocío y giro nocturno (pasos 15, 16, 27).
Fig. 7  Lluvia CHIRPS y flujo de humedad de ERA5 (pasos 7 y 18).
Fig. 8  ERA5 frente a los sondeos (pasos 20 y 23).
Fig. 9  Diferencia sondeo − ERA5 a 925 hPa (pasos 13 y 22).
Salida: figuras/jet/
"""
import glob
import importlib
import os
import sys

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
fp = importlib.import_module("figuras_paper")
from estadistica import mk_hamed_rao, sen  # noqa: E402

OUT = "figuras/jet"
NOMBRES = {"ARM00087623": "Santa Rosa", "ARM00087344": "Córdoba", "ARM00087155": "Resistencia"}
CLAVE = {"ARM00087623": "Santa Rosa", "ARM00087344": "Córdoba", "ARM00087155": "Resistencia", "BRM00083827": "Foz", "BRM00083612": "C. Grande"}
COL = {"A": fp.DRY, "B": fp.WET, "C": fp.NEUTRAL}
ETQ = {"A": "Deep jet (850/700 hPa)", "B": "Shallow jet (full profile only)", "C": "No jet"}
COL_FUENTE = {"usaf-ds3": "#7a9e7e", "ncdc-gts": "#6b8fb5", "otros": "#d9a05b"}


def rotulo(ax, t, x=0.0, y=1.02):
    ax.text(x, y, t, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


from comun import series, temporada, tendencia  # noqa: E402


def fuente_dominante(sid):
    d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
    d = d[(d.index.hour == 12) & d.index.month.isin([10, 11, 12, 1, 2, 3]) & (d.index.year >= 1979)]
    f = d.fuente.where(d.fuente.isin(["usaf-ds3", "ncdc-gts"]), "otros")
    return f.groupby(temporada(d.index)).agg(lambda x: x.value_counts().index[0]).loc[1980:2025]


def quiebres_propios(sid):
    """Quiebres de sondeo − ERA5 de la estación que no son regionales (paso 22): lista de (nivel, fecha decimal)."""
    q = pd.read_csv("analisis/jet/22_quiebres.csv")
    q = q[(q.sid == sid) & (~q.regional)]
    out = []
    for r in q.itertuples():
        t = pd.Period(r.fecha, "M").start_time
        out.append((r.nivel, t.year + t.dayofyear / 365.25))
    return out


def ic_estacional(serie):
    """Media e IC 90 % por bootstrap de temporadas de una Series con índice de fechas."""
    z = serie.dropna()
    g = z.groupby(temporada(z.index)).agg(["sum", "count"])
    r = np.random.default_rng(3)
    bs = []
    for _ in range(300):
        i = r.integers(0, len(g), len(g))
        bs.append(g["sum"].values[i].sum() / g["count"].values[i].sum())
    return z.mean(), np.percentile(bs, 5), np.percentile(bs, 95)


# ---------------------------------------------------------------------------------------------------------------- Fig. 1
def fig1():
    r = pd.read_csv("analisis/jet/03b_tendencias.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 5.0), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, col, pcol, tit, lim in ((axs[0], "ta", "pa", "All reported levels", 6.0), (axs[1], "tb", "pb", "Fixed 850 and 700 hPa levels", 1.0)):
        norma = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        fp.mapa_base(ax)
        ax.set_extent([-80, -34, -45, 6], crs=ccrs.PlateCarree())
        for sig in (True, False):
            for signo, mk in ((1, "^"), (-1, "v")):
                q = r[((r[pcol] < 0.05) == sig) & ((r[col] >= 0) == (signo > 0))]
                if q.empty:
                    continue
                if sig:
                    ax.scatter(q.lon, q.lat, s=52, marker=mk, c=q[col].clip(-lim, lim), cmap=fp.CMAP_T, norm=norma, edgecolors=fp.INK,
                               linewidths=0.6, transform=ccrs.PlateCarree(), zorder=5)
                else:
                    ax.scatter(q.lon, q.lat, s=52, marker=mk, facecolors="white", edgecolors=fp.CMAP_T(norma(q[col].clip(-lim, lim).values)),
                               linewidths=1.5, transform=ccrs.PlateCarree(), zorder=5)
        for sid, nom in CLAVE.items():
            if sid in r.sid.values:
                z = r[r.sid == sid].iloc[0]
                ax.text(z.lon + 1.8, z.lat - 0.5, nom, fontsize=6, color=fp.INK, transform=ccrs.PlateCarree(), zorder=6)
        rotulo(ax, f"({'ab'[ax is axs[1]]}) {tit}")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=fp.GRID, xlocs=[-70, -60, -50, -40], ylocs=[-40, -30, -20, -10, 0])
        gl.top_labels = gl.right_labels = False
        gl.left_labels = ax is axs[0]
        gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": fp.INK2}
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norma, cmap=fp.CMAP_T), ax=ax, orientation="horizontal", shrink=0.85, pad=0.07, aspect=25, extend="both")
        cb.set_label("Trend (pp decade$^{-1}$)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
    fig.text(0.5, 0.005, "Triangle up/down: positive/negative trend; filled: p < 0.05 (uncorrected)", ha="center", fontsize=6.5, color=fp.INK2)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/fig1_tendencias_mapa.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- Fig. 2
def fig2():
    S = pd.read_parquet("analisis/jet/20_series.parquet")
    fig = plt.figure(figsize=(6.8, 8.2))
    gs = fig.add_gridspec(6, 1, height_ratios=[5, 0.55, 5, 0.55, 5, 0.55], hspace=0.28)
    for k, (sid, nom) in enumerate(NOMBRES.items()):
        ax = fig.add_subplot(gs[2 * k])
        st = fig.add_subplot(gs[2 * k + 1], sharex=ax)
        s = series(sid)
        z = S[S.sid == sid].set_index("temp")
        ax.plot(s.index, s.completo, color=fp.DRY, lw=1.5, label="Soundings, all reported levels")
        ax.plot(s.index, s.fijos, color=fp.WET, lw=1.5, label="Soundings, fixed 850/700 hPa")
        ax.plot(z.index, z.era_perfil, color=fp.DRY, lw=1.1, ls=":", label="ERA5, all-level criterion")
        ax.plot(z.index, z.era_fijos, color=fp.WET, lw=1.1, ls=":", label="ERA5, fixed levels")
        top = ax.get_ylim()[1]
        for niv, x in quiebres_propios(sid):
            ax.axvline(x, color=fp.INK if niv == 925 else fp.INK2, lw=0.9, ls="-" if niv == 925 else "--", alpha=0.8)
        if sid == "ARM00087344":
            ax.axvline(2002.9, color=fp.GRID, lw=1.1, ls="-.")
        rotulo(ax, f"({'abc'[k]}) {nom}")
        ax.set_ylabel("LLJ frequency (%)")
        ax.grid(axis="y", color=fp.GRID, lw=0.5)
        ax.tick_params(labelbottom=False)
        fd = fuente_dominante(sid)
        for t, f in fd.items():
            st.axvspan(t - 0.5, t + 0.5, color=COL_FUENTE.get(f, COL_FUENTE["otros"]), lw=0)
        st.set_yticks([])
        st.set_ylabel("archive", fontsize=6.5, rotation=0, ha="right", va="center")
        st.set_xlim(1979.5, 2025.5)
        if k < 2:
            st.tick_params(labelbottom=False)
        if k == 0:
            h, l = ax.get_legend_handles_labels()
            h += [plt.Line2D([], [], color=fp.INK, lw=0.9), plt.Line2D([], [], color=fp.INK2, lw=0.9, ls="--"),
                  plt.Rectangle((0, 0), 1, 1, color=COL_FUENTE["usaf-ds3"]), plt.Rectangle((0, 0), 1, 1, color=COL_FUENTE["ncdc-gts"]),
                  plt.Rectangle((0, 0), 1, 1, color=COL_FUENTE["otros"])]
            l += ["925-hPa wind break (specific to the station)", "850- or 700-hPa wind break", "Archive: US Air Force", "Archive: GTS", "Archive: NCDC 6314/6316/6322"]
            ax.legend(h, l, fontsize=5.8, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95, loc="upper left", ncol=2, columnspacing=1.0, bbox_to_anchor=(0.0, 1.0))
            ax.set_ylim(0, top * 1.55)
    st.set_xlabel("Season (year of March)")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.97, bottom=0.06)
    fig.savefig(f"{OUT}/fig2_series_criterios.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- Fig. 3
def fig3():
    r = pd.read_csv("analisis/jet/10_tendencias.csv")
    orden = [e for e in r.estacion.unique() if not e.startswith("Agrupado")] + ["Agrupado", "Agrupado sin Porto Velho"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True)
    for ax, crit, tit in ((axs[0], "completo", "All reported levels"), (axs[1], "fijos", "Fixed 850/700 hPa")):
        ax.axvline(0, color=fp.INK2, lw=0.7)
        for fu, c, mk, desp, lab in ((False, fp.NEUTRAL, "o", 0.17, "Raw"),
                                      (True, fp.DRY if crit == "completo" else fp.WET, "s", -0.17, "Archive-source corrected")):
            q = r[(r.criterio == crit) & (r.corrige_fuente == fu)].set_index("estacion").reindex(orden)
            y = np.arange(len(orden))[::-1] + desp
            ax.errorbar(q.beta, y, xerr=[q.beta - q.lo, q.hi - q.beta], fmt=mk, color=c, ms=4.5, elinewidth=1, capsize=1.5, label=lab,
                        mfc="white" if not fu else c)
        rotulo(ax, f"({'ab'[crit == 'fijos']}) {tit}")
        ax.set_xlabel("Trend (pp decade$^{-1}$)")
        ax.grid(axis="x", color=fp.GRID, lw=0.5)
        ax.axhline(1.5, color=fp.GRID, lw=0.8)
    etq = {"Agrupado": "Pooled (10 stations)", "Agrupado sin Porto Velho": "Pooled without Porto Velho (9)", "Porto Velho": "Porto Velho*"}
    axs[0].set_yticks(np.arange(len(orden))[::-1], [etq.get(o, o) for o in orden], fontsize=7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.012, "* record with a break in 1992 (Sect. 4.4)", fontsize=6, color=fp.INK2, ha="center")
    fig.savefig(f"{OUT}/fig3_tendencias_corregidas.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- Fig. 4
def fig4():
    K = pd.read_csv("analisis/jet/24_cascada.csv")
    T22 = pd.read_csv("analisis/jet/22_tendencias.csv")

    def fila(desde, viento, ctl):
        q = K[(K.desde == desde) & (K.viento == viento) & (K.control == ctl)]
        return None if q.empty else q.iloc[0]

    filas = [("Full profile, raw", ("crudo", "sólo mes y estación"), fp.NEUTRAL, "o"),
             ("+ archive source", ("crudo", "+ fuente"), fp.INK2, "s"),
             ("+ source, wind breaks removed", ("ajustado", "+ fuente"), fp.DRY, "^"),
             ("  same, with adjustment uncertainty", ("ajustado", "+ fuente, incertidumbre del ajuste"), fp.DRY, "P"),
             ("+ source + mandatory levels (not exogenous; not used)", ("ajustado", "+ fuente + niveles mandatorios"), fp.NEUTRAL, "D"),
             ("Only ≥ 5 levels (selects on level count; not used)", ("ajustado", "sólo sondeos con ≥ 5 niveles en 0-250 hPa + fuente"), fp.NEUTRAL, "v"),
             ("Only USAF + GTS, + source, breaks removed", ("ajustado", "sólo archivos USAF + GTS (excluye NCDC 6314/6316/6322) + fuente"), fp.DRY, "<"),
             ("+ source + all reported levels (endogenous; not used)", ("ajustado", "+ fuente + nº de niveles (endógeno)"), fp.NEUTRAL, "x")]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 5.0), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    yl, yt = [], []
    for ax, ini in zip(axs, (1980, 1993)):
        y0 = 0
        yt = []
        for lab, (viento, ctl), c, mk in filas:
            q = fila(ini, viento, ctl)
            if q is not None:
                ax.errorbar(q.beta, y0, xerr=[[q.beta - q.lo], [q.hi - q.beta]], fmt=mk, color=c, ms=5, capsize=2, elinewidth=1, mfc="white" if mk == "x" else c)
            yt.append(y0)
            if ini == 1980:
                yl.append(lab)
            y0 -= 1
        y0 -= 0.6
        for lab, serie, fu, c, mk in (("Deep jets (fixed levels), raw", "crudo", False, fp.WET, "o"),
                                      ("Deep jets, + source", "crudo", True, fp.WET, "s"),
                                      ("Deep jets, + source, breaks removed", "ajustado con ERA5", True, fp.WET, "^")):
            q = T22[(T22.desde == ini) & (T22.serie == serie) & (T22.con_fuente == fu)].iloc[0]
            ax.errorbar(q.beta, y0, xerr=[[q.beta - q.lo95], [q.hi95 - q.beta]], fmt=mk, color=c, ms=5, capsize=2, elinewidth=1, mfc="white" if not fu else c)
            yt.append(y0)
            if ini == 1980:
                yl.append(lab)
            y0 -= 1
        ax.axvline(0, color=fp.INK2, lw=0.7)
        ax.grid(axis="x", color=fp.GRID, lw=0.5)
        ax.set_xlabel("Pooled trend (pp decade$^{-1}$)")
        rotulo(ax, f"({'ab'[ini == 1993]}) {ini}–2025")
    axs[0].set_yticks(yt, yl, fontsize=6.6)
    axs[1].text(0.97, 0.03, "Restricted samples and the\nendogenous control are shown\nonly for 1980–2025", transform=axs[1].transAxes, fontsize=6, ha="right", color=fp.INK2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_cascada_tendencias.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- Fig. 5
def fig5():
    E = pd.read_csv("analisis/jet/19_emulacion.csv")
    V = pd.read_csv("analisis/jet/19_validacion.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.9))
    ax = axs[0]
    x = np.arange(len(E))
    ax.plot(x - 0.24, E.obs_80_95, "o", mfc="white", mec=fp.INK, ms=5, label="Observed 1980–1995")
    ax.plot(x - 0.08, E.moderno, "o", color=fp.DRY, ms=5, label="Observed 2005–2025")
    ax.plot(x + 0.08, E.no_adaptativo, "s", color=fp.WET, ms=5, label="2005–2025 degraded, fixed levels (lower bound)")
    ax.plot(x + 0.24, E.adaptativo, "^", color=fp.INK2, ms=5, label="2005–2025 degraded, adaptive levels (upper bound)")
    ax.set_xticks(x, E.estacion, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("LLJ frequency, all levels (%)")
    rotulo(ax, "(a) Resolution emulation")
    ax.grid(axis="y", color=fp.GRID, lw=0.5)
    ax = axs[1]
    x = np.arange(len(V))
    ax.axhline(0, color=fp.INK2, lw=0.6)
    for col, mk, c, d, lab in (("obs", "o", fp.INK, -0.2, "Observed USAF − GTS"), ("na", "s", fp.WET, 0.0, "Predicted, fixed levels"),
                               ("ad", "^", fp.INK2, 0.2, "Predicted, adaptive levels")):
        lo, hi = V[col + "_lo"], V[col + "_hi"]
        ax.errorbar(x + d, V[col], yerr=[np.where(np.isfinite(lo), V[col] - lo, 0), np.where(np.isfinite(hi), hi - V[col], 0)], fmt=mk, color=c, ms=5,
                    capsize=2, elinewidth=1, label=lab)
    ax.set_xticks(x, V.estacion, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Difference in LLJ frequency (pp)")
    rotulo(ax, "(b) Emulator vs real archives")
    ax.grid(axis="y", color=fp.GRID, lw=0.5)
    h1, l1 = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(h1, l1, loc="lower left", fontsize=6.3, frameon=False, ncol=1, bbox_to_anchor=(0.04, 0.0))
    fig.legend(h2, l2, loc="lower right", fontsize=6.3, frameon=False, ncol=1, bbox_to_anchor=(0.98, 0.0))
    fig.savefig(f"{OUT}/fig5_emulacion.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- Fig. 6
def fig6():
    O = pd.read_parquet("analisis/jet/27_perfiles.parquet")
    I = pd.read_parquet("analisis/jet/16_ic.parquet")
    P = pd.read_parquet("analisis/jet/15_pares.parquet")
    P = P[P.clase_12.isin(["A", "B", "C"])]
    lat = {"BRM00083827": -25.6, "BRM00083612": -20.5, "BRM00083362": -15.6, "BRM00083928": -29.8, "BRM00083768": -23.3,
           "BRM00083208": -12.7, "BRM00082824": -8.8}
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.2))
    ax = axs[0, 0]
    for k in "CBA":
        q = O[O.clase == k]
        ax.fill_betweenx(q.altura, q.q1, q.q3, color=COL[k], alpha=0.18, lw=0)
        ax.plot(q.med, q.altura, color=COL[k], lw=1.7, label=f"{ETQ[k]} (n = {int(q.n.max())})")
    ax.set_xlabel("Wind speed (m s$^{-1}$)")
    ax.set_ylabel("Height above ground (hPa)")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    rotulo(ax, "(a) Composite wind profiles")
    ax.grid(color=fp.GRID, lw=0.4)
    for ax, var, tit, yl in ((axs[0, 1], "pa", "(b) Surface pressure", "Anomaly (hPa)"), (axs[1, 0], "Tda", "(c) Surface dew point", "Anomaly (°C)")):
        for k in "CBA":
            q = I[(I.clase == k) & (I["var"] == var)].sort_values("lag")
            ax.fill_between(q.lag, q.lo, q.hi, color=COL[k], alpha=0.2, lw=0)
            ax.plot(q.lag, q.media, "-o", color=COL[k], ms=3, lw=1.5)
        ax.axhline(0, color=fp.INK2, lw=0.6)
        ax.axvline(0, color=fp.GRID, lw=0.8)
        ax.set_xlabel("Day relative to the 12-UTC sounding")
        ax.set_ylabel(yl)
        rotulo(ax, tit)
        ax.grid(axis="y", color=fp.GRID, lw=0.4)
    ax = axs[1, 1]
    la = np.linspace(7, 31, 50)
    ax.plot(la, -np.degrees(2 * 7.292e-5 * np.sin(np.radians(la)) * 43200), color=fp.INK, lw=1.0, ls="--", label="Inertial rotation $f$ · 12 h")
    for k, d in zip("CBA", (-0.25, 0.0, 0.25)):
        xs, ys, lo, hi = [], [], [], []
        for sid in lat:
            z = P[(P.sid == sid) & (P.clase_12 == k)].rot925
            if len(z.dropna()) >= 20:
                m, a, b = ic_estacional(z)
                xs.append(abs(lat[sid]) + d)
                ys.append(m)
                lo.append(m - a)
                hi.append(b - m)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=COL[k], ms=4, capsize=1.5, elinewidth=0.8, label=ETQ[k])
    ax.axhline(0, color=fp.INK2, lw=0.6)
    ax.set_xlabel("Latitude (°S)")
    ax.set_ylabel("Mean turning of the 925-hPa wind,\n00→12 UTC (°)")
    ax.legend(fontsize=5.8, frameon=False, loc="lower left")
    rotulo(ax, "(d) Overnight wind turning")
    ax.grid(color=fp.GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_fisica_clases.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- Fig. 7
def fig7():
    import xarray as xr
    m = xr.open_dataset("analisis/jet/18_mapas.nc")
    f = xr.open_dataset("analisis/jet/07_compuestos.nc")
    est = {"Resistencia": (-27.45, -59.05), "Córdoba": (-31.31, -64.22), "Santa Rosa": (-36.57, -64.27), "Foz do Iguaçu": (-25.6, -54.5)}
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.0), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.12, wspace=0.02, hspace=0.32)
    paneles = ((axs[0, 0], "rain", "A", "(a) Rain after deep jets − no jet"), (axs[0, 1], "rain", "B", "(b) Rain after shallow jets − no jet"),
               (axs[1, 0], "flux", "A", "(c) Moisture flux, deep jets"), (axs[1, 1], "flux", "B", "(d) Moisture flux, shallow jets"))
    nl = mpl.colors.Normalize(vmin=-10, vmax=10)
    nf = mpl.colors.Normalize(vmin=-120, vmax=120)
    for ax, tipo, k, tit in paneles:
        fp.mapa_base(ax)
        if tipo == "rain":
            d = m[f"lluvia_{k}"] - m["lluvia_C"]
            ax.pcolormesh(d.lon, d.lat, d.values, cmap="BrBG", norm=nl, transform=ccrs.PlateCarree(), zorder=1)
            ax.set_extent([-65, -50, -40, -20.2], crs=ccrs.PlateCarree())
            ax.plot([-62, -53, -53, -62, -62], [-33, -33, -25, -25, -33], color=fp.INK, lw=0.9, transform=ccrs.PlateCarree(), zorder=6)
        else:
            d = sum(f[f"{sid}_{k}"] for sid in ("ARM00087155", "ARM00087344", "ARM00087623")) / 3.0
            ax.pcolormesh(d.longitude, d.latitude, d.values, cmap=fp.CMAP_HUM, norm=nf, transform=ccrs.PlateCarree(), zorder=1)
            ax.set_extent([-75, -45, -45, -10], crs=ccrs.PlateCarree())
            for cx, cy, ls in (([-65, -55, -55, -65, -65], [-32, -32, -20, -20, -32], "-"), ([-68, -58, -58, -68, -68], [-40, -40, -30, -30, -40], "--")):
                ax.plot(cx, cy, color=fp.INK, lw=0.9, ls=ls, transform=ccrs.PlateCarree(), zorder=6)
        for la, lo in est.values():
            ax.plot(lo, la, "^", color=fp.INK, ms=4.5, mfc="white", transform=ccrs.PlateCarree(), zorder=7)
        rotulo(ax, tit)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=fp.GRID)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = ax in (axs[0, 0], axs[1, 0])
        gl.xlabel_style = gl.ylabel_style = {"size": 6.5, "color": fp.INK2}
    for (a1, a2), nm, cm, lab in (((axs[0, 0], axs[0, 1]), nl, "BrBG", "Rain on the day of the sounding and the next, anomaly relative to days without a jet (mm)"),
                                  ((axs[1, 0], axs[1, 1]), nf, fp.CMAP_HUM, "Southward moisture flux at 850 hPa, mean of three station composites, relative to days without a jet (g kg$^{-1}$ m s$^{-1}$)")):
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=nm, cmap=cm), ax=[a1, a2], orientation="horizontal", shrink=0.7, pad=0.08, extend="both")
        cb.set_label(lab, fontsize=6.8)
        cb.ax.tick_params(labelsize=7)
    fig.savefig(f"{OUT}/fig7_lluvia_flujo.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- Fig. 8
def fig8():
    h = pd.read_parquet("analisis/jet/20_aciertos.parquet")
    g = pd.read_parquet("analisis/jet/20_atribucion.parquet")
    M = pd.read_csv("analisis/jet/23_metricas.csv")
    nom = {"ARM00087155": "Resistencia", "ARM00087344": "Córdoba", "ARM00087576": "Ezeiza", "ARM00087623": "Santa Rosa", "BRM00083827": "Foz",
           "BRM00083612": "C. Grande"}
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.3), gridspec_kw={"width_ratios": [1.15, 1.5, 0.9]})
    ax = axs[0]
    h["q"] = ((h.temp.clip(upper=2024) - 1980) // 5 * 5 + 1980)
    rng_ = np.random.default_rng(1)
    for sid in nom:
        z = h[h.sid == sid].groupby("q").era_fijos.agg(["mean", "size"])
        z = z[z["size"] >= 8]
        if len(z) >= 3:
            ax.plot(z.index + 2.5, z["mean"], "-", color="#b9b9b9", lw=0.9)
            ax.text(z.index[-1] + 3.2, z["mean"].iloc[-1] + {"Córdoba": -0.03, "Resistencia": 0.03, "Ezeiza": 0.03, "Santa Rosa": -0.03, "C. Grande": 0.012, "Foz": -0.02}.get(nom[sid], 0), nom[sid], fontsize=5, va="center", color=fp.INK2)
    for q, z in h.groupby("q"):
        v = z.era_fijos.values
        b = [v[rng_.integers(0, len(v), len(v))].mean() for _ in range(500)]
        ax.errorbar(q + 2.5, v.mean(), yerr=[[v.mean() - np.percentile(b, 5)], [np.percentile(b, 95) - v.mean()]], fmt="o", color=fp.WET, ms=4.5, capsize=2, lw=1, zorder=5)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1978, 2036)
    ax.set_xticks([1980, 2000, 2020])
    ax.set_xlabel("Season (five-season blocks)")
    ax.set_ylabel("POD of ERA5 for observed jets")
    rotulo(ax, "(a) Detection by ERA5")
    ax.grid(axis="y", color=fp.GRID, lw=0.5)
    ax = axs[1]
    est = {"obs": ("Soundings", fp.WET, "-"), "era": ("ERA5", fp.DRY, "-"), "cf": ("ERA5 with 1980–95 detection skill", fp.INK2, "--")}
    for c, (lab, col, ls) in est.items():
        an = 100 * (g[c] - g.groupby("sid")[c].transform("mean"))
        y = an.groupby(g.temp).mean()
        ax.plot(y.index, y.values, "-", color=col, lw=0.6, alpha=0.45)
        ax.plot(y.index, y.rolling(5, center=True, min_periods=5).mean().values, ls, color=col, lw=1.7, label=lab)
    ax.axhline(0, color=fp.INK2, lw=0.6)
    ax.set_xlabel("Season")
    ax.set_ylabel("Jet frequency anomaly (pp);\nthin: seasonal, thick: 5-season mean")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    rotulo(ax, "(b) Same days, fixed levels")
    ax.grid(axis="y", color=fp.GRID, lw=0.5)
    ax = axs[2]
    col = "RMS todos los días"
    ax.bar(np.arange(len(M)), M[col], color=fp.NEUTRAL, yerr=[M[col] - M.RMS_lo, M.RMS_hi - M[col]], capsize=2, error_kw={"lw": 0.8})
    for i, v in enumerate(M[col]):
        ax.text(i, M.RMS_hi.iloc[i] + 0.06, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(M)), [p.replace("-", "–") for p in M.periodo], fontsize=6.5, rotation=20)
    ax.set_ylabel("RMS sounding − ERA5, 850 hPa (m s$^{-1}$)")
    rotulo(ax, "(c) Agreement")
    for a in axs:
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_habilidad_era5.png")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- Fig. 9
def fig9():
    m = pd.read_parquet("analisis/jet/13_mensual.parquet")
    m = m[m.nivel == "d925"].copy()
    m["temp"] = np.where(m.mes.dt.month >= 10, m.mes.dt.year + 1, m.mes.dt.year)
    q = pd.read_csv("analisis/jet/22_quiebres.csv")
    q = q[(q.nivel == 925) & (~q.regional)]
    grupos = (("(a) Argentina", {"ARM00087155": "Resistencia", "ARM00087576": "Ezeiza", "ARM00087623": "Santa Rosa", "ARM00087344": "Córdoba"}),
              ("(b) Brazil", {"BRM00083827": "Foz do Iguaçu", "BRM00083612": "Campo Grande", "BRM00083362": "Cuiabá", "BRM00083208": "Vilhena"}))
    colores = [fp.DRY, fp.WET, fp.INK, fp.NEUTRAL]
    mk = ["^", "s", "D", "o"]
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 5.8), sharex=True, sharey=True)
    for ax, (tit, est) in zip(axs, grupos):
        ax.axhline(0, color=fp.INK2, lw=0.6)
        for c, mm, (sid, nom) in zip(colores, mk, est.items()):
            x = m[m.sid == sid]
            g = x.groupby("temp").dif
            t = g.mean()[g.count() >= 3]
            t = t - t.loc[1993:2025].mean()
            ax.plot(t.index, t.values, "-" + mm, color=c, lw=1.1, ms=3, label=nom)
            for r in q[q.sid == sid].itertuples():
                f0 = pd.Period(r.fecha, "M"); xq = (f0.year + 1 if f0.month >= 10 else f0.year) + 0.18 * list(est).index(sid) - 0.3   # temporada (oct-mar) a la que pertenece la fecha del quiebre
                ax.annotate("", xy=(xq, 1.12), xytext=(xq, 1.28), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3))
        rotulo(ax, tit)
        ax.set_ylabel("Sounding − ERA5, 925-hPa wind speed\nanomaly (m s$^{-1}$)", fontsize=8)
        ax.grid(axis="y", color=fp.GRID, lw=0.5)
        ax.legend(fontsize=6.5, loc="lower right", frameon=False, ncol=2)
    axs[0].set_ylim(-2.7, 1.4)
    axs[-1].set_xlim(1991, 2026)
    axs[-1].set_xlabel("Season (year of March); arrows: breaks of each station (Sect. 4.4)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_departures_era5.png")
    plt.close(fig)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(f"{OUT}/fig*.png"):
        os.remove(f)
    fig1()
    fig2()
    if os.path.exists("analisis/jet/10_tendencias.csv"):
        fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    print("ok")
