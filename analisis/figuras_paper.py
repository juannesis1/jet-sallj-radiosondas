"""Figuras del paper, en inglés y con un estilo único.

Fig. 1  Mapas de tendencias nocturnas (T, Td, HR), oct-mar 1980-2025, estaciones homogeneizadas.
Fig. 2  Huella agrícola: (a) dispersión Δcultivos vs tendencia de Td oct-dic; (b) correlación por mes.
Fig. 3  Perfil vertical de la tendencia de q (radiosondeos 12 UTC).
Fig. 4  Huella temporal: estaciones de alta − baja expansión (dos paneles, sin doble eje).
Fig. S1 Curva de especificaciones: (a) ρ ordenados con control y placebo; (b) ρ por decisión.
Fig. 5  ERA5: campo de cambio de Td (2 m) y q (850 hPa) en oct-dic, con estaciones encima, y
        observación − reanálisis en estaciones.
Estilo: Arial; paletas científicas de Crameri (cmcrameri; Crameri et al. 2020), perceptualmente uniformes
y legibles con daltonismo: vik (temperatura), roma (humedad: marrón = más seco, azul = más húmedo).
Requiere: export SSL_CERT_FILE=$(.venv/bin/python -m certifi) para Cartopy.
"""
# Correspondencia función → figura del paper (tras renumerar las suplementarias por orden de cita, 3 de octubre):
#   fig1→Fig. 1, fig2→Fig. 2, fig3→Fig. 3, fig4→Fig. 5 (temporal), fig5→Fig. 4 (ERA5),
#   figS3→S1 (urbanización), figS4→S2 (estrés térmico), figS7→S3 (ERA5 circulación), figS5→S4 (descomposición),
#   figS1→S5 (curva de especificación), figS6→S6 (ciclo diurno), figS2→S7 (intensificación).
import importlib
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica import sen  # noqa: E402

OUT = "figuras/paper"
INK, INK2, GRID = "#1a1a1a", "#555555", "#e6e6e6"
CMAP_T, CMAP_HUM = cmc.vik, cmc.roma
DRY, WET, NEUTRAL = mpl.colors.to_hex(cmc.roma(0.12)), mpl.colors.to_hex(cmc.roma(0.88)), "#9a9a9a"
PROVINCIAS = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "50m",
                                          facecolor="none", edgecolor="#b8b8b8", linewidth=0.3)


def mapa_base(ax, extent=(-68, -53, -40, -23.5)):
    ax.set_extent(extent)
    ax.add_feature(cfeature.LAND, facecolor="#f6f6f4", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#e9eef2", zorder=0)
    ax.add_feature(PROVINCIAS, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=INK2, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=INK2, zorder=2)
mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight"})
NOMBRE = {"GENERAL URQUIZA": "Paraná", "COMANDANTE ESPORA": "Bahía Blanca", "SANTIAGO DEL ESTERO": "Santiago del Estero", "PASO DE LOS LIBRES": "Paso de los Libres", "MAR DEL PLATA": "Mar del Plata", "MARCOS_RC": "Marcos Juárez, Río Cuarto", "AMBROSIO L V TARAVELLA": "Córdoba", "COMODORO PIERRESTEGUI": "Concordia",
          "MINISTRO PISTARINI": "Ezeiza", "MARCOS JUAREZ": "Marcos Juárez", "JUNIN": "Junín",
          "POSADAS": "Posadas", "RIO CUARTO AREA DE MATERIAL": "Río Cuarto",
          "VILLA REYNOLDS": "Villa Reynolds", "FORMOSA": "Formosa"}


def estaciones():
    d = pd.read_csv("analisis/03_tendencias_anomalias_laxo_aj.csv")
    return d[(d.temporada == "ONDJFM") & d.incluida & ~d.nombre.str.contains("CERES")]


def fig1():
    d = estaciones()
    paneles = (("t", CMAP_T, 0.4, "(a) Temperature", "°C per decade"),
               ("td", CMAP_HUM, 0.4, "(b) Dew point", "°C per decade"),
               ("hr", CMAP_HUM, 2.0, "(c) Relative humidity", "pp per decade"))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.6), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, (v, cmap, lim, titulo, unidad) in zip(axs, paneles):
        mapa_base(ax)
        norma = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        sig = d[f"{v}_fdr"].fillna(False).astype(bool)
        col = cmap(norma(d[f"{v}_sen"].clip(-lim, lim)))
        ax.scatter(d.lon[sig], d.lat[sig], c=col[sig.values], s=46, edgecolors=INK, linewidths=0.5,
                   transform=ccrs.PlateCarree(), zorder=5)
        ax.scatter(d.lon[~sig], d.lat[~sig], facecolors="white", edgecolors="#9a9a9a", s=58,
                   linewidths=0.5, transform=ccrs.PlateCarree(), zorder=4)
        ax.scatter(d.lon[~sig], d.lat[~sig], facecolors="white", edgecolors=col[~sig.values], s=40,
                   linewidths=1.6, transform=ccrs.PlateCarree(), zorder=5)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norma, cmap=cmap), ax=ax, orientation="horizontal",
                          pad=0.07, shrink=0.9, aspect=22, extend="both")
        cb.set_label(unidad)
        cb.outline.set_visible(False)
        ax.set_title(titulo, loc="left")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=GRID, xlocs=[-65, -60, -55],
                          ylocs=[-25, -30, -35, -40])
        gl.top_labels = gl.right_labels = False
        gl.left_labels = ax is axs[0]
        gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": INK2}
    fig.savefig(f"{OUT}/fig1_trends_map.png")
    plt.close(fig)


def tendencias_ond():
    h = importlib.import_module("19_huella_temporal")
    h.DIR = "data/noches_ajustadas_pares_laxo"
    u = pd.read_csv("analisis/13_uso_suelo_verano_100km.csv")
    est = estaciones()
    filas = []
    for _, e in est[est.sid.str.startswith("87")].iterrows():
        fila = {"nombre": e.nombre.strip()}
        for vn, meses in (("ond", [10, 11, 12]), ("jfm", [1, 2, 3])):
            s = h.serie(e.sid, "td", meses)
            fila[vn] = sen(s.index.values.astype(float), s.values) * 10
        filas.append(fila)
    u["nombre"] = u.nombre.str.strip()
    return u.merge(pd.DataFrame(filas), on="nombre")


def fig2():
    """Huella agrícola: (a) Δcultivos vs tendencia de Td oct-dic; (b) ρ por mes para Td; (c) ρ por mes para NDVI;
    (d) tendencia de NDVI oct-dic vs tendencia de Td oct-dic."""
    u = tendencias_ond()
    m = pd.read_csv("analisis/14_uso_suelo_mensual.csv")
    nd = pd.read_csv("analisis/35_ndvi_tendencias.csv")
    sn = pd.read_csv("analisis/35_ndvi_series.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0, 0]
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.scatter(u.delta_soja_pp, u.jfm, s=20, facecolors="white", edgecolors=NEUTRAL, lw=1, zorder=2, label="Jan–Mar")
    ax.scatter(u.delta_soja_pp, u.ond, s=30, color=DRY, edgecolor="white", lw=0.6, zorder=3, label="Oct–Dec")
    b = np.polyfit(u.delta_soja_pp, u.ond, 1)
    xx = np.linspace(u.delta_soja_pp.min(), u.delta_soja_pp.max(), 50)
    ax.plot(xx, np.polyval(b, xx), color=DRY, lw=1, ls="--", zorder=2)
    rho, rj = stats.spearmanr(u.delta_soja_pp, u.ond)[0], stats.spearmanr(u.delta_soja_pp, u.jfm)[0]
    ax.text(0.98, 0.97, f"Oct–Dec: ρ = {rho:.2f}\nJan–Mar: ρ = {rj:.2f}".replace("-", "−"), transform=ax.transAxes,
            ha="right", va="top", fontsize=7)
    ax.set_xlabel("Change in summer-crop fraction within 100 km (pp)")
    ax.set_ylabel("Dew point trend (°C per decade)")
    ax.set_title("(a) Nocturnal dew point", loc="left")
    ax.legend(loc="lower left", fontsize=6.8, handletextpad=0.2)
    meses = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
    x = np.arange(6)
    ax = axs[0, 1]
    ax.bar(x, m.rho, width=0.6, color=[DRY if v < 0 else WET for v in m.rho], zorder=3)
    ax.set_title("(b) Dew point trend vs. crop expansion", loc="left", fontsize=8)
    axs[1, 1].set_title("(d) NDVI trend vs. crop expansion", loc="left", fontsize=8)
    rn = [stats.spearmanr(*nd[nd.mes == mm].dropna(subset=["dcult"])[["dcult", "tend"]].values.T)[0] for mm in (10, 11, 12, 1, 2, 3)]
    axs[1, 1].bar(x, rn, width=0.6, color=[DRY if v < 0 else WET for v in rn], zorder=3)
    for a in (axs[0, 1], axs[1, 1]):
        a.axhline(0, color=INK2, lw=0.7)
        a.set_xticks(x, meses)
        a.set_ylim(-1, 1)
        a.set_ylabel("Spearman ρ")
        a.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax = axs[1, 0]
    # Una sola definición (revisión I-1): tendencias de Sen 1982-2025 de Td y NDVI oct-dic en las mismas
    # temporadas, para las 23 estaciones (script 37). Argentinas coloreadas por expansión; UY/PY en blanco.
    td23 = pd.read_csv("analisis/37_resumen_estaciones.csv").rename(columns={"td": "ond", "cult": "delta_soja_pp"})
    q, extra = td23.dropna(subset=["delta_soja_pp"]), td23[td23.delta_soja_pp.isna()]
    ax.scatter(extra.ndvi, extra.ond, s=34, facecolors="white", edgecolors=INK2, lw=0.8, zorder=3,
               label="Uruguay, Paraguay")
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    sc = ax.scatter(q.ndvi, q.ond, c=q.delta_soja_pp, cmap=cmc.lajolla, s=34, edgecolor=INK2, lw=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, aspect=25)
    cb.set_label("Crop expansion (pp)", fontsize=7)
    cb.outline.set_visible(False)
    r, ra = stats.spearmanr(td23.ndvi, td23.ond)[0], stats.spearmanr(q.ndvi, q.ond)[0]
    ax.text(0.03, 0.97, f"ρ = {r:.2f} (all, n = {len(td23)})\nρ = {ra:.2f} (Argentina, n = {len(q)})".replace("-", "−"),
            transform=ax.transAxes, va="top", fontsize=7)
    ax.legend(loc="lower right", fontsize=6.5)
    ax.set_xlabel("Oct–Dec NDVI trend (per decade)")
    ax.set_ylabel("Oct–Dec dew point trend (°C per decade)")
    ax.set_title("(c) Spring greenness and dew point", loc="left")
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    fig.savefig(f"{OUT}/fig2_land_use.png")
    plt.close(fig)
    return u


def fig3():
    r = pd.read_csv("analisis/22_perfil_vertical_comun.csv")
    r = r[r["var"] == "q"]
    est = (("Córdoba", "+30", mpl.colors.to_hex(cmc.roma(0.05))), ("Santa Rosa", "+13", mpl.colors.to_hex(cmc.roma(0.33))),
           ("Ezeiza", "+10", NEUTRAL), ("Resistencia", "+1", mpl.colors.to_hex(cmc.roma(0.92))))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.3), sharey=True)
    for ax, v, tit in zip(axs, ("OND", "JFM"), ("(a) October–December", "(b) January–March")):
        for n, lab, c in est:
            s = r[(r.estacion == n) & (r.ventana == v)].sort_values("altura")
            ax.plot(s.sen_dec, s.altura, "-", color=c, lw=1.6, label=f"{n} ({lab} pp)")
            for _, q in s.iterrows():
                ax.plot(q.sen_dec, q.altura, "s" if q.altura == 0 else "o", ms=5.5, color=c,
                        mfc=c if q.p < 0.05 else "white", mew=1.3)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlim(-0.6, 0.5)
        ax.set_title(tit, loc="left")
        ax.set_xlabel("Specific humidity trend (g kg$^{-1}$ per decade)")
        ax.grid(color=GRID, lw=0.5)
    axs[0].set_yticks([0, 25, 50, 100, 150])
    axs[0].set_ylim(165, -8)
    axs[0].set_ylabel("Pressure above surface (hPa)")
    sec = axs[1].secondary_yaxis("right", functions=(lambda p: p * 9.6, lambda z: z / 9.6))
    sec.set_ylabel("Approx. height above ground (m)")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=7, title="Crop expansion within 100 km",
               title_fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(f"{OUT}/fig3_vertical_profile.png")
    plt.close(fig)


def fig4():
    d = pd.read_csv("analisis/19_diferencia_grupos.csv", index_col=0)
    suav = d.rolling(7, center=True, min_periods=7).mean()   # sin bordes truncados (revisión I-3)
    fig, axs = plt.subplots(2, 1, figsize=(6.0, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    for ax in axs:
        ax.axvspan(1996, 2010, color="#f2ead8", zorder=0)
        ax.grid(axis="y", color=GRID, lw=0.5)
    ax = axs[0]
    ax.plot(d.index, d.dcult, color=DRY, lw=1.8)
    ax.set_ylabel("Crop fraction\ndifference (pp)")
    ax.set_title("(a) Summer-crop fraction, high- minus low-expansion stations", loc="left")
    ax.text(2003, ax.get_ylim()[0] + 2, "rapid expansion", ha="center", fontsize=7, color=INK2)
    ax = axs[1]
    ax.axhline(0, color=INK2, lw=0.6)
    ax.plot(d.index, d.OND, "o", ms=2.8, color=DRY, alpha=0.45, mec="none")
    ax.plot(suav.index, suav.OND, color=DRY, lw=1.8, label="Oct–Dec (7-yr mean)")
    ax.plot(suav.index, suav.JFM, color=WET, lw=1.3, ls="--", label="Jan–Mar (7-yr mean)")
    ax.set_ylabel("Dew point difference (°C)")
    ax.set_title("(b) Nocturnal dew point, high- minus low-expansion stations", loc="left")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_xlim(1979, 2026)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_temporal.png")
    plt.close(fig)


def figS1():
    r = pd.read_csv("analisis/23_multiverso.csv")
    pr = r[r.ventana.isin(["OND", "ONDJFM"]) & (r.cultivo != "trigo")].sort_values("rho").reset_index(drop=True)
    jfm = np.sort(r[(r.ventana == "JFM") & (r.cultivo != "trigo")].rho.values)
    tri = np.sort(r[r.ventana.isin(["OND", "ONDJFM"]) & (r.cultivo == "trigo")].rho.values)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    q = np.linspace(0, 1, len(pr))
    ax.scatter(q, pr.rho, s=3, c=DRY, zorder=3)
    ax.plot(np.linspace(0, 1, len(jfm)), jfm, color=WET, lw=1.4, label="Control: Jan–Mar")
    ax.plot(np.linspace(0, 1, len(tri)), tri, color=NEUTRAL, lw=1.4, ls="--", label="Placebo: wheat")
    ax.scatter([], [], s=12, color=DRY, label="Main: Oct–Dec and Oct–Mar, summer crops")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("Specifications, ranked (quantile)")
    ax.set_ylabel("Spearman ρ, crop expansion vs. dew point trend")
    ax.set_title(f"(a) {len(pr)} specifications", loc="left")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax = axs[1]
    decis = [("datos", {"crudos": "raw", "homog": "homogenized"}, "Data"),
             ("ventana", {"OND": "Oct–Dec", "ONDJFM": "Oct–Mar"}, "Season"),
             ("estimador", {"sen": "Sen", "mco": "least squares"}, "Trend"),
             ("inicio", {1980: "1980", 1985: "1985"}, "Start"),
             ("radio", {50: "50 km", 100: "100 km", 150: "150 km"}, "Radius"),
             ("cultivo", {"verano": "summer crops", "soja": "soybean", "maiz": "maize"}, "Crop"),
             ("periodo", {"80-84→15-19": "1980–84→2015–19", "80-84→10-14": "1980–84→2010–14",
                          "85-89→15-19": "1985–89→2015–19"}, "Crop periods"),
             ("estaciones", {"todas": "all", "sin_urbanas": "no urban"}, "Stations")]
    y, ticks, labs = 0, [], []
    for col, opciones, grupo in decis:
        for val, lab in opciones.items():
            s = pr[pr[col] == val].rho
            ax.plot([s.quantile(0.05), s.quantile(0.95)], [y, y], color=DRY, lw=1.4, alpha=0.6)
            ax.plot(s.median(), y, "o", color=DRY, ms=4.5)
            ticks.append(y)
            labs.append(f"{grupo}: {lab}")
            y += 1
        y += 0.6
    ax.set_yticks(ticks, labs, fontsize=6.6)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axvline(pr.rho.median(), color=DRY, lw=0.6, ls=":")
    ax.set_xlim(-1, 0.2)
    ax.set_xlabel("ρ: median and 5–95 % range")
    ax.set_title("(b) By analytic choice", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/figS5_specification_curve.png")
    plt.close(fig)


def figS7():
    """ERA5 1980-2025, oct-dic: jet, transporte meridional de humedad a 850 hPa y q850 por región (script 41)."""
    d = pd.read_csv("analisis/41_series.csv", index_col="temporada").loc[1980:2025]
    suav = lambda s: s.rolling(7, center=True, min_periods=7).mean()
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
    ax = axs[0]
    ax.plot(d.index, d.jet_OND, color=INK2, lw=0.8, alpha=0.6)
    ax.plot(d.index, suav(d.jet_OND), color=INK, lw=1.6, label="Chaco jet days, ERA5")
    ax.plot(d.index, suav(d.jet_res_obs), color=DRY, lw=1.4, label="Resistencia, sondes")
    ax.plot(d.index, suav(d.jet_res_era5), color=DRY, lw=1.4, ls="--", label="Resistencia, ERA5")
    ax.set_ylabel("% of days")
    ax.set_ylim(0, 58)
    ax.set_title("(a) Low-level jet", loc="left", fontsize=8)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax = axs[1]
    for col, lab, c in (("flujo25", "25°S", WET), ("flujo30", "30°S", DRY)):
        ax.plot(d.index, d[col], color=c, lw=0.8, alpha=0.5)
        ax.plot(d.index, suav(d[col]), color=c, lw=1.6, label=lab)
    ax.set_ylabel("−qv (g kg$^{-1}$ m s$^{-1}$)")
    ax.set_title("(b) Southward moisture flux, 850 hPa", loc="left", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    ax = axs[2]
    for col, lab, c in (("q850_NE", "Northeast littoral", WET), ("q850_Pampa", "Intensified Pampas", DRY)):
        a = d[col] - d[col].loc[1981:2010].mean()
        ax.plot(d.index, a, color=c, lw=0.8, alpha=0.5)
        ax.plot(d.index, suav(a), color=c, lw=1.6, label=lab)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("q anomaly (g kg$^{-1}$)")
    ax.set_title("(c) 850-hPa humidity, 06 UTC", loc="left", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    for ax in axs:
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xlim(1979, 2026)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{OUT}/figS3_era5_circulation.png")
    plt.close(fig)



def figS2():
    r = pd.read_csv("analisis/26_intensificacion.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, col, xl, tit in ((axs[0], "d_agro", "Change in agricultural area (pp)", "(a) Agricultural area"),
                             (axs[1], "d_int", "Change in intensification index (pp)", "(b) Intensification")):
        q = r.dropna(subset=[col])
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.scatter(q[col], q.td_ond, s=34, color=DRY, edgecolor="white", lw=0.6, zorder=3)
        rho = stats.spearmanr(q[col], q.td_ond)[0]
        arriba = col == "d_int"
        ax.text(0.97, 0.96 if arriba else 0.04, f"ρ = {rho:.2f}, n = {len(q)}".replace("-", "−"),
                transform=ax.transAxes, ha="right", va="top" if arriba else "bottom", fontsize=7.5)
        for _, e in q.iterrows():
            if e.nombre in ("MARCOS JUAREZ", "AMBROSIO L V TARAVELLA", "RIO CUARTO AREA DE MATERIAL",
                            "COMODORO PIERRESTEGUI", "JUNIN"):
                izq = e.nombre == "MARCOS JUAREZ"
                ax.annotate(NOMBRE.get(e.nombre, e.nombre.title()), (e[col], e.td_ond),
                            xytext=(4, -9 if izq else 2), ha="left",
                            textcoords="offset points", fontsize=6.5, color=INK2)
        ax.set_xlabel(xl)
        ax.set_title(tit, loc="left")
    axs[0].set_ylabel("Oct–Dec dew point trend (°C per decade)")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/figS7_intensification.png")
    plt.close(fig)


def figS3():
    """Urbanización: crecimiento construido en 10 km vs tendencias de T nocturna y diurna."""
    e = pd.read_csv("analisis/18_urbano.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, col, tit in ((axs[0], "t_sen", "(a) Night (06–09 UTC)"), (axs[1], "t_dia_aj", "(b) Day (15–18 UTC)")):
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.scatter(e.db_10km, e[col], s=34, color=mpl.colors.to_hex(cmc.vik(0.85)), edgecolor="white", lw=0.6, zorder=3)
        ts = stats.theilslopes(e[col], e.db_10km)
        xx = np.linspace(0, e.db_10km.max(), 20)
        ax.plot(xx, ts.intercept + ts.slope * xx, color=mpl.colors.to_hex(cmc.vik(0.85)), lw=1, ls="--")
        rho = stats.spearmanr(e.db_10km, e[col])[0]
        ax.text(0.03, 0.96, f"ρ = {rho:.2f}".replace("-", "−"), transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_xlabel("Change in built-up fraction within 10 km,\n1975–2020 (percentage points)")
        ax.set_title(tit, loc="left")
    axs[0].set_ylabel("Temperature trend (°C per decade)")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/figS1_urbanization.png")
    plt.close(fig)


def figS4():
    """Estrés térmico nocturno: noches sobre el P90 local del heat index (tendencia por estación)."""
    r = pd.read_csv("analisis/17_indice_salud.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, (col, lim, tit, uni) in zip(axs, (("p90_hi_sen", 4, "(a) Nights above local 90th percentile", "pp per decade"),
                                                ("sin_alivio_sen", 4, "(b) Nights with heat index ≥ 25 °C", "pp per decade"))):
        mapa_base(ax)
        norma = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        sig = r[col.replace("_sen", "_fdr")].fillna(False).astype(bool)
        col_ = CMAP_T(norma(r[col].clip(-lim, lim)))
        ax.scatter(r.lon[sig], r.lat[sig], c=col_[sig.values], s=46, edgecolors=INK, linewidths=0.5,
                   transform=ccrs.PlateCarree(), zorder=5)
        ax.scatter(r.lon[~sig], r.lat[~sig], facecolors="white", edgecolors="#9a9a9a", s=58,
                   linewidths=0.5, transform=ccrs.PlateCarree(), zorder=4)
        ax.scatter(r.lon[~sig], r.lat[~sig], facecolors="white", edgecolors=col_[~sig.values], s=40,
                   linewidths=1.6, transform=ccrs.PlateCarree(), zorder=5)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norma, cmap=CMAP_T), ax=ax, orientation="horizontal",
                          pad=0.07, shrink=0.9, aspect=22, extend="both")
        cb.set_label(uni)
        cb.outline.set_visible(False)
        ax.set_title(tit, loc="left", fontsize=8.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=GRID, xlocs=[-65, -60, -55], ylocs=[-25, -30, -35, -40])
        gl.top_labels = gl.right_labels = False
        gl.left_labels = ax is axs[0]
        gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": INK2}
    fig.savefig(f"{OUT}/figS2_heat_stress.png")
    plt.close(fig)


def campo_ond(da):
    """Medias de oct-dic por temporada (oct-dic del año y-1 → temporada y), 1980-2025, de un campo ERA5."""
    da = da.sel(time=da.time.dt.month.isin([10, 11, 12]))
    da = da.assign_coords(year=("time", da.time.dt.year.values + 1)).groupby("year").mean("time")
    return da.sel(year=slice(1980, 2025))


def cambio(campo):
    """Tendencia lineal por década si hay ≥ 40 años; si no, diferencia 2016-24 menos 1979-87."""
    anios = campo.year.values
    if len(anios) >= 40:
        x = anios - anios.mean()
        anom = campo - campo.mean("year")
        pend = (anom * x[:, None, None]).sum("year") / (x ** 2).sum()
        resid = anom - pend * xr.DataArray(x, dims="year")
        ee = np.sqrt((resid ** 2).sum("year") / (len(x) - 2) / (x ** 2).sum())
        cambio.t = abs(pend / ee)                       # t de MCO (sin corrección por autocorrelación)
        valido = campo.notnull().sum("year") >= 40       # .sum() omite NaN: sin esto el terreno enmascarado daría 0
        pend, cambio.t = pend.where(valido), cambio.t.where(valido)
        return pend * 10, f"trend {anios.min()}–{anios.max()} (per decade)"
    tarde = campo.sel(year=slice(2016, 2024)).mean("year")
    temprano = campo.sel(year=slice(1979, 1987)).mean("year")
    return tarde - temprano, "change 2016–24 minus 1979–87"


def fig5():
    import glob
    import xarray as xr
    sfc = xr.open_mfdataset(sorted(glob.glob("data/era5/sfc_*.nc")), combine="by_coords").rename({"valid_time": "time"})
    import era5io
    pl = era5io.abrir_pl()
    td = campo_ond(sfc.d2m - 273.15).compute()
    q8 = campo_ond(pl.q.sel(level=850).where(pl.time.dt.hour == 6, drop=True) * 1000).compute()
    # enmascarar terreno alto: a 850 hPa ERA5 extrapola bajo tierra si la presión de superficie < 870 hPa;
    # el mismo umbral se aplica al campo de 2 m para centrar la escala en la llanura
    sp = (sfc.sp.isel(time=slice(0, 400)).mean("time") / 100).compute()
    td = td.where(sp >= 870)
    q8 = q8.where(sp.interp(latitude=q8.latitude, longitude=q8.longitude) >= 900)   # 850 hPa al menos 50 hPa sobre el suelo
    dtd, etiqueta = cambio(td)
    t_td = cambio.t.where(td.isel(year=0).notnull())
    dq8, _ = cambio(q8)
    t_q8 = cambio.t.where(q8.isel(year=0).notnull())
    r = pd.read_csv("analisis/21_omr_101112.csv")
    dq8 = dq8.where(np.isfinite(dq8))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.7), subplot_kw={"projection": ccrs.PlateCarree()})
    paneles = ((dtd, 0.4, "(a) 2-m dew point, Oct–Dec", "°C per decade", "obs"),
               (dq8, 0.3, "(b) 850-hPa specific humidity", "g kg$^{-1}$ per decade", None),
               (None, 0.4, "(c) Station minus ERA5", "°C per decade", "omr"))
    signif = {0: t_td, 1: t_q8}
    for k, (ax, (campo, lim, tit, uni, col)) in enumerate(zip(axs, paneles)):
        mapa_base(ax)
        norma = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        if campo is not None:
            campo.plot.pcolormesh(ax=ax, x="longitude", y="latitude", cmap=CMAP_HUM.with_extremes(bad=(0, 0, 0, 0)), norm=norma,
                                  add_colorbar=False, transform=ccrs.PlateCarree(), zorder=1, rasterized=True)
            tt = signif[k]
            paso = 4 if k == 0 else 2                     # ~1° entre puntos
            tt = tt.isel(latitude=slice(None, None, paso), longitude=slice(None, None, paso))
            yy, xx = np.meshgrid(tt.latitude.values, tt.longitude.values, indexing="ij")
            m = (tt > 2.02).values
            ax.scatter(xx[m], yy[m], s=0.6, color=INK2, transform=ccrs.PlateCarree(), zorder=1.2, lw=0)
            ax.add_feature(cfeature.OCEAN, facecolor="#e9eef2", zorder=1.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=INK2, zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=INK2, zorder=2)
        if col is not None:
            ax.scatter(r.lon, r.lat, c=r[col].clip(-lim, lim), cmap=CMAP_HUM, norm=norma, s=42,
                       edgecolors=INK, linewidths=0.6, transform=ccrs.PlateCarree(), zorder=5)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norma, cmap=CMAP_HUM), ax=ax, orientation="horizontal",
                          pad=0.07, shrink=0.9, aspect=22, extend="both")
        cb.set_label(uni)
        cb.outline.set_visible(False)
        ax.set_title("", loc="center")
        ax.set_title(tit, loc="left", fontsize=8)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=GRID, xlocs=[-65, -60, -55], ylocs=[-25, -30, -35, -40])
        gl.top_labels = gl.right_labels = False
        gl.left_labels = ax is axs[0]
        gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": INK2}
    print("Fig. 5:", etiqueta)
    fig.savefig(f"{OUT}/fig4_era5.png")
    plt.close(fig)


def figS5():
    """Descomposición del cambio de Td oct-dic (1980-2002 → 2003-2025) con tipos de circulación observados."""
    r = pd.read_csv("analisis/12b_descomposicion_igra.csv")
    r = r[(r.periodos == "1980-2002 vs 2003-2025") & (r.ventana == "OND")].dropna(subset=["delta_soja_pp"])
    r = r.sort_values("delta_soja_pp").reset_index(drop=True)
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.axvline(0, color=INK2, lw=0.6)
    ax.barh(y + 0.2, r.frecuencia, height=0.38, color=mpl.colors.to_hex(cmc.roma(0.75)), label="Change in type frequency")
    ax.barh(y - 0.2, r.dentro, height=0.38, color=DRY, label="Change within types")
    ax.errorbar(r.dentro, y - 0.2, xerr=[r.dentro - r.dentro_lo, r.dentro_hi - r.dentro], fmt="none",
                ecolor=INK2, elinewidth=0.6, capsize=1.5)
    etq = lambda c: "0" if round(c) == 0 else f"{round(c):+d}".replace("-", "−")
    ax.set_yticks(y, [f"{NOMBRE.get(n, n.title())}  ({etq(c)})" for n, c in zip(r.nombre, r.delta_soja_pp)], fontsize=7)
    ax.set_xlabel("Contribution to Oct–Dec dew point change, 1980–2002 to 2003–2025 (°C)")
    ax.set_ylabel("Station (crop expansion, percentage points)")
    ax.legend(loc="lower left", fontsize=7)
    ax.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figS4_decomposition.png")
    plt.close(fig)


def figS6():
    """Ciclo diurno de la huella agrícola: correlación parcial y contraste por hora, oct-dic y oct-mar."""
    def parcial(x, y, Z):
        X = np.c_[np.ones(len(x)), Z]
        rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
        ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        return stats.pearsonr(rx, ry)[0]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    horas = [0, 3, 6, 9, 12, 15, 18, 21]
    for suf, lab, c in (("", "Oct–Dec", DRY), ("_ondjfm", "Oct–Mar", mpl.colors.to_hex(cmc.roma(0.8)))):
        r = pd.read_csv(f"analisis/29_ciclo_diurno{suf}.csv")
        r = r[r.hora.isin([str(h) for h in horas])].dropna(subset=["td_sen", "delta_soja_pp"])
        par, con = [], []
        for h in horas:
            q = r[r.hora == str(h)]
            par.append(parcial(q.delta_soja_pp.values, q.td_sen.values, np.c_[q.lat, q.lon]))
            con.append(q[q.delta_soja_pp > 20].td_sen.mean() - q[q.delta_soja_pp < 5].td_sen.mean())
        local = [(h - 3) % 24 for h in horas]
        orden = np.argsort(local)
        axs[0].plot(np.array(local)[orden], np.array(par)[orden], "o-", color=c, lw=1.6, ms=4, label=lab)
        axs[1].plot(np.array(local)[orden], np.array(con)[orden], "o-", color=c, lw=1.6, ms=4, label=lab)
    for ax, tit, yl in ((axs[0], "(a) Partial correlation (lat, lon)", "Correlation with crop expansion"),
                        (axs[1], "(b) High- minus low-expansion stations", "Dew point trend contrast (°C per decade)")):
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvspan(0, 6, color="#eeeeee", zorder=0)
        ax.axvspan(19.5, 24, color="#eeeeee", zorder=0)
        ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
        ax.set_xlabel("Local time (h)")
        ax.set_ylabel(yl)
        ax.set_title(tit, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
    axs[0].set_ylim(-1, 0.1)
    axs[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout(w_pad=2)
    fig.savefig(f"{OUT}/figS6_diurnal.png")
    plt.close(fig)



if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    figS1()
    figS2()
    figS3()
    figS4()
    fig5()
    figS5()
    figS6()
    figS7()
    print("ok")
