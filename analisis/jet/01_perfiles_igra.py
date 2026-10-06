"""Paper 2, paso 1: estructura del jet de bajos niveles en los radiosondeos IGRA v2 de Sudamérica (1979-2025).

Para cada sondeo (todas las horas y meses) se usa el perfil vertical completo (niveles obligatorios y significativos),
como Oliveira et al. (2018) y Sasaki et al. (2022), en lugar de dos niveles fijos:
  - capa de búsqueda del núcleo: desde la superficie hasta 250 hPa por encima de ella (~0-3 km);
  - núcleo = máximo de velocidad en esa capa; caída = núcleo menos el mínimo de velocidad entre el núcleo y 300 hPa
    por encima de la superficie (~3 km);
  - jet (criterio 1 de Bonner, 1968, como en Oliveira et al. 2018): núcleo ≥ 12 m/s y caída ≥ 6 m/s;
    jet del norte si la dirección en el núcleo está entre 292.5° y 67.5° (NO a NE).
Además, el mismo criterio aplicado solo a los niveles estándar (superficie, 1000, 925, 850, 700 y 500 hPa), comunes a
todo el período (jet_norte_std), y el código de origen de cada sondeo (fuente). Se guardan también u, v a 925, 850 y 700 hPa y q a 850 hPa (para comparar con los reanálisis en niveles fijos).
La altura se expresa como presión sobre la superficie (hPa), porque los sondeos recientes del GTS no traen altura
geopotencial en los niveles significativos (igual que en el paper 1).
Salida: data/igra_sa/perfiles/<estación>.parquet y analisis/jet/01_resumen.txt
"""
import glob
import io
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

CAPA_NUCLEO, CAPA_CAIDA = 250.0, 300.0          # hPa sobre la superficie (~2.5 y ~3 km; caída hasta 3 km como Bonner 1968)
MIN_NUCLEO, MIN_CAIDA = 12.0, 6.0               # m/s (Bonner 1968, criterio 1)


def num(s, escala=1.0):
    s = s.strip().rstrip("ABab")
    if not s or s.startswith("-9999") or s.startswith("-8888"):
        return np.nan
    return float(s) / escala


def analizar(cab, niveles):
    """niveles: lista de (p_hPa, wdir, wspd, temp, dpd, tipo_superficie)."""
    a = np.array([n[:5] for n in niveles], float)
    sup = [n for n in niveles if n[5]]
    p_sup = sup[0][0] if sup else np.nanmax(a[:, 0])
    ok = np.isfinite(a[:, 0]) & np.isfinite(a[:, 2]) & np.isfinite(a[:, 1])
    a = a[ok]
    fila = {"fecha": cab, "p_sup": p_sup}
    if len(a):
        dp = p_sup - a[:, 0]
        bajo = (dp >= 0) & (dp <= CAPA_NUCLEO)
        if bajo.sum() >= 3:
            i = np.where(bajo)[0][np.argmax(a[bajo, 2])]
            arriba = (dp > dp[i]) & (dp <= CAPA_CAIDA)
            caida = a[i, 2] - a[arriba, 2].min() if arriba.any() else np.nan
            d = a[i, 1]
            fila.update(nucleo_v=a[i, 2], nucleo_dp=dp[i], nucleo_dir=d, caida=caida,
                        n_niveles=int(bajo.sum()), tope=float(dp.max()))
            fila["jet"] = bool(a[i, 2] >= MIN_NUCLEO and caida >= MIN_CAIDA) if np.isfinite(caida) else np.nan
            fila["jet_norte"] = bool(fila["jet"] and (d >= 292.5 or d <= 67.5)) if fila["jet"] == fila["jet"] else np.nan
    est = np.isin(np.round(a[:, 0]), (1000.0, 925.0, 850.0, 700.0, 500.0)) | np.isclose(a[:, 0], p_sup) if len(a) else []
    if len(a) and np.any(est):
        e = a[est]
        dp = p_sup - e[:, 0]
        bajo = (dp >= 0) & (dp <= CAPA_NUCLEO)
        if bajo.sum() >= 2:
            i = np.where(bajo)[0][np.argmax(e[bajo, 2])]
            arriba = (dp > dp[i]) & (dp <= CAPA_CAIDA)
            if arriba.any():
                caida = e[i, 2] - e[arriba, 2].min()
                d = e[i, 1]
                fila["jet_norte_std"] = bool(e[i, 2] >= MIN_NUCLEO and caida >= MIN_CAIDA and (d >= 292.5 or d <= 67.5))
    for p in (925.0, 850.0, 700.0):
        k = np.where(np.isclose(a[:, 0], p))[0] if len(a) else []
        if len(k):
            sp, di = a[k[0], 2], np.radians(a[k[0], 1])
            fila[f"u{int(p)}"], fila[f"v{int(p)}"] = -sp * np.sin(di), -sp * np.cos(di)
    t = [n for n in niveles if np.isclose(n[0], 850.0) and np.isfinite(n[3]) and np.isfinite(n[4])]
    if t:
        td = t[0][3] - t[0][4]
        e = 6.112 * np.exp(17.67 * td / (td + 243.5))
        fila["q850"] = 1000 * 0.622 * e / (850 - 0.378 * e)
    return fila


def leer(zip_path):
    sid = os.path.basename(zip_path)[:11]
    dest = f"data/igra_sa/perfiles/{sid}.parquet"
    if os.path.exists(dest):
        return sid, len(pd.read_parquet(dest))
    filas, cab, niveles, fuente = [], None, [], ""
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as fh:
        for line in io.TextIOWrapper(fh, encoding="ascii", errors="ignore"):
            if line.startswith("#"):
                if cab is not None and niveles:
                    filas.append({**analizar(cab, niveles), "fuente": fuente})
                y, m, d, h = int(line[13:17]), int(line[18:20]), int(line[21:23]), int(line[24:26])
                cab = pd.Timestamp(y, m, d, h if h < 24 else 0) if 1979 <= y <= 2025 and h != 99 else None
                fuente = line[37:45].strip()
                niveles = []
                continue
            if cab is None:
                continue
            p = num(line[9:15], 100.0)
            if not np.isfinite(p):
                continue
            wd_, ws_ = num(line[40:45]), num(line[46:51], 10.0)
            if np.isfinite(ws_) and ws_ > 80.0:     # QC: viento implausible (> 80 m/s)
                wd_ = ws_ = np.nan
            niveles.append((p, wd_, ws_, num(line[22:27], 10.0),
                            num(line[34:39], 10.0), line[1] == "1"))
    if cab is not None and niveles:
        filas.append({**analizar(cab, niveles), "fuente": fuente})
    df = pd.DataFrame(filas).set_index("fecha").sort_index(kind="stable")
    df = df[~df.index.duplicated(keep="first")]
    df.to_parquet(dest)
    return sid, len(df)


def main():
    os.makedirs("data/igra_sa/perfiles", exist_ok=True)
    zips = sorted(glob.glob("data/igra_sa/*-data.txt.zip"))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(leer, zips))
    # coordenadas de la lista oficial de IGRA (incluye las 6 estaciones del norte que no están en estaciones.txt)
    est = pd.read_fwf("data/igra/igra2-station-list.txt", colspecs=[(0, 11), (12, 20), (21, 30)], header=None,
                      names=["sid", "lat", "lon"]).set_index("sid")
    L = ["Estación          lat     lon    sondeos  00UTC  12UTC  %jet_norte(oct-mar,12UTC)  núcleo_dp mediana"]
    for sid, n in res:
        d = pd.read_parquet(f"data/igra_sa/perfiles/{sid}.parquet")
        h = d.index.hour
        cal = d[d.index.month.isin([10, 11, 12, 1, 2, 3]) & (h == 12)]
        L.append(f"{sid} {est.lat[sid]:7.2f} {est.lon[sid]:7.2f} {n:8d} {(h == 0).sum():6d} {(h == 12).sum():6d} "
                 f"{100 * cal.jet_norte.astype(float).mean():8.1f} {cal.loc[cal.jet_norte == True, 'nucleo_dp'].median():12.0f}")
    open("analisis/jet/01_resumen.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
