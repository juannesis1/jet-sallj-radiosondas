"""Tendencias: pendiente de Sen, Mann-Kendall modificado (Hamed & Rao 1998) y FDR (Wilks 2016)."""
import numpy as np
from scipy import stats


def sen(x, y):
    ok = np.isfinite(y)
    return stats.theilslopes(y[ok], x[ok])[0] if ok.sum() > 2 else np.nan


def mk_hamed_rao(x, y, alfa_acf=0.05):
    """Mann-Kendall con corrección de varianza por autocorrelación (Hamed & Rao 1998).

    La autocorrelación se estima sobre los rangos de la serie sin tendencia (restando la
    pendiente de Sen) y solo se usan los rezagos significativos (±1.96/sqrt(n)).
    Devuelve (S, p_valor).
    """
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(y)
    if n < 8:
        return np.nan, np.nan
    s = 0.0
    for k in range(n - 1):
        s += np.sign(y[k + 1:] - y[k]).sum()
    # varianza con empates
    _, cuentas = np.unique(y, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(cuentas * (cuentas - 1) * (2 * cuentas + 5))) / 18
    # corrección por autocorrelación de los rangos de la serie sin tendencia
    b = sen(x, y)
    r = stats.rankdata(y - b * x)
    r = r - r.mean()
    den = np.sum(r * r)
    corr = 0.0
    for k in range(1, n - 1):
        rk = np.sum(r[:-k] * r[k:]) / den
        if abs(rk) > stats.norm.ppf(1 - alfa_acf / 2) / np.sqrt(n):
            corr += (n - k) * (n - k - 1) * (n - k - 2) * rk
    factor = 1 + 2 / (n * (n - 1) * (n - 2)) * corr
    var_s *= max(factor, 1e-6)
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    return s, 2 * (1 - stats.norm.cdf(abs(z)))


def fdr(p, alfa=0.10):
    """Benjamini-Hochberg: devuelve un array booleano de rechazos (Wilks 2016, α_FDR=0.10)."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    rech = np.zeros_like(p, bool)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return rech
    orden = np.argsort(pv)
    umbral = alfa * np.arange(1, m + 1) / m
    pasa = pv[orden] <= umbral
    if pasa.any():
        kmax = np.max(np.where(pasa)[0])
        idx = np.where(ok)[0][orden[:kmax + 1]]
        rech[idx] = True
    return rech


def parcial(x, y, Z):
    """Correlación parcial de Pearson de x e y controlando por las columnas de Z, con su p bilateral.

    Usa n − 2 − k grados de libertad (k = número de covariables); pearsonr sobre los residuos usaría n − 2.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(Z, float).reshape(len(x), -1)
    X = np.c_[np.ones(len(x)), Z]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r = stats.pearsonr(rx, ry)[0]
    gl = len(x) - 2 - Z.shape[1]
    return r, 2 * stats.t.sf(abs(r) * np.sqrt(gl / (1 - r ** 2)), gl)
