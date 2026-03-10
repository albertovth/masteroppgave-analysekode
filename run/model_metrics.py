import numpy as np


def compute_aic(ll: float, k: int) -> float:
    """AIC = 2k - 2LL."""
    return 2 * k - 2 * ll if np.isfinite(ll) else np.nan


def compute_bic(ll: float, k: int, n: int) -> float:
    """BIC = k*ln(n) - 2LL."""
    if not (np.isfinite(ll) and n > 0):
        return np.nan
    return k * np.log(n) - 2 * ll


def compute_pseudo_r2(ll_full: float, ll_null: float) -> float:
    """McFadden-lignende pseudo-R²: 1 - (LL_full / LL_null)."""
    if ll_null == 0:
        return np.nan
    return 1.0 - (ll_full / ll_null)


def compute_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """R² = 1 - RSS/TSS."""
    y = np.asarray(y, float)
    y_hat = np.asarray(y_hat, float)
    rss = np.nansum((y - y_hat) ** 2)
    tss = np.nansum((y - np.nanmean(y)) ** 2)
    return np.nan if tss == 0 else 1.0 - rss / tss
