import numpy as np
from numpy.testing import assert_allclose

from model_metrics import compute_aic, compute_bic, compute_pseudo_r2, compute_r2


def check_compute_aic_basic():
    """Sjekk at compute_aic implementerer AIC = 2k - 2LL."""
    ll = -123.456
    k = 7
    expected = 2 * k - 2 * ll
    got = compute_aic(ll, k)
    assert_allclose(got, expected)


def check_compute_bic_basic():
    """Sjekk at compute_bic implementerer BIC = k*ln(n) - 2LL."""
    ll = -123.456
    k = 7
    n = 250
    expected = k * np.log(n) - 2 * ll
    got = compute_bic(ll, k, n)
    assert_allclose(got, expected)


def check_compute_pseudo_r2_properties():
    """Sjekk formel og egenskaper for pseudo-R²."""
    ll_null = -100.0
    ll_full = -80.0
    r2 = compute_pseudo_r2(ll_full, ll_null)
    expected = 1.0 - (ll_full / ll_null)
    assert_allclose(r2, expected)
    assert 0.0 <= r2 <= 1.0

    # Når full-modellen er lik null-modellen -> 0
    r2_same = compute_pseudo_r2(-80.0, -80.0)
    assert abs(r2_same) < 1e-12


def check_compute_r2_against_manual():
    """Sammenlign compute_r2 mot manuelt beregnet R²."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    beta0, beta1 = 1.5, -0.7
    eps = rng.normal(scale=0.5, size=200)
    y = beta0 + beta1 * x + eps

    # "Perfekt" modell: bruk lukket form OLS
    X_design = np.column_stack([np.ones_like(x), x])
    beta_hat, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta_hat

    r2 = compute_r2(y, y_hat)

    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    r2_expected = 1.0 - rss / tss
    assert_allclose(r2, r2_expected)


if __name__ == "__main__":
    check_compute_aic_basic()
    check_compute_bic_basic()
    check_compute_pseudo_r2_properties()
    check_compute_r2_against_manual()
    print("Alle konsistenskontroller for model_metrics bestått")
