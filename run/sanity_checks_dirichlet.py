import numpy as np
import statsmodels.api as sm
from numpy.testing import assert_array_less, assert_allclose
from model_metrics import compute_aic, compute_bic, compute_pseudo_r2, compute_r2


def check_dirichlet_mle_estimator(dirichlet_fit_func, alpha_true, n: int = 5000, tol: float = 0.1) -> None:
    """
    Simulerer data fra Dirichlet(alpha_true), fitter med gitt MLE-funksjon,
    og sjekker at relativ feil per komponent er < tol.
    """
    alpha_true = np.asarray(alpha_true, dtype=float)
    assert alpha_true.ndim == 1, "alpha_true må være 1D"
    rng = np.random.default_rng(12345)
    X = rng.dirichlet(alpha_true, size=n)

    # Kjør MLE-funksjon; håndter tuple (alpha, method) eller bare alpha.
    fit = dirichlet_fit_func(X)
    if isinstance(fit, tuple):
        alpha_hat = np.asarray(fit[0], dtype=float)
    else:
        alpha_hat = np.asarray(fit, dtype=float)

    assert alpha_hat.shape == alpha_true.shape, "alpha_hat og alpha_true må ha samme shape"

    rel_err = np.abs(alpha_hat - alpha_true) / alpha_true
    assert_array_less(rel_err, tol, err_msg=f"Relativ feil > {tol}: {rel_err}")


def check_aic_bic_monotonic(ll_better: float, ll_worse: float, k: int, n: int) -> None:
    """
    Sjekker at for samme k gir høyere log-likelihood lavere AIC/BIC,
    og for lik LL men høyere k blir AIC/BIC høyere (bruker compute_aic/bic).
    """
    aic_better = compute_aic(ll_better, k)
    bic_better = compute_bic(ll_better, k, n)
    aic_worse  = compute_aic(ll_worse, k)
    bic_worse  = compute_bic(ll_worse, k, n)

    assert aic_better < aic_worse, "AIC ble ikke lavere når LL ble bedre"
    assert bic_better < bic_worse, "BIC ble ikke lavere når LL ble bedre"

    k2 = k + 1
    aic_more_params = compute_aic(ll_better, k2)
    bic_more_params = compute_bic(ll_better, k2, n)
    assert aic_more_params > aic_better, "AIC ble ikke høyere med flere parametre (samme LL)"
    assert bic_more_params > bic_better, "BIC ble ikke høyere med flere parametre (samme LL)"


def check_pseudo_r2_properties(ll_full: float, ll_null: float, tol: float = 1e-9) -> None:
    """
    Sjekker compute_pseudo_r2:
    - i [0,1] med toleranse
    - er 0 når ll_full == ll_null
    - blir høy når ll_full >> ll_null (bedre modell).
    """
    if ll_null == 0:
        raise ValueError("ll_null kan ikke være 0 i denne testen")
    pseudo_r2 = compute_pseudo_r2(ll_full, ll_null)

    assert pseudo_r2 <= 1 + tol and pseudo_r2 >= -tol, "pseudo-R² ligger ikke i [0,1]"

    if np.isclose(ll_full, ll_null, atol=tol):
        assert abs(pseudo_r2) <= tol, "pseudo-R² skal være 0 når ll_full == ll_null"

    # Når full-modell er mye bedre (ll_null << ll_full for negative LL), skal pseudo_R2 være høy
    # eksempel: ll_null = -100, ll_full = -10 => 1 - (-10/-100) = 0.9
    ll_null_far = -100.0
    ll_full_far = -10.0
    pseudo_high = compute_pseudo_r2(ll_full_far, ll_null_far)
    assert pseudo_high > 0.5, "pseudo-R² blir ikke høy når full-modellen er mye bedre"


def check_r2_against_statsmodels(n: int = 200, noise: float = 0.1, tol: float = 1e-8) -> None:
    """
    Simulerer y = 2x + 1 + støy, fitter OLS, og sjekker at compute_r2 == statsmodels rsquared.
    """
    rng = np.random.default_rng(123)
    x = rng.normal(size=n)
    y = 2.0 * x + 1.0 + rng.normal(scale=noise, size=n)

    X_design = sm.add_constant(x)
    model = sm.OLS(y, X_design).fit()
    y_hat = model.fittedvalues

    r2_model = model.rsquared
    r2_custom = compute_r2(y, y_hat)

    assert_allclose(r2_custom, r2_model, atol=tol, rtol=0.0)
