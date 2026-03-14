"""
Rask smoke-test av ILR/CLR, Dirichlet-MLE og en enkel ILR-regresjon.
- Genererer syntetiske komposisjoner (4 deler) fra kjent Dirichlet.
- Kjører MLE, ILR-transformasjon og en liten OLS-regresjon på ILR-koordinater.
- Kaller sanity-sjekkene fra sanity_checks_ilr og sanity_checks_dirichlet.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from sanity_checks_ilr import (
    check_clr_rowsum_zero,
    check_psi_orthonormal,
    check_ilr_roundtrip,
)
from sanity_checks_dirichlet import (
    check_dirichlet_mle_estimator,
    check_aic_bic_monotonic,
    check_pseudo_r2_properties,
)
from sanity_checks_model_metrics import (
    check_compute_aic_basic,
    check_compute_bic_basic,
    check_compute_pseudo_r2_properties,
    check_compute_r2_against_manual,
)

try:
    from dirichlet import dirichlet as _dir_pkg  # type: ignore
except Exception as e:  # pragma: no cover - fallback
    raise RuntimeError("PyPI-pakken 'dirichlet' må være installert for denne testen.") from e


# Pivot-Ψ (p1 vs {p2,p3,p4}, p2 vs {p3,p4}, p3 vs p4)
PSI = np.array([
    [ np.sqrt(3/4),         0.0,                 0.0               ],
    [-1/np.sqrt(12),  np.sqrt(2/3),              0.0               ],
    [-1/np.sqrt(12), -1/np.sqrt(6),        1/np.sqrt(2)           ],
    [-1/np.sqrt(12), -1/np.sqrt(6),       -1/np.sqrt(2)           ],
])


def closure(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.asarray(P, float)
    P[P <= 0] = eps
    return P / P.sum(axis=1, keepdims=True)


def clr(P: np.ndarray) -> np.ndarray:
    P = closure(P)
    L = np.log(P)
    return L - L.mean(axis=1, keepdims=True)


def ilr(P: np.ndarray, psi: np.ndarray = PSI) -> np.ndarray:
    return clr(P) @ psi


def ilr_inv(Z: np.ndarray, psi: np.ndarray = PSI) -> np.ndarray:
    clr_mat = np.asarray(Z) @ psi.T
    U = np.exp(clr_mat)
    return U / U.sum(axis=1, keepdims=True)


def dirichlet_mle_fixedpoint(X: np.ndarray):
    """Wrapper rundt dirichlet.mle (fixedpoint) for bruk i sanity-sjekken."""
    return _dir_pkg.mle(X, method="fixedpoint")


def main():
    rng = np.random.default_rng(42)

    # 1) Syntetiske komposisjoner med kjent alpha
    alpha_true = np.array([5.0, 3.0, 2.0, 1.5])
    X = rng.dirichlet(alpha_true, size=800)

    # 2) ILR / CLR
    X_clr = clr(X)
    Z = ilr(X)
    X_back = ilr_inv(Z)

    # 3) En enkel ILR-regresjon: ilr_k ~ ilr_(andre) + konstant
    Z_df = Z
    y = Z_df[:, 0]
    X_reg = sm.add_constant(Z_df[:, 1:])
    model = sm.OLS(y, X_reg).fit()
    _ = model.params  # ikke brukt videre, men sikrer at fit virker

    # 4) Kjør sanity-sjekker
    check_clr_rowsum_zero(X_clr)
    check_psi_orthonormal(PSI)
    check_ilr_roundtrip(X, ilr, ilr_inv, PSI)

    check_dirichlet_mle_estimator(dirichlet_mle_fixedpoint, alpha_true, n=3000, tol=0.15)

    # AIC/BIC monotoni: sett opp enkle LL-verdier
    ll_better, ll_worse, k, n = -100.0, -150.0, 6, 500
    check_aic_bic_monotonic(ll_better, ll_worse, k, n)

    # pseudo-R² egenskaper
    check_pseudo_r2_properties(ll_full=-80.0, ll_null=-80.0)
    check_pseudo_r2_properties(ll_full=-20.0, ll_null=-100.0)

    # model_metrics spesifikke sanity-checker
    check_compute_aic_basic()
    check_compute_bic_basic()
    check_compute_pseudo_r2_properties()
    check_compute_r2_against_manual()

    print("Alle konsistenskontroller bestått")


if __name__ == "__main__":
    main()
