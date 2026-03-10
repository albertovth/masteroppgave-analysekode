import numpy as np
from numpy.testing import assert_allclose


def check_clr_rowsum_zero(X_clr: np.ndarray, tol: float = 1e-10) -> None:
    """Sjekker at CLR-matrisen har radsum ≈ 0 (innen tol)."""
    X_clr = np.asarray(X_clr, float)
    assert X_clr.ndim == 2, "X_clr må være 2D"
    row_sums = X_clr.sum(axis=1)
    assert_allclose(row_sums, 0.0, atol=tol, rtol=0.0)


def check_psi_orthonormal(Psi: np.ndarray, tol: float = 1e-10) -> None:
    """Sjekker ortonormalitet: Psi.T @ Psi ≈ I (innen tol)."""
    Psi = np.asarray(Psi, float)
    gram = Psi.T @ Psi
    I = np.eye(gram.shape[0])
    assert_allclose(gram, I, atol=tol, rtol=0.0)


def check_ilr_roundtrip(
    X: np.ndarray,
    ilr_func,
    ilr_inv_func,
    Psi: np.ndarray,
    tol: float = 1e-8,
) -> None:
    """
    Rundturstest: X -> ILR -> X_hat og sjekk at X_hat ≈ X innen tol.
    Forventer X som (n,k) med radsum>0; ilr_inv_func skal bruke samme Psi/basis.
    """
    X = np.asarray(X, float)
    assert X.ndim == 2, "X må være 2D"
    # sikker closure
    X[X <= 0] = np.finfo(float).tiny
    X = X / X.sum(axis=1, keepdims=True)
    Z = ilr_func(X, psi=Psi) if "psi" in ilr_func.__code__.co_varnames else ilr_func(X)
    X_hat = ilr_inv_func(Z, psi=Psi) if "psi" in ilr_inv_func.__code__.co_varnames else ilr_inv_func(Z)
    assert_allclose(X_hat, X, atol=tol, rtol=0.0)
