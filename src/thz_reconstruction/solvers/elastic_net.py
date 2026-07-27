from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator


def _soft_threshold(
    v: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Apply standard signed soft-thresholding."""
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def _soft_threshold_nonnegative(
    v: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Apply soft-thresholding subject to x >= 0."""
    return np.maximum(v - threshold, 0.0)


def solve_fista_elasticnet_backtracking(
    Aop: LinearOperator,
    y: np.ndarray,
    lam1: float,
    lam2: float,
    *,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    L0: float = 1.0,
    eta: float = 2.0,
    verbose: bool = False,
    nonnegative: bool = True,
) -> np.ndarray:
    """
    Solve the Elastic Net problem using FISTA with backtracking:

        minimize_x  0.5 * ||A x - y||_2^2
                  + lam1 * ||x||_1
                  + 0.5 * lam2 * ||x||_2^2

    If ``nonnegative=True``, additionally impose:

        x >= 0

    Parameters
    ----------
    Aop
        Forward model represented as a SciPy LinearOperator.
    y
        Measurement vector.
    lam1
        L1 regularization parameter. Must be nonnegative.
    lam2
        L2 regularization parameter. Must be nonnegative.
    x0
        Optional initial estimate.
    max_iter
        Maximum number of FISTA iterations.
    tol
        Relative-change stopping tolerance.
    L0
        Initial estimate of the gradient Lipschitz constant.
    eta
        Backtracking multiplier; must be greater than 1.
    verbose
        Print convergence information when True.
    nonnegative
        If True, constrain the solution to be nonnegative.

    Returns
    -------
    np.ndarray
        Reconstructed solution.
    """
    if lam1 < 0:
        raise ValueError("lam1 must be nonnegative.")
    if lam2 < 0:
        raise ValueError("lam2 must be nonnegative.")
    if L0 <= 0:
        raise ValueError("L0 must be positive.")
    if eta <= 1:
        raise ValueError("eta must be greater than 1.")

    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if y.size != Aop.shape[0]:
        raise ValueError(
            f"y has length {y.size}, but Aop has {Aop.shape[0]} rows."
        )

    n = Aop.shape[1]

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()

        if x.size != n:
            raise ValueError(
                f"x0 has length {x.size}, but Aop has {n} columns."
            )

        if nonnegative:
            x = np.maximum(x, 0.0)

    z = x.copy()
    t = 1.0
    L = float(L0)

    def smooth_objective(u: np.ndarray) -> float:
        residual = Aop.matvec(u) - y

        return (
            0.5 * float(np.vdot(residual, residual).real)
            + 0.5 * lam2 * float(np.vdot(u, u).real)
        )

    for k in range(max_iter):
        Az = Aop.matvec(z)
        residual_z = Az - y

        grad = Aop.rmatvec(residual_z) + lam2 * z
        grad = np.asarray(grad, dtype=np.float64)

        fz = (
            0.5 * float(np.vdot(residual_z, residual_z).real)
            + 0.5 * lam2 * float(np.vdot(z, z).real)
        )

        while True:
            v = z - grad / L
            threshold = lam1 / L

            if nonnegative:
                x_new = _soft_threshold_nonnegative(v, threshold)
            else:
                x_new = _soft_threshold(v, threshold)

            dx = x_new - z

            quadratic_bound = (
                fz
                + float(np.vdot(grad, dx).real)
                + 0.5 * L * float(np.vdot(dx, dx).real)
            )

            if smooth_objective(x_new) <= quadratic_bound + 1e-12:
                break

            L *= eta

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))

        z = (
            x_new
            + ((t - 1.0) / t_new) * (x_new - x)
        )

        relative_change = (
            np.linalg.norm(x_new - x)
            / max(np.linalg.norm(x), 1e-12)
        )

        x = x_new
        t = t_new

        if relative_change < tol:
            if verbose:
                print(
                    f"Elastic Net converged at iteration {k + 1}; "
                    f"L={L:.3e}, relative change={relative_change:.3e}"
                )
            break

    return x