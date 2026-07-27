from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator


def solve_ridge_fista(
    Aop: LinearOperator,
    y: np.ndarray,
    lam2: float,
    *,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 2000,
    tol: float = 1e-6,
    L: Optional[float] = None,
    nonnegative: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve the Ridge problem:

        minimize_x  0.5 * ||A x - y||_2^2
                  + 0.5 * lam2 * ||x||_2^2

    If ``nonnegative=True``, additionally impose:

        x >= 0

    using accelerated projected-gradient iterations.

    Parameters
    ----------
    Aop
        Forward model represented as a SciPy LinearOperator.
    y
        Measurement vector.
    lam2
        L2 regularization parameter. Must be nonnegative.
    x0
        Optional initial estimate.
    max_iter
        Maximum number of iterations.
    tol
        Relative-change stopping tolerance.
    L
        Lipschitz constant of the gradient. If None, it is
        estimated as ||A||_2^2 + lam2.
    nonnegative
        If True, constrain the solution to be nonnegative.
    verbose
        Print convergence information when True.

    Returns
    -------
    np.ndarray
        Reconstructed solution.
    """
    if lam2 < 0:
        raise ValueError("lam2 must be nonnegative.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if tol < 0:
        raise ValueError("tol must be nonnegative.")

    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if y.size != Aop.shape[0]:
        raise ValueError(
            f"y has length {y.size}, but Aop has "
            f"{Aop.shape[0]} rows."
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

    if L is None:
        L = estimate_lipschitz(Aop) + lam2
    else:
        L = float(L)

    if L <= 0:
        raise ValueError("L must be positive.")

    z = x.copy()
    t = 1.0

    for k in range(max_iter):
        residual = Aop.matvec(z) - y
        grad = Aop.rmatvec(residual) + lam2 * z
        grad = np.asarray(grad, dtype=np.float64)

        x_new = z - grad / L

        if nonnegative:
            x_new = np.maximum(x_new, 0.0)

        relative_change = (
            np.linalg.norm(x_new - x)
            / max(np.linalg.norm(x), 1e-12)
        )

        if relative_change < tol:
            if verbose:
                constraint = "nonnegative " if nonnegative else ""
                print(
                    f"{constraint}Ridge converged at iteration "
                    f"{k + 1}; relative change={relative_change:.3e}"
                )

            return x_new

        t_new = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * t * t)
        )

        z = (
            x_new
            + ((t - 1.0) / t_new) * (x_new - x)
        )

        x = x_new
        t = t_new

    return x