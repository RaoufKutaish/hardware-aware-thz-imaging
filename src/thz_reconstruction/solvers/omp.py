from typing import List, Optional

import numpy as np
from scipy.optimize import nnls


def solve_omp_explicit(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    tol: Optional[float] = None,
    nonnegative: bool = True,
) -> np.ndarray:
    """
    Solve a sparse approximation problem using OMP.

    Standard OMP solves approximately:

        minimize_x  ||A x - y||_2^2
        subject to  ||x||_0 <= k

    If ``nonnegative=True``, additionally impose:

        x >= 0

    Notes
    -----
    OMP requires explicit access to the columns of A.
    The columns of A should normally be normalized before calling
    this function.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional array.")

    m, n = A.shape

    if y.size != m:
        raise ValueError(
            f"y has length {y.size}, but A has {m} rows."
        )
    if k < 0:
        raise ValueError("k must be nonnegative.")
    if k > n:
        raise ValueError(
            f"k cannot exceed the number of columns in A ({n})."
        )
    if tol is not None and tol < 0:
        raise ValueError("tol must be nonnegative.")

    x = np.zeros(n, dtype=np.float64)

    if k == 0:
        return x

    residual = y.copy()
    support: List[int] = []
    active_coefficients = np.empty(0, dtype=np.float64)

    for _ in range(k):
        correlations = A.T @ residual

        if nonnegative:
            # Only positive correlations can produce positive coefficients.
            j = int(np.argmax(correlations))

            if correlations[j] <= 0:
                break
        else:
            # Standard OMP permits positive and negative coefficients.
            j = int(np.argmax(np.abs(correlations)))

        if j in support:
            break

        support.append(j)
        active_matrix = A[:, support]

        if nonnegative:
            active_coefficients, _ = nnls(active_matrix, y)

            # Remove atoms whose NNLS coefficients became zero.
            keep = active_coefficients > 0.0
            support = [
                index
                for index, retained in zip(support, keep)
                if retained
            ]
            active_coefficients = active_coefficients[keep]

            if not support:
                break

            active_matrix = A[:, support]
        else:
            active_coefficients, *_ = np.linalg.lstsq(
                active_matrix,
                y,
                rcond=None,
            )

        residual = y - active_matrix @ active_coefficients

        if tol is not None and np.linalg.norm(residual) <= tol:
            break

    if support:
        x[support] = active_coefficients

    return x