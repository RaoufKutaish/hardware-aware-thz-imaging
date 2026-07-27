from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

def solve_lsqr(
    Aop: LinearOperator,
    y: np.ndarray,
    *,
    atol: float = 1e-8,
    btol: float = 1e-8,
    iter_lim: Optional[int] = None,
) -> np.ndarray:
    """
    Solve the linear least-squares problem min_x ||A x - y||_2
    using SciPy's iterative LSQR algorithm.

    Parameters
    ----------
    Aop
        Forward operator with shape (m, n). It must implement both
        ``matvec`` and ``rmatvec``.
    y
        Measurement vector with shape (m,).
    atol, btol
        LSQR stopping tolerances.
    iter_lim
        Maximum number of iterations. If None, SciPy selects the default.

    Returns
    -------
    np.ndarray
        Reconstructed vector with shape (n,).
    """
    y = np.asarray(y).reshape(-1)

    if y.size != Aop.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: A has {Aop.shape[0]} rows, "
            f"but y has length {y.size}."
        )

    result = lsqr(
        Aop,
        y,
        atol=atol,
        btol=btol,
        iter_lim=iter_lim,
    )
    return result[0]