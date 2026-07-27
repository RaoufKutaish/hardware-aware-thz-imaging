from typing import Optional

import numpy as np
from scipy.optimize import nnls


def solve_cosamp_explicit(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    max_iter: int = 50,
    tol: Optional[float] = None,
    candidate_mult: int = 2,
    ridge_lam2: float = 0.0,
    nonnegative: bool = True,
) -> np.ndarray:
    """
    Solve a k-sparse approximation problem using CoSaMP.

    Approximately solves:

        minimize_x  ||A x - y||_2^2
        subject to  ||x||_0 <= k

    If ``nonnegative=True``, additionally impose:

        x >= 0

    Parameters
    ----------
    A
        Explicit sensing matrix. Its columns should normally be normalized.
    y
        Measurement vector.
    k
        Maximum solution sparsity.
    max_iter
        Maximum number of CoSaMP iterations.
    tol
        Optional stopping threshold on the residual norm.
    candidate_mult
        Number of candidate atoms relative to k.
    ridge_lam2
        Optional L2 regularization for active-set fitting.
    nonnegative
        If True, use positive correlations and nonnegative least squares.

    Returns
    -------
    np.ndarray
        Reconstructed sparse solution.
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
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if candidate_mult < 1:
        raise ValueError("candidate_mult must be at least 1.")
    if tol is not None and tol < 0:
        raise ValueError("tol must be nonnegative.")
    if ridge_lam2 < 0:
        raise ValueError("ridge_lam2 must be nonnegative.")

    if k <= 0:
        return np.zeros(n, dtype=np.float64)

    k_eff = min(int(k), n)
    candidate_count = min(candidate_mult * k_eff, n)

    x = np.zeros(n, dtype=np.float64)
    residual = y.copy()
    support = np.empty(0, dtype=np.int64)

    def topk_indices(
        values: np.ndarray,
        count: int,
        *,
        positive_only: bool,
    ) -> np.ndarray:
        """Return indices of the largest eligible entries."""
        count = min(max(int(count), 0), values.size)

        if count == 0:
            return np.empty(0, dtype=np.int64)

        if positive_only:
            eligible = np.flatnonzero(values > 0.0)

            if eligible.size == 0:
                return np.empty(0, dtype=np.int64)

            count = min(count, eligible.size)
            eligible_values = values[eligible]

            local_indices = np.argpartition(
                eligible_values,
                -count,
            )[-count:]

            local_indices = local_indices[
                np.argsort(eligible_values[local_indices])[::-1]
            ]

            return eligible[local_indices].astype(
                np.int64,
                copy=False,
            )

        count = min(count, values.size)
        indices = np.argpartition(
            np.abs(values),
            -count,
        )[-count:]

        indices = indices[
            np.argsort(np.abs(values[indices]))[::-1]
        ]

        return indices.astype(np.int64, copy=False)

    def fit_active_set(
        active_matrix: np.ndarray,
    ) -> np.ndarray:
        """Fit coefficients on the current active set."""
        active_size = active_matrix.shape[1]

        if active_size == 0:
            return np.empty(0, dtype=np.float64)

        if nonnegative:
            if ridge_lam2 > 0.0:
                # Ridge-regularized NNLS:
                #
                # min_{z >= 0} ||As z - y||^2 + lam2 ||z||^2
                augmented_matrix = np.vstack(
                    [
                        active_matrix,
                        np.sqrt(ridge_lam2)
                        * np.eye(active_size),
                    ]
                )
                augmented_y = np.concatenate(
                    [
                        y,
                        np.zeros(active_size, dtype=np.float64),
                    ]
                )

                coefficients, _ = nnls(
                    augmented_matrix,
                    augmented_y,
                )
            else:
                coefficients, _ = nnls(active_matrix, y)

            return coefficients

        if ridge_lam2 > 0.0:
            # Ridge-stabilized unconstrained least squares.
            gram_matrix = active_matrix.T @ active_matrix
            gram_matrix.flat[
                :: gram_matrix.shape[0] + 1
            ] += ridge_lam2

            return np.linalg.solve(
                gram_matrix,
                active_matrix.T @ y,
            )

        coefficients, *_ = np.linalg.lstsq(
            active_matrix,
            y,
            rcond=None,
        )
        return coefficients

    for _ in range(max_iter):
        # 1. Form the correlation proxy.
        proxy = A.T @ residual

        # 2. Identify candidate atoms.
        candidates = topk_indices(
            proxy,
            candidate_count,
            positive_only=nonnegative,
        )

        if candidates.size == 0:
            break

        # 3. Merge old support with new candidates.
        merged_support = np.unique(
            np.concatenate([support, candidates])
        ).astype(np.int64)

        # 4. Fit on the merged support.
        merged_matrix = A[:, merged_support]
        merged_coefficients = fit_active_set(merged_matrix)

        merged_solution = np.zeros(n, dtype=np.float64)
        merged_solution[merged_support] = merged_coefficients

        # 5. Prune the solution to at most k entries.
        support_new = topk_indices(
            merged_solution,
            k_eff,
            positive_only=nonnegative,
        )

        x_new = np.zeros(n, dtype=np.float64)

        if support_new.size > 0:
            if nonnegative:
                # Refit after pruning to preserve the NNLS optimum
                # on the final support.
                pruned_coefficients = fit_active_set(
                    A[:, support_new]
                )
                x_new[support_new] = pruned_coefficients
            else:
                x_new[support_new] = merged_solution[support_new]

        # Remove any zero-valued atoms returned by NNLS.
        support_new = np.flatnonzero(x_new != 0.0).astype(
            np.int64
        )

        # 6. Update the residual.
        residual = y - A @ x_new

        if tol is not None:
            if np.linalg.norm(residual) <= tol:
                x = x_new
                break

        support_unchanged = np.array_equal(
            np.sort(support),
            np.sort(support_new),
        )

        x = x_new

        if support_unchanged:
            break

        support = support_new

    return x