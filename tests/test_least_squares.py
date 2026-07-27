import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from thz_reconstruction.solvers import solve_lsqr


def test_lsqr_recovers_exact_solution():
    A = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    x_true = np.array([2.0, 3.0])
    y = A @ x_true

    x_hat = solve_lsqr(aslinearoperator(A), y)

    np.testing.assert_allclose(x_hat, x_true, rtol=1e-7, atol=1e-7)


def test_lsqr_matches_numpy_least_squares():
    rng = np.random.default_rng(0)

    A = rng.normal(size=(20, 5))
    y = rng.normal(size=20)

    x_hat = solve_lsqr(aslinearoperator(A), y)
    x_expected = np.linalg.lstsq(A, y, rcond=None)[0]

    np.testing.assert_allclose(x_hat, x_expected, rtol=1e-6, atol=1e-6)


def test_lsqr_accepts_column_measurement_vector():
    A = np.eye(3)
    y = np.array([[1.0], [2.0], [3.0]])

    x_hat = solve_lsqr(aslinearoperator(A), y)

    np.testing.assert_allclose(x_hat, y.reshape(-1))


def test_lsqr_rejects_incompatible_dimensions():
    A = aslinearoperator(np.eye(3))
    y = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Incompatible dimensions"):
        solve_lsqr(A, y)


def test_lsqr_with_matrix_free_operator():
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    Aop = LinearOperator(
        shape=A.shape,
        matvec=lambda x: A @ x,
        rmatvec=lambda y: A.T @ y,
        dtype=A.dtype,
    )

    x_true = np.array([1.0, 2.0])
    y = A @ x_true

    x_hat = solve_lsqr(Aop, y)

    np.testing.assert_allclose(x_hat, x_true, rtol=1e-7, atol=1e-7)