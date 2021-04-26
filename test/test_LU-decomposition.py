import pytest
from utilities import *
from numpy import random
import numpy as np
from utilities import LU_inplace, get_Lk, solve_L, solve_U


@pytest.mark.parametrize('m, k', [(20, 4), (204, 100), (18, 7)])
def test_get_Lk(m, k):
    random.seed(9752*m)
    lk = random.randn(m-k-1)
    Lk = get_Lk(m, lk)
    assert(np.count_nonzero(Lk) == 2*m - k - 1)

    b = random.randn(m)
    x = np.dot(Lk, b)
    assert(np.linalg.norm(x[0:k+1]-b[0:k+1]) < 1.0e-6)
    for i in range(k+1, m):
        assert(np.linalg.norm(x[i] - b[i] - lk[i-k-1]*b[k]) < 1.0e-6)


@pytest.mark.parametrize('m', [20, 204, 18])
def test_LU_inplace(m):
    random.seed(8564*m)
    A = random.randn(m, m)
    A0 = 1.0*A
    LU_inplace(A)
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    A1 = np.dot(L, U)
    err = A1 - A0
    assert(np.linalg.norm(err) < 1.0e-6)


@pytest.mark.parametrize('m, k', [(20, 4), (204, 100), (18, 7)])
def test_solve_L(m, k):
    random.seed(1002*m + 2987*k)
    b = random.randn(m, k)
    Q, R = np.linalg.qr(random.randn(m, m))
    L = R.T
    x = solve_L(L, b)
    err1 = b - np.dot(L, x)
    assert(np.linalg.norm(err1) < 1.0e-6)
    A = random.randn(m, m)
    x = solve_L(A, b)
    err2 = b - np.dot(A, x)
    assert(np.linalg.norm(err2) > 1.0e-6)


@pytest.mark.parametrize('m, k', [(20, 4), (204, 100), (18, 7)])
def test_solve_U(m, k):
    random.seed(1002*m + 2987*k)
    b = random.randn(m, k)
    _, U = np.linalg.qr(random.randn(m, m))
    x = solve_U(U, b)
    err1 = b - np.dot(U, x)
    assert(np.linalg.norm(err1) < 1.0e-6)
    A = random.randn(m, m)
    err2 = b - np.dot(A, x)
    assert(np.linalg.norm(err2) > 1.0e-6)


@pytest.mark.parametrize('m', [20, 204, 18])
def test_inverse_LU(m):
    random.seed(5422*m)
    A = random.randn(m, m)
    A0 = 1.0*A

    Ainv = inverse_LU(A0)
    err = np.dot(Ainv, A) - np.eye(m)
    assert(np.linalg.norm(err) < 1.0e-6)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
