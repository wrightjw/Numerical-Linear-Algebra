import pytest
from utilities import *
from numpy import random
import numpy as np

@pytest.mark.parametrize('m, n', [(20, 17), (40, 3), (20, 12)])
def test_orthog_cpts(m, n):
    random.seed(1878*m + 1950*n)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    v = random.randn(m) + 1j*random.randn(m)
    Q, R = np.linalg.qr(A)
    Q = Q[:, 0:n]

    r, u = orthog_cpts(v, Q)
    err = v - r - Q.dot(u)

    assert(np.linalg.norm(err) < 1.0e-6)


@pytest.mark.parametrize('m', [17, 35, 100])
def test_solveQ(m):
    random.seed(1431*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    v = random.randn(m) + 1j*random.randn(m)
    Q, R = np.linalg.qr(A)

    x = solveQ(Q, v)
    x0 = np.linalg.solve(Q, v)
    err = x - x0

    assert(np.linalg.norm(err) < 1.0e-6)


@pytest.mark.parametrize('m, n', [(20, 17), (40, 3), (20, 12)])
def test_orthog_proj(m, n):
    random.seed(1878*m + 1950*n)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    Q, R = np.linalg.qr(A)
    Q = Q[:, 0:n]

    P = orthog_proj(Q)

    for i in range(n):
        q1 = Q[:, i]
        q2 = np.dot(P, q1)
        if i < m+1:
            assert(np.linalg.norm(q1 - q2) < 1.0e-6)
        else:
            assert(np.linalg.norm(q2) < 1.0e-6)


@pytest.mark.parametrize('m, n', [(211, 17), (40, 3)])
def test_orthog_space(m, n):
    random.seed(1321*m + 1765*n)
    U = random.randn(m, n) + 1j*random.randn(m, n)
    Qhat = orthog_space(U)
    #check that the dimensions are correct
    assert(Qhat.shape == (m, m-n))
    #Check the orthogonality
    assert(np.linalg.norm(np.dot(Qhat.conj().T, U)) < 1.0e-6)
    #Check full rank
    assert(np.linalg.matrix_rank(Qhat) == m-n)


@pytest.mark.parametrize('m, n', [(20, 17), (40, 3), (20, 12)])
def test_GS_classical(m, n):
    random.seed(1312*m + 2020*n)

    # artificially construct a basis with good linear independence
    A = random.randn(m, m) + 1j*random.randn(m, m)
    U, _ = np.linalg.qr(A)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    V, _ = np.linalg.qr(A)
    D = np.diag(1.0 + 0.1*random.rand(m))
    A = np.dot(U, np.dot(D, V))
    A = A[:, 0:n]
    A0 = 1.0*A

    Q, R = GS_classical(A0)

    err = A0 - np.dot(Q, R)

    assert(np.linalg.norm(err) < 1.0e-6)


@pytest.mark.parametrize('m, n', [(20, 17), (40, 3), (20, 12)])
def test_GS_modified(m, n):
    random.seed(1312*m + 2020*n)

    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = A[:, 1:n]
    A0 = 1.0*A

    Q, R = GS_modified(A0)

    err = A0 - np.dot(Q, R)

    assert(np.linalg.norm(err) < 1.0e-6)


@pytest.mark.parametrize('m, n', [(4, 3), (5, 3), (6, 3)])
def test_GS_modified_R(m, n):
    random.seed(1312*m + 2020*n)

    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = A[:, 0:n]

    A0 = 1.0*A
    Q, R = GS_modified_R(A0.copy())

    err = A0 - np.dot(Q, R)

    assert(np.linalg.norm(err) < 1.0e-6)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
