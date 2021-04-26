import pytest
from utilties import *
from numpy import random
import numpy as np

@pytest.mark.parametrize('m', [20, 30, 18])
def test_pure_QR(m):
    random.seed(1302*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + A.conj().T)
    A0 = 1.0*A
    A2 = cla_utils.pure_QR(A0, maxit=10000, tol=1.0e-5)
    #check it is still Hermitian
    assert(np.linalg.norm(A2 - np.conj(A2).T) < 1.0e-4)
    #check for upper triangular
    assert(np.linalg.norm(A2[np.tril_indices(m, -1)])/m**2 < 1.0e-5)
    #check for conservation of trace
    assert(np.abs(np.trace(A0) - np.trace(A2)) < 1.0e-6)

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
