import pytest
from utilities import *
from numpy import random
import numpy as np

@pytest.mark.parametrize('m', [20, 30, 40, 50, 11, 5, 99, 18])
def test_GMRES(m):
    A = random.randn(m, m)
    b = random.randn(m)

    x = np.random.random(m)
    x, _ = GMRES(A, b, maxit=1000, tol=1.0e-3)
    assert(np.linalg.norm(np.dot(A, x) - b) < 1.0e-3)

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
