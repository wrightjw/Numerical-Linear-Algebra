import pytest
from utilities import *
from numpy import random
import numpy as np


@pytest.mark.parametrize('m, k', [(20, 4), (40, 20), (70, 13)])
def test_arnoldi(m, k):
    A = random.randn(m, m) + 1j*random.randn(m, m)
    b = random.randn(m) + 1j*random.randn(m)

    Q, H = arnoldi(A, b, k)
    assert(Q.shape == (m, k+1))
    assert(H.shape == (k+1, k))
    assert(np.linalg.norm((Q.conj().T)@Q - np.eye(k+1)) < 1.0e-6)
    assert(np.linalg.norm(A@Q[:, :-1] - Q@H) < 1.0e-6)

