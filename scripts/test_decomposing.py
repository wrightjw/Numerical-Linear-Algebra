import pytest
from utilities import *
from decomposing_tools import *
from numpy import random
import numpy as np


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4])
def test_approx_zeros(n):
    """
    Test to see if count and selected zero rows are correct.
    
    :param m: An integer corresponding to degree of polynomial
    :param n: An integer corresponding to number of points taken from that polynomial
    
    :return bool: Pass or fail
    """

    # Constructing test upper triangular matrix and epsilon
    epsilon = 10**(-n)
    A = np.zeros([4,5])
    A[0,:] = 0.1*np.ones(5)
    A[1, :] = -0.01*np.ones(5)
    A[2, :] = 0.001*np.ones(5)
    A[3, :] = -0.0001*np.ones(5)

    # Testing values of epsilon that include another row - each smaller epsilon should include the next row until all rows are deemed 'nonzero'
    test_zeros, test_zero_rows = find_approx_zero(A, epsilon)
    zeros = 4-n
    zero_rows = np.array(range(n,4))

    # Checking correct number of rows and positions taken, checking right number of positions.
    assert(zeros == test_zeros)
    assert((zero_rows == test_zero_rows).all())
    assert(test_zeros == len(test_zero_rows))
    
if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
