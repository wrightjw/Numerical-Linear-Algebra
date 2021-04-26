import numpy as np
from numpy import *
from utilties import *

def find_approx_zero(M, epsilon):
    """
    Tests whether an mxn matrix rows are approximately zero vectors - where approximately means less than some epsilon, returning how many zero rows there are and where they are.
    
    :param M: An mxn numpy array
    :param epsilon: A float
    
    :return count: An integer
    :return zero_rows: A list of integers
    """

    # Initialising zero_rows and extracting matrix dimensions.
    m, n = M.shape
    zero_rows = []
    count = 0

    # Assigning elements 0 or 1 value depending on whether they are less than some epsilon.
    M_approx = np.where(abs(M) < epsilon, +0.0, 1.0)

    for k in range(m):
        if (abs(M_approx[k,:]) == np.zeros(n)).all():
            count += 1
            zero_rows.append(k)

    return count, zero_rows
