import numpy as np
from numpy import linalg
from random import random
from utilities import *

def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.
    
    :param m: the matrix dimension parameter.
    
    :return Q: the mxm numpy array containing the orthogonal matrix.
    """

    A = np.random.randn(m, m)

    Q, R = linalg.qr(A)
    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.
    
    :param m: the matrix dimension parameter.
    
    :return R: the mxm numpy array containing the upper triangular matrix.
    """
    
    A = np.random.randn(m, m)
    return np.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.
    
    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        Q1 = randomQ(m)
        R1 = randomR(m)
        A = Q1.dot(R1)
        
        Q2,R2 = np.linalg.qr(A)

        Q_norm = np.linalg.norm(Q2-Q1)
        R_norm = np.linalg.norm(R2-R1)
        A_norm = np.linalg.norm(A-Q2.dot(R2))

        print("-------------------------------------")
        print("The Q norm is", Q_norm)
        print("The R norm is", R_norm)
        print("The A norm is", A_norm)


def solve_R(R, b):
    """
    Solve the system Rx=b where R is an mxm upper triangular matrix
    and b is an m dimensional vector.
    
    :param R: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array
    :param x: an m-dimensional numpy array
    """
    # Initialising x as the normalisation of last term in b
    m = R.shape[0]
    x = np.zeros(m, dtype = complex)
    x[-1] = 1.0*b[-1]/R[-1,-1]

    #Back substitution
    for i in range(m-2, -1, -1):
        x[i] = (b[i] - R[i, i+1:].dot(x[i+1:]))/R[i, i] #Computing previous sum entries via matrix multiplication

    return x


def back_stab_solve_R(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.
    
    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        A = np.random.randn(m, m)
        R = np.triu(A)
        x_true = np.random.randn(m)
        b = R.dot(x_true)

        x = solve_R(R, b)
        x_norm = np.linalg.norm(x-x_true)

        print("-------------------------------------")
        print("The norm is", x_norm)


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.
    
    :param m: the matrix dimension parameter.
    """

    for k in range(20):
    
        A = np.random.randn(m, m)
        x_true = np.random.randn(m,5)
        b = A.dot(x_true)

        x = householder_solve(A,b)
        x_norm = np.linalg.norm(x-x_true)

        print("-------------------------------------")
        print("The norm is", x_norm)
