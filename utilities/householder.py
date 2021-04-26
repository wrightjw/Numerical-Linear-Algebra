import numpy as np
from scipy import linalg
import cmath

def householder(A, kmax=None):
    """
    Given an mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.
    
    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce
    to upper triangular. If not present, will default to n.
    
    :return R: an mxn-dimensional numpy array containing the upper
    triangular matrix
    """
    
    # Extracting dimensions of A and copying it
    m, n = A.shape
    R = 1.0*A
    R = R.astype(complex)

    if kmax is None:
        kmax = n

    # Implementing Householder algorithm
    for k in range(kmax):
        x = R[k:, k]

        # Standard unit vector in each step
        e_1 = np.zeros(x.shape)
        e_1[0] = 1

        # Cases to work around sign(0)=0 in python
        if x[0] == 0:
            vk = np.linalg.norm(x)*e_1 + x
            vk = vk/np.linalg.norm(vk)
        else:
            a, b = x[0].real, x[0].imag
            theta = np.arctan2(b,a)
            vk = np.exp(1j*theta)*np.linalg.norm(x)*e_1 + x
            vk = vk/np.linalg.norm(vk)
        
        #Forming R
        R[k:,k:] = R[k:,k:] - 2*np.outer(vk,(vk.conjugate().T.dot(R[k:,k:])))

    return R


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.
    
    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the
    right-hand side vectors b_1,b_2,...,b_k.
    
    :return x: an mxk-dimensional numpy array whose columns are the
    right-hand side vectors x_1,x_2,...,x_k.
    """

    # Extracting shape of A and b
    m, n = A.shape

    #Producing augmented [A|b] and solving for upper triangular form [R|b']
    A_hat = np.column_stack((A,b))
    R_aug = householder(A_hat, kmax = m)
    R = R_aug[:, :m]
    b_dash = R_aug[:, m:]
    
    # Solving system [R|b] for x
    x = linalg.solve_triangular(R,b_dash)

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the QR factorisation of A.
    
    :param A: an mxn-dimensional numpy array
    
    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    #Extracting shape of A and initialising I
    m, n = A.shape
    I = np.eye(m, dtype=complex)
    A_hat = np.concatenate((A, I), axis=1)

    #The solution for Q is such that [A|I]=[QR|I], applying householder returns [R|Q*] as we are right multiplying by Qi's repeatedly.
    R_aug = householder(A_hat, kmax = n)
    R = R_aug[:, :n]
    Q = R_aug[:, n:].conjugate().T # Inverting Q* to get Q
    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.
    
    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array
    
    :return x: an n-dimensional numpy array
    """

    m, n = A.shape

    #Constructing augmented matrix [A|b]
    A_hat = np.column_stack((A, b))
    # Householder [A|b] -> [R|Q*b] where this is equivalent to Ax = Pb -  the projection onto the range of A
    R_aug = householder(A_hat, kmax=n)

    # Solving this upper triangular system where the system is consistent returns x_min
    R = R_aug[:n, :n]
    Q_starb = R_aug[:n, -1:]

    # Returning an n-dimensional numpy array solving Rx=Q*b
    x = linalg.solve_triangular(R, Q_starb).T[0]

    return x
