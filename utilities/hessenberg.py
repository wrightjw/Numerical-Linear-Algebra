import numpy as np
from utilities import *

def my_sign(k):
    """
    Adjust np.sign(0)=-1 to be sign(0)=1
    :param k: A float
    :return s: 1 or -1
    """

    if k == 0:
        s = 1
        return s
    else:
        s = np.sign(k)
        return s

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.
    :param A: an mxm numpy array
    :return A1: an mxm numpy array
    """
    
    #Extracting shape data and initialising eigenvector and vector from A
    x = A[:, 0]
    A1 = 1.0*A

    # Standard unit vector in each step
    e_1 = np.zeros(x.shape)
    e_1[0] = 1

    v = my_sign(x[0])*np.linalg.norm(x) * e_1 + x
    v = v/np.linalg.norm(v)

    # Apply householder
    A1 = A1 - 2*np.outer(v,(v.conjugate().T.dot(A))) # Q_1*A
    A1 = A1 - 2*np.dot(A1, np.outer(v,v).conjugate().T) #Q_1*A*Q_1^*

    return A1


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.
    :param A: an mxm numpy array
    """

    #Extracting data and initialising
    m, n = A.shape

    # Hessenberg algorithm
    for k in range(m-2):
        # Generate vector from matrix and eigenvectors
        x = A[k+1:, k]
        e_1 = np.zeros(m-(k+1))
        e_1[0] = 1

        vk = my_sign(x[0])*np.linalg.norm(x)*e_1 + x
        vk = vk/np.linalg.norm(vk)

        A[k+1:, k:] = A[k+1:, k:] - 2*np.outer(vk, vk.conjugate().T.dot(A[k+1:, k:]))
        A[k:, k+1:] = A[k:, k+1:] - 2*np.dot(A[k:, k+1:], np.outer(vk, vk).conjugate().T)


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.
    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """
    m, _ = A.shape
    Q = np.eye(m, dtype=A.dtype)

    # Hessenberg Algorithm
    for k in range(m-2):
        x = A[k+1:, k]
        e1 = np.zeros(m-(k+1))
        e1[0] = 1

        vk = my_sign(x[0])*np.linalg.norm(x)*e1 + x
        vk = vk/np.linalg.norm(vk)

        A[k+1:, k:] = A[k+1:, k:] - 2*np.outer(vk, vk.conjugate().T.dot(A[k+1:, k:]))
        A[:, k+1:] = A[:, k+1:] - 2*np.dot(A[:, k+1:], np.outer(vk, vk).conjugate().T)

        # Adapted to find Q
        Q[k+1:, :] = Q[k+1:, :] - 2*np.outer(vk, vk.conj().dot(Q[k+1:, :]))
        
    return Q.conj().T
    

def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.
    :param H: an mxm numpy array
    :return ee: an m dimensional numpy array containing the eigenvalues of H
    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """
    m, n = H.shape
    assert(m == n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    ee, V = np.linalg.eig(H)
    return ee, V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).
    :param A: an mxm numpy array
    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    # Compute to Hessenberg and return Q matrix
    Q = hessenbergQ(A)

    # Computing eigenvalues and eigenvectors of H
    V = hessenberg_ev(A)

    # Converting into eigenvalues of A
    V = Q.dot(V)

    return V
