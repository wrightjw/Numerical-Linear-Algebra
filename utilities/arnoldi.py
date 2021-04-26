import numpy as np
import numpy.random as random
from utilities import *
from scipy.sparse import csgraph
import scipy as sp

def arnoldi(A, b, k, n_step = None):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.
    
    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations
    
    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper
    Hessenberg matrix
    """

    # Initialise matrices
    m, n = A.shape
    Q = np.zeros((m, k+1), dtype = complex)
    H = np.zeros((k+1, k), dtype = complex)
    Q[:, 0] = b/np.linalg.norm(b)

    # Run Arnoldi algorithm
    for n in range(k):
        # Compute v
        v = A.dot(Q[:,n])
        
        # Inner loop exploiting vector operations
        H[:n+1,n] = Q[:,:n+1].conjugate().T.dot(v)
        v = v - Q[:,:n+1].dot(H[:n+1,n])

        # Normalisations of Q and H
        H[n+1,n] = np.linalg.norm(v)
        Q[:,n+1] = v/H[n+1,n]
    
    return Q, H

def apply_pc(b):
    """
    Function to solve Mb' = b by exploiting the structure of upper triangular M, where M is fixed.
    
    :param b: An m dimensional numpy array
    """

    # Generate a graph
    m = b.shape[0]
    G = np.zeros((m, m))
    np.fill_diagonal(G, 0)

    # Produce Laplacian of graph
    L = sp.sparse.csgraph.laplacian(G)

    # Produce A and M
    A = np.eye(m) + L
    M = np.triu(A)

    # Solve upper triangular Mx = b
    b_twiddle = solve_R(M, b)
    
    return b_twiddle
