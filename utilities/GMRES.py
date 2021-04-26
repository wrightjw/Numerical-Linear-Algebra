import numpy as np
import numpy.random as random
from utilities import *
from scipy.sparse import csgraph
import scipy as sp

def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False, return_residuals=False, use_apply_pc = False, apply_pc = apply_pc):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm. use_apply_pc to solve Mb' = b.
    
    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical
    :param use_apply_pc: logical
    :param apply_pc: function
    
    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual at iteration k
    """
    
    # Initialsing and shape data
    m, n = A.shape
    e1 = np.zeros(maxit+1, dtype=complex)
    e1[0] = 1

    # If Mb'=b used
    if use_apply_pc == True:
        b0 = 1.0*b
        b = apply_pc(b)

    if x0 is None:
        x0 = b
    
    # Initialising Arnoldi
    H = np.zeros((maxit, maxit), dtype=complex)
    Q = np.zeros((m, maxit+1), dtype=complex)
    be1 = np.linalg.norm(b)*e1
    Q[:, 0] = x0/np.linalg.norm(x0)
    x = 1.0*x0

    # Compute residuals ||Rn||/||b||
    if use_apply_pc == True:
        res = A.dot(x) - b0
        Rn = np.linalg.norm(res)/np.linalg.norm(b0)
    else:
        res = A.dot(x) - b
        Rn = np.linalg.norm(res)/np.linalg.norm(b)

    # Options to store residuals and residual norms
    if return_residual_norms:
        rnorms = []
        # rnorms.append(Rn)
    if return_residuals:
        r = []
        # r.append(res)

    # For index and iterations
    i = 0
    nits = 0

    # Compute GMRES algorithm until tolerance
    while Rn > tol :
        nits = nits + 1
        if nits>maxit:
            nits = -1
            break
        
        v = A.dot(Q[:, i])
        # Apply n step of Arnoldi with Mv=Aq
        if use_apply_pc == True:
            v = apply_pc(v)

        # Inner loop exploiting vector operations
        H[:i+1, i] = Q[:, :i+1].conj().T.dot(v)
        v = v - Q[:, :i+1].dot(H[:i+1, i])

        # Normalisations of Q and H
        H[i+1, i] = np.linalg.norm(v)
        Q[:, i+1] = v/H[i+1, i]

        # Compute minimiser y of ||hy-||b||e1||
        y = householder_ls(H[:i+1, :i+1], be1[:i+1])
        x = Q[:, :(i+1)].dot(y)

        # Next index
        i = i + 1

        # Compute residuals ||Rn||/||b||
        if use_apply_pc == True:
            res = A.dot(x) - b0
            Rn = np.linalg.norm(res)/np.linalg.norm(b0)
        else:
            res = A.dot(x) - b
            Rn = np.linalg.norm(res)/np.linalg.norm(b)

        # Options to store residuals and residual norms
        if return_residual_norms:
            rnorms.append(Rn)
        if return_residuals:
            r.append(res)

    # Options to return
    if return_residual_norms and return_residuals:
        rnorms = np.asarray(rnorms)
        r = np.asarray(r)
        return x, nits, rnorms, r
    elif return_residual_norms:
        rnorms = np.asarray(rnorms)
        return x, nits, rnorms
    elif return_residuals:
        r = np.asarray(r)
        x, nits, r
    else:
        return x, nits


def get_AA100():
    """
    Get the AA100 matrix.
    
    :return A: a 100x100 numpy array used in exercises 10.
    """
    
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.
    
    :return B: a 100x100 numpy array used in exercises 10.
    """
    
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.
    
    :return C: a 100x100 numpy array used in exercises 10.
    """
    
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
