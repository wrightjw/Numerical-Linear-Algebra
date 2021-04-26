import numpy as np
from utilities import *

def pure_QR(A, maxit, tol):
    """
    For matrix A, apply the QR algorithm and return the result.
    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    
    :return Ak: the result
    """
    epsilon = tol
    Ak = 1.0*A
    
    # Apply pure QR algorithm
    # While r is above tolerance
    i = 0
    while epsilon>=tol:
        
        # Check max iterations
        i = i+1
        if i > maxit:
            break
        
        # Apply QR algorithm
        Qk, Rk = householder_qr(Ak)
        Ak = Rk.dot(Qk)
        epsilon = np.linalg.norm(Ak-A)
    
    return Ak

