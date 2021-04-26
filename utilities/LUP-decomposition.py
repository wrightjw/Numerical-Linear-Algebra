import numpy as np
from utilities import *

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.
    
    :param p: an m-dimensional numpy array of integers.
    """
    
    # Simultaneously swapping rows
    p[i], p[j] = p[j], p[i]


# Including sign option as suggested in tutorial
def LUP_inplace(A, sign = False):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.
    
    :param A: an mxm-dimensional numpy array
    :param sgn: Boolean to change sign
    
    :return p: an m-dimensional integer array describing the permutation
    i.e. (Px)[i] = x[p[i]]
    """
    
    # Initialising data
    sgn = 1 # signature of permutation
    m = A.shape[0]
    p = np.arange(m)

    # Applying LUP factorisation algorithm with pivots
    for k in range(m-1):
        # Obtain largest row to swap into highest place
        i = np.argmax(np.abs(A[k:,k])) + k #Choosing i>=k to max |uik|
        A[i,:], A[k,:] = 1.0*A[k,:], 1.0*A[i,:] #Swap the rows
        perm(p, i, k) #Permutation p
        if i != k: # Change parity
            sgn = -1*sgn
        A[k+1:, k] = A[k+1:, k]/A[k,k]
        A[k+1:, k+1:] = A[k+1:, k+1:] - np.outer(A[k+1:, k], A[k, k+1:])
    if sign == True:
        return p, sgn
    else:
        return p

def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.
    
    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array
    
    :return x: an m-dimensional numpy array
    """
    
    #Extracting data
    m,n = A.shape

    # Find permutation vector
    p = LU_inplace(A)

    Pb = b[p] # Permutate b
    Pb = Pb.reshape((m,1)) # Rotate to column vector

    L = np.tril(A) # Extract lower triangular of A
    U = np.triu(A) # Extract upper triangular of A

    # Ax=b so L(Ux)=Ly=Pb
    y = solve_L(L,Pb,diag_one=True)
    x = solve_U(U,y)

    return x.reshape(m)


def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.
    
    :param A: an mxm-dimensional numpy array
    
    :return detA: floating point number, the determinant.
    """
                     
    p, sgn = LUP_inplace(A, sign = True)

    #det(A) = det(LUP^-1)=det(L)det(U)det(P) = sgn*det(U)
    # Extract diagonal entries
    diagU = np.diag(np.triu(A))

    prodU = np.prod(diagU)

    detA = sgn*prodU

    return detA
