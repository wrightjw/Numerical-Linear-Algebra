import numpy as np

def get_Lk(m, lvec):
    """
    Compute the lower triangular row operation mxm matrix L_k
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).
    
    :param m: integer giving the dimensions of L.
    :param lvec: a k-1 dimensional numpy array.
    
    :return Lk: an mxm dimensional numpy array.
    """

    # Inserting column entries below diagonal of ones
    Lk = np.eye(m, dtype=complex)
    k = m-1-lvec.size
    Lk[k+1:, k] = lvec
    
    return Lk

def LU_inplace(A):
    """
    Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.
    
    :param A: an mxm-dimensional numpy array
    """
    
    #Extracting data of matrix shape
    m, m = A.shape

    # Looping L and U matrices into A, exploiting outer product column operations to replace loop
    for k in range(m-1):
        A[k+1:, k] = A[k+1:,k]/A[k,k] # Computing L
        A[k+1:, k+1:] = A[k+1:,k+1:]-np.outer(A[k+1:,k],A[k,k+1:]) # Computing U


def solve_L(L, b, diag_one = False):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,\ldots,k
    
    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing
    b_i
    :param diag_one: boolean, L has ones on diagonal
    
    :return x: an mxk-dimensional numpy array, with ith column containing
    the solution x_i
    """
    # Extracting shape data and initialising x
    m , n = L.shape
    k = b.shape[1]
    x = np.zeros((m,k), dtype=complex)

    #Giving optional argument for diagonal of ones already in L as suggested in problem class
    # Solving system via forward substitution
    if diag_one == True:
        x[0,:] = b[0,:]
        for i in range(1,m):
            x[i,:] = b[i,:]-L[i,:i].dot(x[:i,:])
    else:
        x[0,:] = b[0,:]/L[0,0]
        for i in range(1,m):
            x[i,:] = (b[i,:] - L[i,:i].dot(x[:i,:]))/L[i,i]

    return x



def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,\ldots,k
    
    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing
    b_i
    
    :return x: an mxk-dimensional numpy array, with ith column containing
    the solution x_i
    """
    
    # Extracting shape data and initialising x
    m, n = U.shape
    m, k = b.shape
    x = np.zeros((m,k), dtype = complex)

    # Solving system via backward substitution
    x[m-1:,:] = b[m-1:,:]/U[m-1,m-1]
    for i in range(m-2, -1, -1):
        x[i,:] = (b[i,:]-U[i,i+1:].dot(x[i+1:,:]))/U[i,i]

    return x


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.
    
    :param A: an mxm-dimensional numpy array.
    
    :return Ainv: an mxm-dimensional numpy array.
    """
                     
    # Extracting shape data from A
    m, n = A.shape

    # Obtaining A=LU
    LU_inplace(A)

    # invA = invU invL
    # Finding half of inverse corresponding to L via identity
    Linv = solve_L(A, np.eye(m, dtype = complex), diag_one=True)
    
    # Finding A inverse via solving U
    Ainv = solve_U(A,Linv)

    return Ainv
