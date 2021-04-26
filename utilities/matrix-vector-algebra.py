import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)


def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication. Returns an m-dimensional numpy array which is the product of A with x.
    
    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array
    
    :return b: m-dimensional numpy array
    """

    # Extracting dimensions and initialising vector b
    m , n = A.shape
    b = np.zeros(m)

    # Computing Ax=b using definition of matrix-vector multiplication
    for i in range(m):
        for j in range(n):
            b[i] += A[i,j] * x[j]

    return b


def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in
    x as coefficients.
    
    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array
    
    :return b: an m-dimensional numpy array which is the product of A with x
    """

    # Extracting dimensions and initialising vector b
    m , n = A.shape
    b = np.zeros(m)

    # Computing Ax=b using column-vector interpretation of matrix multiplication
    for j in range(n):
        b[:] += x[j] *  A[:, j]

    return b

def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0)  # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Timing matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1v1^* + u2v2^*.
    
    :param u1: m-dimensional numpy array
    :param u2: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    
    :return A: mxn dimensional numpy array
    """

    # Forming matrices B and C such that A=BC
    B = np.array([u1, u2]).T #Converting from rows to matrix of column vectors
    C = np.array([v1, v2]).conjugate() #Computing equivalent to conjugate transpose as v1 are stored as row vectors

    #Computing A=BC
    A = B.dot(C)

    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with
    
    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    
    :return Ainv: mxm-dimensional numpy array
    """

    #Extracting dimensions of u and v and taking conjugate of v
    m = u.size
    v_star = v.conjugate()

    # Computing derived value of alpha for A^-1 and A^-1 itself
    alpha = -1/(1+v.conjugate().dot(u))
    Ainv = np.eye(m) + alpha*np.outer(u,v_star)
    
    return Ainv


def inverse_timing():
    """
    Compute the times to compute a matrix of size 400 using rank1pert_inv() and np.linalg.inv(A)
    """

    #Generating vectors u and v that construct an A=I+uv^* 400x400 matrix to invert
    u = np.random.rand(400)
    v = np.random.rand(400)

    print("Timing for rank1pert_inv")
    print(timeit.Timer('rank1pert_inv',
                       'from __main__ import rank1pert_inv').timeit(number=1))

    # Computing matrix A
    A = np.eye(u.size) + np.outer(u, v.conjugate())
    print("Timing for np.linalg.inv")
    print(timeit.Timer('np.linalg.inv', 'import numpy as np').timeit(number=2))



def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with
    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j]
    for i<=j and Ahat[i,j] = C[i,j] for i>j.
    :param xr: an m-dimensional numpy array
    :param xi: an m-dimensional numpy array
    
    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """
    
    #Extracting dimensions and initialising matrices for right-hand side of the equation A(xr + i*xi) = Bxr + i Bxi + iCxr - Cxi
    m = Ahat.shape[0]
    Bxr = np.zeros(m)
    Bxi = np.zeros(m)
    Cxr = np.zeros(m)
    Cxi = np.zeros(m)

    #Computing vector element values of terms on right-hand side of the equation by exploiting the symmetry of B and anti-symmetry of C
    for k in range(m):
        # Computing Bxr
        u_Bxr = Ahat[k, k:m].dot(xr[k:m]) # Terms in dot product sum from upper triangular terms of B
        l_Bxr = Ahat[0:k,k].dot(xr[0:k]) # Terms in dot product sum from below diagonal terms of B
        Bxr[k] = u_Bxr + l_Bxr

        # Computing Bxi
        u_Bxi = Ahat[k, k:m].dot(xi[k:m]) # Terms in dot product sum from upper triangular terms of B
        l_Bxi = Ahat[0:k,k].dot(xi[0:k])  # Terms in dot product sum from below diagonal terms of B
        Bxi[k] = u_Bxi + l_Bxi

        # Computing Cxr - note: diagonal is zero as -C^T = C and real so ignored in computation
        u_Cxr = -1*Ahat[k+1:m,k].dot(xr[k+1:m])  # Terms in dot product sum from upper triangular terms of C
        l_Cxr = Ahat[k,0:k].dot(xr[0:k])  # Terms in dot product sum from lower triangular terms of C
        Cxr[k] = u_Cxr + l_Cxr

        # Computing Cxi - note: diagonal is zero as -C^T = C and real so ignored in computation
        u_Cxi = -1*Ahat[k+1:m,k].dot(xi[k+1:m])  # Terms in dot product sum from upper triangular terms of C
        l_Cxi = Ahat[k,0:k].dot(xi[0:k]) # Terms in dot product sum from lower triangular terms of C
        Cxi[k] = u_Cxi + l_Cxi


    # Computing left-hand side vector z
    zr = Bxr - Cxi
    zi = Bxi + Cxr

    return zr, zi
