import numpy as np

def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r.
    
    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the
    orthonormal vectors
    
    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """

    # Extracting dimensions for v and Q, and initialising r and u
    m, n = Q.shape
    r = v
    u = np.zeros(n, dtype=Q.dtype)

    # Computing coefficients of the orthonormal component of v and residual respectively
    for i in range(n):
        u[i] = np.dot(Q[:,i].conjugate(),v)
        r = r - u[i]*Q[:,i]
    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.
    
    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS
    
    :return x: m dimensional array containing the solution
    """

    # Using that Q*Q = I to solve Qx=b as x = Q*b
    Q_star = Q.conjugate().T
    x = Q_star.dot(b)

    return x

def time_solveQ():
    #Generating vectors u and v that construct an A=I+uv^* 400x400 matrix to invert

    # n=100
    Q = np.random.rand(100, 100)
    b = np.random.rand(100)

    print("Timing for solveQ for matrix of size 100;")
    print(timeit.Timer('rank1pert_inv',
                       'from __main__ import rank1pert_inv').timeit(number=1))

    print("Timing for np.linalg.solve")
    print(timeit.Timer('np.linalg.solve', 'import numpy as np').timeit(number=2))

    # n=200
    Q = np.random.rand(200, 200)
    b = np.random.rand(200)


    print("Timing for solveQ for matrix of size 200;")
    print(timeit.Timer('rank1pert_inv',
                       'from __main__ import rank1pert_inv').timeit(number=1))

    print("Timing for np.linalg.solve")
    print(timeit.Timer('np.linalg.solve', 'import numpy as np').timeit(number=2))


    # n=400
    Q = np.random.rand(400, 400)
    b = np.random.rand(400)

    print("Timing for solveQ for matrix of size 400;")
    print(timeit.Timer('rank1pert_inv',
                       'from __main__ import rank1pert_inv').timeit(number=1))

    print("Timing for np.linalg.solve")
    print(timeit.Timer('np.linalg.solve', 'import numpy as np').timeit(number=2))


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.
    
    :param Q: an mxn-dimensional numpy array whose columns are the
    orthonormal vectors
    
    :return P: an mxm-dimensional numpy array containing the projector
    """

    #Using that P=QQ* to construct an orthogonal projector
    P = np.dot(Q,Q.conjugate().T)
    
    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.
    
    :param V: an mxn-dimensional numpy array whose columns are the
    vectors u_1,u_2,...,u_n.
    
    :return Q: an lxm-dimensional numpy array whose columns are an
    orthonormal basis for the subspace orthogonal to U.
    """

    # Using numpy qr algorithm to obtain complete factorisation and extracting orthogonal columns
    Q , R = np.linalg.qr(V, mode='complete')
    m, n = R.shape
    Q = Q[:,n:] #First n columns gives orthonormal basis for Col(A), taking remaining columns which are orthogonal to these

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm.
    
    :param A: mxn numpy array
    
    :return Q: mxn numpy array
    :return R: nxn numpy array
    """
    
    # Extracting dimensions from A and initalising Q and R matrices, and vector v
    m, n = A.shape
    Q = np.zeros([m,n], dtype=A.dtype)
    R = np.zeros([n,n], dtype=A.dtype)

    # Implementing Gram-Schmidt pseudocode exploiting vector operations
    for j in range(n):
        v = A[:, j]
        R[:, j] = Q.conjugate().T.dot(v)
        v = v - Q.dot(R[:, j])
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v/R[j, j]
    return Q, R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm.
    
    :param A: mxn numpy array
    
    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    # Extracting dimensions of A and initialising Q and R
    m, n = A.shape
    R = np.zeros([n, n], dtype=A.dtype)
    Q = 1.0*A # Copying A into V equivalent to first loop
    
    # Implementing modified Gram-Schmidt algorithm exploiting matrix-vector multiplication
    for i in range(n):
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:,i] = Q[:,i]/R[i,i]

        R[i, i+1:] = Q[:, i].conjugate().T.dot(Q[:, i+1:])
        Q[:, i+1:] = Q[:, i+1:] - np.outer(Q[:, i], R[i, i+1:])
    return Q, R

def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that Ahat[:, 0:k] = A[:, 0:k], and A[:, k] is orthogonal to the columns of A[:, 0:k].
    
    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise
    
    :return R: nxn numpy array
    """
    # Extracting shape of A and initialising R
    m, n = A.shape
    R = np.eye(n, dtype = A.dtype)
    
    #Computing entries for Rk along row k - the other rows are standard unit vectors
    R[k,k] = 1/np.linalg.norm(A[:,k])
    q_k = A[:,k]/R[k,k]
    R[k, k+1:] = -1*q_k.conjugate().T.dot(A[:, k+1:])

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular formulation with Rs provided from GS_modified_get_R.
    
    :param A: mxn numpy array
    
    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    R = np.eye(n, dtype = A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        np.dot(A, Rk, out=A)
        np.dot(R, Rk, out=R)
    R = np.linalg.inv(R)
    return A, R
