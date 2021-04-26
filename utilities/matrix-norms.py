import numpy as np

def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.
    
    :param A: an mxn-dimensional numpy array
    
    :return o2norm: operator 2-norm
    """

    # A^TAx = \lambda x so Ax = \sqrt{\lambda} x
    # Computing eigenvalues of A
    M = A.T.dot(A)
    eigenvalues = np.linalg.eig(M)[0]

    # Computing norm from known eigenvalues
    o2norm = np.sqrt(np.max(eigenvalues))

    return o2norm

def inequality(A, x):
    """
    Given a real mxn dimensional matrix A and n dimensional vector x,
    return true or false whether ||Ax||<=||A|| ||x||
    
    :param A: an mxn numpy array
    :param B: an lxk-dimensional numpy array
    
    :return : A boolean
    """

    truth = operator_2_norm(A.dot(x)) <= operator_2_norm(A)*operator_2_norm(x)

    return truth
        
def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.
    
    :return A: an mxn-dimensional numpy array
    
    :param ncond: condition number in 2-norm
    """

    #  Finding quadratic variation eigenvalues
    L = A.T.dot(A)
    eigenvalue = np.linalg.eig(L)[0]

    # Computing condition number
    ncond = np.sqrt(np.amax(eigenvalue)/np.amin(eigenvalue))

    return ncond
