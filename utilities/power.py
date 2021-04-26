import numpy as np
from utilities import *

def pow_it(A, x0, tol, maxit, store_iterations=False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either
    ||r|| < tol where
    r = Ax - lambda*x,
    or the number of iterations exceeds maxit.
    
    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence
    of power iterates, instead of just the final iteration. Default is
    False.
    
    :return x: an m dimensional numpy array containing the final iterate, or
    if store_iterations, an m x (maxit) dimensional numpy array containing all
    the iterates.
    :return lambda0: the final eigenvalue.
    """

    # Initialising given input data
    lambda0 = x0.conjugate().T.dot(A).dot(x0)
    r = A.dot(x0) - lambda0*x0

    if store_iterations == True:
        # Initialising
        x = np.zeros((m, maxit))
        x[:, 0] = x0
        lambdas = np.zeros(maxit)
        lambdas[0] = lambda0
        
        # While r is above tolerance
        i = 0
        while np.linalg.norm(r) > tol:

            # Break if exceeds iterations
            i = i + 1
            if i > maxit:
                break

            # Apply power algorithm
            else:
                w = A.dot(x[:, i-1])
                x[:, i] = w/np.linalg.norm(w)
                lambdas[i] = x[:, i].conjugate().T.dot(A).dot(x[:, i])

                # Compute tolerance
                r = A.dot(x[:, i]) - lambda0*x[:, i]

        return x, lambda0

    else:
        x = x0

        # While r is above tolerance
        i = 0
        while np.linalg.norm(r) > tol:

            # Break if exceeds iterations
            i = i + 1
            if i > maxit:
                break

            # Apply power algorithm
            else:
                w = A.dot(x)
                x = w/np.linalg.norm(w)
                lambda0 = x.conjugate().T.dot(A).dot(x)
            
                # Compute tolerance
                r = A.dot(x) - lambda0*x
    
        return x, lambda0


def inverse_it(A, x0, mu, tol, maxit, store_iterations=False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.
    
    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence
    of inverse iterates, instead of just the final iteration. Default is
    False.
    
    :return x: an m dimensional numpy array containing the final iterate, or
    if store_iterations, an mxmaxit dimensional numpy array containing
    all the iterates.
    :return l: a floating point number containing the final eigenvalue
    estimate, or if store_iterations, an m dimensional numpy array containing
    all the iterates.
    """

    # Initialising given input data
    m, n = A.shape
    lambda0 = x0.conjugate().T.dot(A).dot(x0)
    r = A.dot(x0) - lambda0*x0
    I = np.eye(m, dtype = complex)

    if store_iterations == True:
        #Initilaising
        x = np.zeros((m, maxit), dtype=complex)
        x[:, 0] = x0/np.linalg.norm(x0)
        lambdas = np.zeros(maxit)
        lambdas[0] = lambda0

        # While r is above tolerance
        i = 0
        while np.linalg.norm(r) > tol:

            # Break if exceeds iterations
            i = i + 1
            if i > maxit:
                break

            # Apply Rayleigh quotient iteration
            else:
                B = A - mu*I
                w = solve_LUP(B, x[:, i-1])
                x[:, i] = w/np.linalg.norm(w)
                lambdas[i] = x[:, i].conjugate().T.dot(A).dot(x[:, i])

                # Compute tolerance
            r = A.dot(x[:, i]) - lambda0*x[:, i]
            
        return x, lambdas

    else:
        x = x0/np.linalg.norm(x0)

        # While r is above tolerance
        i = 0
        while np.linalg.norm(r) > tol:

            # Break if exceeds iterations
            i = i + 1
            if i > maxit:
                break

            # Apply Rayleigh quotient iteration
            else:
                B = A - mu*I
                w = solve_LUP(B, x)
                x = w/np.linalg.norm(w)
                lambda0 = x.conjugate().T.dot(A).dot(x)

                # Compute tolerance
                r = A.dot(x) - lambda0*x
            
        return x, lambda0
    

def rq_it(A, x0, tol, maxit, store_iterations=False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.
    
    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence
    of inverse iterates, instead of just the final iteration. Default is
    False.
    
    :return x: an m dimensional numpy array containing the final iterate, or
    if store_iterations, an mxmaxit dimensional numpy array containing
    all the iterates.
    :return l: a floating point number containing the final eigenvalue
    estimate, or if store_iterations, an m dimensional numpy array containing
    all the iterates.
    """

    # Initialising given input data
    m, n = A.shape
    lambda0 = x0.conjugate().T.dot(A).dot(x0)
    r = A.dot(x0) - lambda0*x0
    I = np.eye(m, dtype=complex)

    if store_iterations == True:
        x = np.zeros((m, maxit), dtype=complex)
        x[:, 0] = x0/np.linalg.norm(x0)
        lambdas = np.zeros(maxit)
        lambdas[0] = lambda0

        # While r is above tolerance
        i = 0
        while np.linalg.norm(r) > tol:

            # Break if exceeds iterations
            i = i + 1
            if i > maxit:
                break
            
            # Apply Rayleigh quotient iteration
            else:
                B = A - lambdas[i-1]*I
                w = solve_LUP(B, x[:, i-1])
                x[:, i] = w/np.linalg.norm(w)
                lambdas[i] = x[:, i].conjugate().T.dot(A).dot(x[:, i])

                # Compute tolerance
                r = A.dot(x[:, i]) - lambda0*x[:, i]
            
        return x, lambdas

    else:
        x = x0/np.linalg.norm(x0)

        # While r is above tolerance
        i = 0
        while np.linalg.norm(r) > tol:

            # Break if exceeds iterations
            i = i + 1
            if i > maxit:
                break

            # Apply Rayleigh quotient iteration
            else:
                B = A - lambda0*I
                w = solve_LUP(B, x)
                x = w/np.linalg.norm(w)
                lambda0 = x.conjugate().T.dot(A).dot(x)

                # Compute tolerance
                r = A.dot(x) - lambda0*x

        return x, lambda0
