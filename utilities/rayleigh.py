import numpy as np
import numpy.random as random
from utilities import *
import matplotlib.pyplot as plt

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.
    
    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.
    
    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.
    
    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.
    
    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return array([[0.68557183+0.46550108j,  0.12934765-0.1622676j,
                   0.24409518+0.25335939j],
                  [0.1531015 + 0.66678983j,  0.45112492+0.18206976j,
                   -0.02633966+0.43477693j],
                  [-0.10817164-1.16879196j, -0.18446849+0.03755672j,
                   0.06430325-0.44757084j]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.
    
    :return B3: a 3x3 numpy array.
    """
    return array([[0.46870499+0.37541453j,  0.19115959-0.39233203j,
                   0.12830659+0.12102382j],
                  [0.90249603-0.09446345j,  0.51584055+0.84326503j,
                   -0.02582305+0.23259079j],
                  [0.75419973-0.52470311j, -0.59173739+0.48075322j,
                   0.51545446-0.21867957j]])

def rayleigh_quotient(x, A):
    """
    Function to compute the Rayleigh quotiant of an eigenpair.
    
    :param x: m dimensonal numpy array
    :param A: An mxm real symmetric matrix
    
    :return R: A float, Rayleigh quotient
    """

    R = (x.conj().T.dot(A).dot(x))/(x.conj().T.dot(x))

    return R


def investigate_rayleigh(m):
    """
    Function to investigate perturbations to eigenvectors on rayleigh coefficient
    
    :param m: A natural number
    """

    #Generate Hermitian Matrix
    randR = np.random.randn(m, m)
    randC = np.random.randn(m,m)*1j
    randM =  randR + randC
    A = randM.conjugate().T.dot(randM)

    # Extract eigenvalue and eigenvector
    lamd, psi = np.linalg.eig(A)
    lamd, psi = lamd[0], psi[:, 0]

    # Initialise for perturbations
    epsilons = np.arange(0.0001, 0.1, 0.0001)
    deltas = np.zeros(epsilons.size)
    r = np.random.rand(m)
    r = r/np.linalg.norm(r)
    
    # Compute quotients
    for epsilon in epsilons:
        psi_pert = psi + epsilon*r
        rayleigh = rayleigh_quotient(psi_pert, A)
        deltas[i]= np.linalg.norm(rayleigh - lamd)

    # Plotting
    plt.figure(0)
    plt.loglog(epsilons, deltas)
    plt.xlabel('$\epsilon$')
    plt.ylabel('$||r(x) - \lambda||_2$')
    plt.title('Error in Rayleigh Estimate')
    plt.savefig('investigate_rayleigh.png')
    plt.show()
