import itertools
import numpy as np
from numpy import *
from utilities import *
import random

def linear_system_find(x, m):
    """
    Given points x and function values f, compute the matrix for the linear system Ma=F for degree m.
    
    :param x: An n dimensional numpy array of floats
    
    :return M: An nxm dimensional numpy array of floats
    """

    # Initialising matrix for linear system
    M = np.zeros((x.size, m+1), dtype=x.dtype)
    n = len(x)

    # Producing matrix M for Ma=F
    for i in range(n):
        for j in range(m+1):
            M[i,j] = (x[i])**j

    return M

# Defining function to computing matrix for linear system M and solving for coefficents a - to apply for all questions
def coefficients_solve(x, f, m):
    """
    Given points x and function values f, compute coefficients for a n approximating polynomial of degree m.
    
    :param x: An n dimensional numpy array of floats
    :param f: An n dimensional numpy array of floats
    :param m: An integer
    
    :return a: An m dimensional numpy array of floats
    """

    # Initialising matrix for linear system
    M = linear_system_find(x, m)

    # Computing coefficients for polynomial using householder least squares method
    a = householder_ls(M, f)

    return(a)

# Made to create a list of every combination of +/- perturbations amounts for use in sensitivity test
def alternate_perturbations(epsilons):
    """
        Given a length n, compute the possible 2^n +/- alternations of epsilon
        
        :param epsilons: An n dimensional numpy array of floats
        
        :return perturbations: A list of numpy arrays of length n
        """
    
    # Initialising list of alternations and creating list of all 2^n arrays of length n with 1 or -1 as entries
    n = len(epsilons)
    perturbations = []
    # For this I adapted code to generate 2^n binary strings from here https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value as the problem is isomorphic
    alternate = list(itertools.product([1, -1], repeat=n))

    # Adding to list of alternating perturbations amounts e.g. array[(-epsilon, epsilon, epsilon, ..., -epsilon, epsilon, -epsilon)] is one element of this list
    for i in range(2**n):
        alternation = np.asarray(epsilons*alternate[i])
        perturbations.append(alternation)

    return perturbations

# Constructed to perturb given data in question
def perturb_points(f, perturbation):
    """
    Given points f and some perturbation values, perturbates each point f by that amount.
    :param f: An n dimensional numpy array of floats representing f(xi)
    :param perturbation: An n dimensional numpy array of floats representing the perturbations for each xi
    :return f_perturbed:
    """

    # Initialising array of perturbed points and finding number of points being perturbed
    n = len(f)
    f_perturbed = np.zeros(n)

    for i in range(n):
        f_perturbed[i] = f[i] + perturbation[i]

    return f_perturbed

def relative_condition_single_perturbation(y, y_perturbed, x, x_perturbed):
    """
    Computing the relative condition number of y given perturbations on x.
    :param y: An n dimensional numpy array of floats
    :param x: An m dimensional numpy array of floats
    :return k: A float
    """

    normed_y = np.linalg.norm(y)
    normed_delta_y = np.linalg.norm(y - y_perturbed)
    normed_x = np.linalg.norm(x)
    normed_delta_x = np.linalg.norm(x - x_perturbed)

    k = (normed_delta_y/normed_y)/(normed_delta_x/normed_x)

    return k

# Constructed function to compute the sensitivities of coefficients and polynomial given perturbed f
def sensitivities(x, f, epsilons, a, F):
    """
    Given points (x,f) and perturbation sizes in epsilons, perturbates each point f by some amount epsilon in epsilons (both in up and down directions). Returns the
    sensitivity of the coefficients and polynomial - measured using the l2 norm between the perturbed values and unperturbed values - and which perturbation values achieved this.
    
    :param x: An n dimensional numpy array of floats representing xi
    :param f: An n dimensional numpy array of floats representing f(xi)
    :param epsilons: An n dimensional numpy array of floats representing the perturbations for each xi
    :param a: An m dimensional numpy array of floats representing the coefficients of unperturbed polynomial
    :param F: A p dimensional numpy array of floats representing points on unperturbed polynomial F(x)
    
    :return coefficient_relative_conditions: A float giving the relative condition of the coefficients
    :return polynomial_relative_conditions: A float giving the relative condition of the polynomial
    """

    # Initialising array of perturbed points and finding number of points being perturbed

    n = len(f)
    coefficient_relative_conditions = np.zeros(2**n)
    polynomial_relative_conditions = np.zeros(2**n)

    # Creating list of 2^n possible up and down perturbations given input epsilon
    alternating_epsilons = alternate_perturbations(epsilons)

    # Simulating 2^n perturbations of each point of f (each point up and down by some given amount epsilon)
    for i in range(len(alternating_epsilons)):
        perturbation = alternating_epsilons[i]

        f_perturbed = perturb_points(f, perturbation)

        # Returning coefficients for interpolating polynomial of degree m from lowest to highest order
        m = len(a) - 1 # Degree of polynomial 1 less due to constant term
        a_perturbed = coefficients_solve(x, f_perturbed, m)
        a_perturbed_flip = np.flip(a_perturbed) #Reversing order for polyval

        # Finding relative condition of a given perturbation of f and perturbed coefficients then appending to a list
        coefficient_relative_conditions[i] = relative_condition_single_perturbation(a, a_perturbed, f, f_perturbed)

        # Computing dense set of points (x,F(x)) of polynomial of degree 10 given perturbed coefficients
        x_plot = np.linspace(-1, 1, len(F))
        F_perturbed = np.polyval(a_perturbed_flip, x_plot)

        # Finding relative condition of F given perturbation of f then appending to a list
        polynomial_relative_conditions[i] = relative_condition_single_perturbation(F, F_perturbed, f, f_perturbed)
    
    # Computing the relative conditions over all perturbations
    coefficient_relative_condition = max(coefficient_relative_conditions)
    polynomial_relative_condition = max(polynomial_relative_conditions)

    return coefficient_relative_condition, polynomial_relative_condition
