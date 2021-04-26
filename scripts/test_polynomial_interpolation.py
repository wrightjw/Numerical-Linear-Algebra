import pytest
from utilities import *
from polynomial_interpolation_tools import *
from numpy import random
import numpy as np

@pytest.mark.parametrize('m, n', [(10,11), (7,11), (5,10)])
def test_coefficients_solve(m, n):
    """
    Test to see if obtained coefficients match that of a true polynomial these points came from.
    
    :param m: An integer corresponding to degree of polynomial
    :param n: An integer corresponding to number of points taken from that polynomial
    
    :return bool: Pass or fail
    """

    # Finding n points f on a known polynomial of degree m, assuming n >= m
    x_test = np.random.rand(n)
    f_test = 0.0*x_test
    a_test = np.random.rand(m)

    for i in range(n):
        for n in range(m):
            f_test[i] = f_test[i] + a_test[n]*x_test[i]**n

    # Reverse engineering for coefficients from these points
    a = coefficients_solve(x_test, f_test, n)

    # Checking coefficients are the same
    assert(np.linalg.norm(a-a_test) < 1.0e-6)

@pytest.mark.parametrize('n', [1,2,3,4,5])
def test_alternate_perturbations(n):
    """
    Test to see if all up/down combinations of perturbations achieved.
    
    :param n: An integer for size of array to find combinations of
    
    :return bool: Pass or fail
    """
    # Creating random array of values and finding all possible alternations
    epsilons = np.random.rand(n)
    perturbations = alternate_perturbations(epsilons)

    # Checking found all possible combinations
    assert(len(perturbations) == 2**n)


@pytest.mark.parametrize('n', [2, 3, 11])
def test_perturb_points(n):
    """
    Test to see if points f are perturbed correctly.
    
    :param n: An integer to define number of points being perturbed
    
    :return bool: Pass or fail
    """

    # Creating random array of points and perturbations
    f = np.random.rand(n)
    epsilons = np.random.rand(n)
    f_perturbed = perturb_points(f, epsilons)

    # Checking to see if points in punctured ball of perturbations
    assert(((f-f_perturbed) != np.zeros(n)).all())
    assert((f-f_perturbed <= epsilons).all())
    

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
