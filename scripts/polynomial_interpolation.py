from numpy import *
from utilities import *
import matplotlib.pyplot as plt
import random
from polynomial_interpolation_tools import *

# Setting seed for reproducability
random.seed(2020)

print('To test implementations - run python3 test_polynomial_interpolation.py')
print('-------------------------------------------------------------------------------')

# Given data
x = arange(-1, 1.1, 0.2)
f = 0.0*x
f[3:6] = 1

# Returning coefficients for interpolating polynomial of degree 10 (using function from q2_tools.py) from constant to highest order term,
# and reversing order for use in polyval
a_10 = coefficients_solve(x, f, 10)
a_10_flip = np.flip(a_10)

# Computing dense set of points (x,F(x)) on polynomial of degree 10 with given coefficients
x_plot_10 = np.linspace(-1, 1, 200)
F_10 = np.polyval(a_10_flip, x_plot_10)

# Plotting polynomial F of degree 10
plt.figure(0)
plt.plot(x_plot_10, F_10, color='fuchsia')
plt.plot(x, f, 'bx') #Given data points
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Plot of Degree 10 Polynomial, and the \n Observed Datapoints, for Function F(x).')
plt.savefig('2c_m10.png')
plt.show()

# Perturbations of amount 0.02
epsilons_002 = 0.02* np.ones(len(f))

# Computing up/down perturbations of amount 0.02 for f and corresponding coefficient and polynomial relative conditions for m=10
a_10_sensitivity, F_10_sensitivity = sensitivities(x, f, epsilons_002, a_10, F_10)

# Outputting results for report
print('Trying perturbations on each point f of amount 0.02 and obtain:')
print('The relative condition of the coefficients is, for m=10,', a_10_sensitivity)
print('The relative condition of the polynomial is, for m=10,', F_10_sensitivity)
print('-------------------------------------------------------------------------------')

# Generating random set of perturbation amounts for each point
epsilons_random = np.ones(len(f))
for i in range(len(epsilons_random)):
    epsilons_random[i] = random.uniform(0,0.02)

# Computing f perturbations pointwise by random amount up and down and corresponding coefficient and polynomial sensitivities for m=10
a_10_sensitivity_random, F_10_sensitivity_random = sensitivities(x, f, epsilons_random, a_10, F_10)

# Outputting results for report
print('Trying the following randomly selected perturbation amounts on U[0,0.2) for each point f \n', epsilons_random)
print('Perturbing these up and down we obtain:')
print('The relative condition of the coefficients is, for m=10,', a_10_sensitivity_random)
print('The relative condition of the polynomial is, for m=10,', F_10_sensitivity_random)
print('-------------------------------------------------------------------------------')

# Returning coefficients for polynomial of degree 7 from constant to highest order term (using function from q2_tools.py)
# and reversing for use in polyval
# Using same starting data as in questions 2b and 2c
a_7 = coefficients_solve(x, f, 7)
a_7_flip = np.flip(a_7) # Reversing order for use in polyval

# Computing dense set of points (x,F(x)) on polynomial of degree 7 with given coefficients
x_plot_7 = x_plot_10
F_7 = np.polyval(a_7_flip, x_plot_7)

# Plotting polynomial F of degree 7
plt.figure(2)
plt.plot(x_plot_7, F_7, color='chocolate')
plt.plot(x, f, 'bx')  # Given data points
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Plot of Degree 7 Polynomial, and the \n Observed Datapoints, for Function F(x).')
plt.savefig('2e_m7.png')
plt.show()

# Computing up/down perturbations of f and corresponding coefficient and polynomial sensitivities for m=7
# using same perturbation values as for m=10
a_7_sensitivity, F_7_sensitivity = sensitivities(x, f, epsilons_002, a_7, F_7)

#  Outputting results for report
print('Trying perturbations on each point f of amount 0.02 up and down and obtain:')
print('The relative condition of the coefficients is, for m=7,', a_7_sensitivity)
print('The relative condition of the polynomial is, for m=7,', F_7_sensitivity)
print('-------------------------------------------------------------------------------')

# Computing f points pointwise by random amount up and down and corresponding coefficient and polynomial sensitivities for m=7
# using the same perturbation values as for m=7
a_7_sensitivity_random, F_7_sensitivity_random = sensitivities(x, f, epsilons_random, a_7, F_7)

# Outputting results for report
print('Trying the following randomly selected perturbation amounts on U[0,0.2) for each point f \n', epsilons_random)
print('Perturbing these up and down we obtain:')
print('The relative condition of the coefficients is, for m=7,', a_7_sensitivity_random)
print('The relative condition of the polynomial is, for m=7,', F_7_sensitivity_random)
print('-------------------------------------------------------------------------------')
