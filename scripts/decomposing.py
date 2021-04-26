from numpy import *
from utilities import *
from decomposing_tools import *

print('Run python3 test_decomposing.py to test functions')
print('------------------------------------------------------------------------------')

# Uploading dataset
A = load('values.npy')

# Outputting the shape of A for report
print('The shape of A is')
print(A.shape)
print('------------------------------------------------------------------------------')

# Using modified Gram-Schmidt algorithm to compute QR decomposition of A
Q, R = GS_modified(A)

# Saving full matrices Q and R to csv file
np.savetxt("Q_q1.csv", Q, delimiter=',')
np.savetxt("R_q1.csv", R, delimiter=',')

# Displaying Q and R for report
print('Q = \n', Q)
print('------------------------------------------------------------------------------')
print('R= \n', R)
print('------------------------------------------------------------------------------')

# Testing orthonormality of Q aand outputting for report
print('Q*Q gives \n', Q.conjugate().T.dot(Q))
print('------------------------------------------------------------------------------')

# Finding rows that are almost zero (using function in q1_tools.py) and outputting for report
R_zeros, R_zero_rows = find_approx_zero(R, 1e-5)
print('The bottom-right 4x4 submatrix of R is \n', R[-4:,-4:])
print('------------------------------------------------------------------------------')
print('The number of rows approximately equal to zero is ',R_zeros)
print('These rows are (from 0 to 99) \n', R_zero_rows)
print('------------------------------------------------------------------------------')
