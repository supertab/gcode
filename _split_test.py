import numpy as np
#
A = np.array([[1,2, 3],[2,3, 3], [7,1, 2]])
# target: [[0.5, 1.5], [1.5, 2.5], [1.5, 2.5], [2.5, 3.5]]
e = 0.5
# '0,2': specify dim
Merge = np.r_['0,2', A-e, A+e] #np.r_ and np.c_ tile array
