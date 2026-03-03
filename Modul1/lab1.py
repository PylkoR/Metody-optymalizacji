import numpy as np
import cvxpy as cp

###############################
## method 1
###############################
x1 = cp.Variable()
x2 = cp.Variable()
objective = cp.Minimize(x1 + 0.5*x2)
constraints = [ x1 + x2 <= 2,
x1 + x2/4 <= 1,
x1 - x2 <= 2,
x1/4 + x2 >= -1,
x1 + x2 >= 1,
-x1 + x2 <= 2,
x1 + x2/4 == 1/2,
-1 <= x1,
x1 <= 1.5,
-1/2 <= x2,
x2 <= 1.25 ]
p1 = cp.Problem(objective, constraints)
p1.solve()
print("--------------")
print(x1.value)
print(x2.value)
print("--------------")