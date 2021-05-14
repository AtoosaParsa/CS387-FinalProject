# Optimizing Schwefel's function, using BO with GPyOpt package
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import matplotlib.pyplot as plt
import time
import pickle
import random

random.seed(a=1)

# n-dimensional Schwefel function, global optima at 0
# X is a list [x_i]
def Schwefel(X):
    # X is a n dimensional row vector
    dim = 2
    X = X.reshape((dim, 1))
    return dim * 418.9829 - sum([x * np.sin(np.sqrt(np.abs(x))) for x in X])

kernel = GPy.kern.Matern52(input_dim=2, variance=1.0, lengthscale=0.5)
bds = [{'name': 'X1', 'type': 'continuous', 'domain': (-500, 500)},
{'name': 'X2', 'type': 'continuous', 'domain': (-500, 500)}]


optimizer = BayesianOptimization(f=Schwefel, 
                                 domain=bds,
                                 model_type='GP',
                                 kernel=kernel,
                                 acquisition_type ='EI',
                                 maximize=False)

t0 = time.time()
optimizer.run_optimization(max_iter=200, max_time=7200)
t1 = time.time()
print("time:")
print(t1-t0)

#optimizer.plot_acquisition()
optimizer.plot_convergence()

# get the candidate solutions and their evaluations
ins = optimizer.get_evaluations()[0]
outs = optimizer.get_evaluations()[1]
outputs = outs.flatten()
# sort in descending order
outputs.sort()
reverse_array = outputs[::-1]
#print(reverse_array)

plt.figure(figsize=(6.4,4.8))
plt.plot(reverse_array, color='blue')
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.title("Best Candidate Solution at each Iteration", fontsize='small')
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.ylim(bottom=0)
plt.show()

f = open('BO-Schwefel-best-2.dat', 'ab')
pickle.dump(reverse_array , f)
f.close()

print("The minimum value obtained by the function was:")
print(optimizer.fx_opt)
print(optimizer.x_opt)
