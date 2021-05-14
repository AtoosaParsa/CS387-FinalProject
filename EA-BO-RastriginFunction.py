# Optimizing Rastrigin's function, using a combination of EA and BO
from typing import Counter
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

# n-dimensional Rastrigin function, global optima at 0
# X is a list [x_i]
def rastrigin(X):
    # X is a n dimensional row vector
    A = 10
    dim = 20
    X = X.reshape((dim, 1))
    return dim*A + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])

kernel = GPy.kern.Matern52(input_dim=20, variance=1.0, lengthscale=0.2)
bds = [{'name': 'X1', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X2', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X3', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X4', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X5', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X6', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X7', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X8', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X9', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X10', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X11', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X12', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X13', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X14', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X15', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X16', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X17', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X18', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X19', 'type': 'continuous', 'domain': (-5, 5)},
{'name': 'X20', 'type': 'continuous', 'domain': (-5, 5)}]


optimizer = BayesianOptimization(f=rastrigin, 
                                 domain=bds,
                                 model_type='GP',
                                 kernel=kernel,
                                 acquisition_type ='EI',
                                 maximize=False)

t0 = time.time()
optimizer.run_optimization(max_iter=100, max_time=3600)
t1 = time.time()
print("time:")
print(t1-t0)
t_bo = t1-t0

#optimizer.plot_acquisition()
optimizer.plot_convergence()

# get the candidate solutions and their evaluations
ins = optimizer.get_evaluations()[0]
outs = optimizer.get_evaluations()[1]
outputs = outs.flatten()
# sort in descending order
indices = np.argsort(outputs)
ins_sorted = ins[indices]
ins_sorted_inverse = ins_sorted[::-1]
outputs.sort()
reverse_array = outputs[::-1]

#f = open('BO-Rastrigin-best-20.dat', 'ab')
#pickle.dump(reverse_array , f)
#f.close()

print("The minimum value obtained by the function was:")
print(optimizer.fx_opt)
print(optimizer.x_opt)


from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from deap import creator, base, tools, algorithms
import pickle

counter_var = 0

# n-dimensional Rastrigin function, global optima at 0
# X is a list [x_i]
def rastrigin2(X, A=10):
    dim=20
    return dim*A + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X]),

def genFunky(icls, ndim, ins):
    global counter_var

    genome = list()
    # choose randomly from the last 50 solutions from BO
    #inverse_ins = ins[::-1]
    #candidate = inverse_ins[random.randint(0, 51)]
    #candidate = inverse_ins[counter_var] # change input to ins
    candidate = ins[counter_var]
    counter_var += 1
    for i in np.arange(0, ndim):
        genome.append(candidate[i])

    return icls(genome)

# dimention of the optimization
dim = 20
popSize = 50
genSize = 2000

random.seed(a=1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr", random.random)
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=dim)
# bias the initial population
toolbox.register("individual", genFunky, creator.Individual, dim, ins[::-1])#ins_sorted) # ins)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", rastrigin2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2) # indpb: Independent probability for each attribute to be mutated
toolbox.register("select", tools.selTournament, tournsize=3)

#population = toolbox.population(n=popSize)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

toolbox.pop_size = popSize
toolbox.max_gen = genSize
toolbox.mut_prob = 0.1

logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

hof = tools.HallOfFame(1, similar=np.array_equal) #can change the size

def run_ea(toolbox, stats=stats, verbose=True, hof=hof):
    pop = toolbox.population(n=toolbox.pop_size)
    pop = toolbox.select(pop, len(pop))

    return algorithms.eaSimple(pop, toolbox,
                                     cxpb=0,#1-toolbox.mut_prob, : no crossover
                                     mutpb=toolbox.mut_prob, 
                                     stats=stats, 
                                     ngen=toolbox.max_gen, 
                                     verbose=verbose,
                                     halloffame=hof)

t0 = time.time()
res,log = run_ea(toolbox, stats=stats, verbose=True, hof=hof)
t1 = time.time()
t_ea = t1-t0

# print info for best solution found:
print("-----")
print(len(hof))
best = hof.items[0]
print("-- Best Individual = ", best)
print("-- Best Fitness = ", best.fitness.values)

avg = log.select("avg")
std = log.select("std")
min_ = log.select("min")
max_ = log.select("max")

plt.figure(figsize=(6.4,4.8))
plt.plot(reverse_array, color='red', label='Bayesian Optimization')
plt.plot(list(range(105, 105+2000+1)), min_, color='blue', label='Evolutionary Algorithm')
#plt.fill_between(list(range(0, toolbox.max_gen+1)), avg-std, avg+std, color='cornflowerblue', alpha=0.2)
plt.xlabel("Generations/Iterations")
plt.ylabel("Objective Value")
plt.title("EA + BO", fontsize='small')
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.ylim(bottom=0)
plt.show()

print("time:")
print(t_bo+t_ea)

#f = open('GA-Rastrigin-best-20.dat', 'ab')
#pickle.dump(min_ , f)
#f.close()

#f = open('GA-Rastrigin-avg-20.dat', 'ab')
#pickle.dump(avg , f)
#f.close()