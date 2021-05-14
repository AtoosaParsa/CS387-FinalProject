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
optimizer.run_optimization(max_iter=50, max_time=3600)
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

# n-dimensional Schwefel function, global optima at 0
# X is a list [x_i]
def Schwefel2(X, A=10):
    dim=2
    return dim * 418.9829 - sum([x * np.sin(np.sqrt(np.abs(x))) for x in X]),

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
dim = 2
popSize = 50
genSize = 200

random.seed(a=1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr", random.random)
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=dim)
# bias the initial population
toolbox.register("individual", genFunky, creator.Individual, dim, ins_sorted) # ins)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", Schwefel2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=1, sigma=1, indpb=0.2) # indpb: Independent probability for each attribute to be mutated
toolbox.register("select", tools.selTournament, tournsize=3)

#population = toolbox.population(n=popSize)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

toolbox.pop_size = popSize
toolbox.max_gen = genSize
toolbox.mut_prob = 0.2

logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

hof = tools.HallOfFame(1, similar=np.array_equal) #can change the size

def run_ea(toolbox, stats=stats, verbose=True, hof=hof):
    pop = toolbox.population(n=toolbox.pop_size)
    pop = toolbox.select(pop, len(pop))

    return algorithms.eaSimple(pop, toolbox,
                                     cxpb=1-toolbox.mut_prob, #: no crossover
                                     mutpb=toolbox.mut_prob, 
                                     stats=stats, 
                                     ngen=toolbox.max_gen, 
                                     verbose=verbose,
                                     halloffame=hof)

t0 = time.time()
res,log = run_ea(toolbox, stats=stats, verbose=True, hof=hof)
t1 = time.time()
t_ea = t1-t0
print("total time:")
print(t_bo + t_ea)

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
plt.plot(list(range(55, 55+200+1)), min_, color='blue', label='Evolutionary Algorithm')
#plt.fill_between(list(range(0, toolbox.max_gen+1)), avg-std, avg+std, color='cornflowerblue', alpha=0.2)
plt.xlabel("Generations/Iterations")
plt.ylabel("Objective Value")
plt.title("Evolutionary Algorithm + Bayesian Optimization", fontsize='small')
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.ylim(bottom=0)
plt.show()

#f = open('GA-Rastrigin-best-20.dat', 'ab')
#pickle.dump(min_ , f)
#f.close()

#f = open('GA-Rastrigin-avg-20.dat', 'ab')
#pickle.dump(avg , f)
#f.close()