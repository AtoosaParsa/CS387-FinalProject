# Optimizing Schwefel's function, with EA using DEAP package
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from deap import creator, base, tools, algorithms
import pickle
import time

# n-dimensional Schwefel function, global optima at 0
# X is a list [x_i]
def Schwefel(X, A=10):
    dim=2
    return dim * 418.9829 - sum([x * np.sin(np.sqrt(np.abs(x))) for x in X]),

# dimention of the optimization
dim = 2
popSize = 50
genSize = 200

random.seed(a=1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr", random.uniform, -500, 500)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=dim)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", Schwefel)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2) # indpb: Independent probability for each attribute to be mutated
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=popSize)

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
                                     cxpb=1-toolbox.mut_prob, #: no crossover
                                     mutpb=toolbox.mut_prob, 
                                     stats=stats, 
                                     ngen=toolbox.max_gen, 
                                     verbose=verbose,
                                     halloffame=hof)

t0 = time.time()

res,log = run_ea(toolbox, stats=stats, verbose=True, hof=hof)

t1 = time.time()
print("time:")
print(t1-t0)

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
plt.plot(min_, color='blue')
#plt.fill_between(list(range(0, toolbox.max_gen+1)), avg-std, avg+std, color='cornflowerblue', alpha=0.2)
plt.xlabel("Generations")
plt.ylabel("Objective Value")
plt.title("Fitness of the Best Individual in the Population", fontsize='small')
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.ylim(bottom=0)
plt.show()

f = open('GA-Schwefel-best-2.dat', 'ab')
pickle.dump(min_ , f)
f.close()

f = open('GA-Schwefel-avg-2.dat', 'ab')
pickle.dump(avg , f)
f.close()