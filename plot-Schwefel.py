# plot Rastrigin function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
import math

X = np.linspace(-500, 500, 100)     
Y = np.linspace(-500, 500, 100)
X, Y = np.meshgrid(X, Y) 

Z =  2 * 418.9829 - X * np.sin(np.sqrt(np.abs(X))) - Y * np.sin(np.sqrt(np.abs(Y)))

# plot 3D
fig = plt.figure() 
ax = fig.gca(projection='3d') 
CS = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.nipy_spectral, linewidth=0.08, antialiased=True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Schwefel Function", fontsize='small')
plt.grid(color='silver', linestyle='-', linewidth=0.2)
# add colorbar
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Objective Value')
plt.tight_layout()
plt.show()

# plot the contour
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.contour(X, Y, Z, 100, cmap=cm.nipy_spectral)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Schwefel Function", fontsize='small')
plt.grid(color='silver', linestyle='-', linewidth=0.2)
plt.tight_layout()
plt.show()
