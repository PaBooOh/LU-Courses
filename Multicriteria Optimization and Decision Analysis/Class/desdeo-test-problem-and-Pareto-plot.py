# # Basics of desdeo-emo
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd

from desdeo_problem import variable_builder, ScalarObjective, MOProblem
#from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs import NSGAIII

#from sklearn.datasets import load_iris, load_boston, load_wine
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
import plotly.graph_objects as go

# Define Objective function
def f_1(x):
    r = x[:,0]
    h = x[:,1]
    area_cylinder = 2 * np.pi * r**2 + 2 * np.pi * r *h
    #area_icecreamcone = np.pi * r * np.sqrt( h**2 + r**2)
    return area_cylinder

def f_2(x):
    r = x[:, 0]
    h = x[:, 1]
    volume_cylinder= (np.pi) * r**2 * h
    #volume_icecreamcone = 1.0 / 3.0 * (np.pi) * r ** 2 * h
    #volume_filled_icecreamcone=1.0 /3.0 * (np.pi) * r**2 * h + 0*1/2*4.0 / 3.0 *np.pi *r**3
    return -volume_cylinder

#def f_3(x):
#    term1 = ((x[:, 0] + (2 * x[:, 1]) - 1) ** 2) / 175
#    term2 = ((-x[:, 0] + 2* x[:, 1]) ** 2) / 17
#    return term1 + term2 - 13


# Note that the expected input `x` is two dimensional. It should be a 2-D numpy array.

# ## Create Variable objects

# [lower bound x1, lower bound x2], [upper  bound x1, upper bound x2]
list_vars = variable_builder(['x', 'y'],
                             initial_values = [0,0],
                             lower_bounds=[0, 0],
                             upper_bounds=[10, 10])
#list_vars


# ## Create Objective objects


f1 = ScalarObjective(name='f1', evaluator=f_1)
f2 = ScalarObjective(name='f2', evaluator=f_2)
list_objs = [f1, f2]


# ## Create the problem object
problem = MOProblem(variables=list_vars, objectives=list_objs)


# ## Using the EAs
#
# Pass the problem object to the EA, pass parameters as arguments if required.
evolver = NSGAIII(problem,
                  n_iterations=10,
                  n_gen_per_iter=100,
                  population_size=100)

while evolver.continue_evolution():
    evolver.iterate()

# ## Visualization of optimized decision variables and objective values using Plotly

# individuals: decision variable vectors
# solutions: points in objective space that approximate Parto front
individuals, solutions = evolver.end()

pd.DataFrame(solutions).to_csv("ParetoFront.csv")
import matplotlib.pyplot as plt

# Add a random sample to the plot
from numpy import random
X = random.rand(1000,1)*(10.0)
Y = random.rand(1000,1)*(10.0)
Z = [];

len_data = 1000+len(solutions)
labels = np.zeros(len_data)

# Scatterplot F1, F2

randomsample = np.hstack((X,Y)) #matrix 1000x2

for i in range(1000):
  F1randomsample=f_1(randomsample)
  F2randomsample=f_2(randomsample)
print(randomsample)

plt.scatter(F1randomsample,-F2randomsample)
plt.scatter(solutions[:,0],-solutions[:,1])
plt.show()

# Scatterplot X1, X2

plt.scatter(X,Y)
plt.scatter(individuals[:,0],individuals[:,1])
plt.show()

s1=solutions[:,0]
F1=np.concatenate((F1randomsample,s1))
F1=F1/np.max(F1)

s2=solutions[:,1]
F2=np.concatenate((F2randomsample,s2))
F2=F2/np.max(-F2)

d1=individuals[:,0]
X1=np.concatenate((X.flatten(),d1))
X1=X1/np.max(X1)

d2=individuals[:,1]
X2=np.concatenate((Y.flatten(),d2))
X2=X2/np.max(X2)

for i in range(1000):
    labels[i]='0'
for i in range(1001,1000+len(d2)):
    labels[i]='1'

data = pd.read_csv(r'ParetoFront.csv', sep=',')
df = pd.DataFrame([labels,F1,-F2,X1,X2],["Label", "F1", "F2", "X1", "X2"])
dft=df.transpose()
print(dft)

pd.plotting.parallel_coordinates(dft,'Label', color=["lime", "tomato"])
plt.show()

cols = ["Label", "F1", "F2", "X1", "X2"]

fig = px.parallel_coordinates(dft, color="Label", dimensions=cols,
                              title="Geomertrical Shape Pareto Parallel Coorinates Plot")
fig.show()
