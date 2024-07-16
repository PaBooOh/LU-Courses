
import numpy as np
import matplotlib.pyplot as plt


N = []
compound_coord = []
n_size = 10000
np.random.seed(123)

for p in range(2,102,2):
    N.append(p*n_size)
    r = np.random.randint(0, p*n_size, n_size)
    compound_coord.append(len(set(r))/n_size)

poly_weights = np.polyfit(N, compound_coord, deg=3)
polynomial = np.poly1d(poly_weights)
print(polynomial)