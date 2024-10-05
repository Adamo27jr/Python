# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:40:47 2024

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Système de Lorenz
def lorenz(t, state, sigma=10.0, beta=8.0/3.0, rho=28.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Générer les données du système de Lorenz
def generate_lorenz_data(t_max=50, dt=0.01):
    t = np.arange(0, t_max, dt)
    initial_state = [1.0, 1.0, 1.0]
    solution = solve_ivp(lorenz, [0, t_max], initial_state, t_eval=t, method='RK45')
    return solution.y.T

# Convertir les données en une image binaire
def data_to_binary_image(data, img_size=(500, 500)):
    x, y, z = data.T
    x = (x - x.min()) / (x.max() - x.min()) * (img_size[0] - 1)
    y = (y - y.min()) / (y.max() - y.min()) * (img_size[1] - 1)
   
    # S'assurer que les indices sont dans les limites
    x = np.clip(x.round().astype(int), 0, img_size[0] - 1)
    y = np.clip(y.round().astype(int), 0, img_size[1] - 1)
   
    image = np.zeros(img_size, dtype=bool)
    image[y, x] = 1  # Corrigé l'indexation ici
    return image

# Algorithme de box counting
def box_count(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)

    return np.count_nonzero(S)

def fractal_dimension(Z):
    Z = (Z > 0)

    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p)).astype(int)

    sizes = 2 ** np.arange(np.log2(n), 1, -1).astype(int)

    counts = []
    for size in sizes:
        counts.append(box_count(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

if __name__ == "__main__":
    data = generate_lorenz_data()
    binary_image = data_to_binary_image(data)

    plt.imshow(binary_image, cmap='binary')
    plt.title("Binary Image of Lorenz Attractor")
    plt.show()

    fractal_dim = fractal_dimension(binary_image)
    print("Fractal Dimension:", fractal_dim)
