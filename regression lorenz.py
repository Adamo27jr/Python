# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:54:55 2024

@author: user
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Générer l'attracteur de Lorenz
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Paramètres de simulation
t_span = (0, 50)
initial_state = [1.0, 1.0, 1.0]
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Résolution des équations différentielles
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

# Extraire les solutions
x, y, z = sol.y

# Méthode de box counting pour calculer la dimension fractale
def box_count(data, box_size):
    # Normalisation des données pour qu'elles soient dans le cube unité
    data_min = np.min(data, axis=1)
    data_max = np.max(data, axis=1)
    normalized_data = (data - data_min[:, None]) / (data_max[:, None] - data_min[:, None])
   
    # Calcul du nombre de boîtes
    grid = np.floor(normalized_data / box_size).astype(int)
    unique_boxes = np.unique(grid, axis=1)
    return unique_boxes.shape[1]

# Différentes tailles de boîte
box_sizes = np.logspace(-2.5, 0, num=50)
counts = np.array([box_count(np.vstack((x, y, z)), size) for size in box_sizes])

# Régression linéaire pour trouver la dimension fractale
log_box_sizes = np.log(1 / box_sizes)
log_counts = np.log(counts)

coeffs = np.polyfit(log_box_sizes, log_counts, 1)
fractal_dimension = coeffs[0]

# Afficher la dimension fractale
print(f"Dimension fractale de l'attracteur de Lorenz: {fractal_dimension:.3f}")

# Afficher le schéma de la régression linéaire
plt.figure(figsize=(8, 6))
plt.scatter(log_box_sizes, log_counts, label='Données')
plt.plot(log_box_sizes, np.polyval(coeffs, log_box_sizes), color='red', label='Régression linéaire')
plt.xlabel('log(1/box_size)')
plt.ylabel('log(counts)')
plt.title('Box-Counting Dimension de l\'Attracteur de Lorenz')
plt.legend()
plt.show()