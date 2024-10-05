
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Paramètres physiques
L1 = 1.0  # Longueur de la première barre
L2 = 1.0  # Longueur de la deuxième barre
m1 = 1.0  # Masse de la première barre
m2 = 1.0  # Masse de la deuxième barre
g = 9.81  # Accélération due à la gravité

# Fonction décrivant les équations du mouvement
def equations(t, y):
    theta1, theta2, omega1, omega2 = y
    dydt = [
        omega1,
        omega2,
        (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * m2 * (omega2 ** 2 * L2 + omega1 ** 2 * L1 * np.cos(theta1 - theta2))) / (L1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))),
        (2 * np.sin(theta1 - theta2) * (omega1 ** 2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + omega2 ** 2 * L2 * m2 * np.cos(theta1 - theta2))) / (L2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    ]
    return dydt

# Conditions initiales
y0 = [np.pi / 4, np.pi / 2, 0, 0]  # Angles initiaux et vitesses angulaires

# Intervalle de temps
t_span = [0, 10]

# Résolution des équations du mouvement
sol = solve_ivp(equations, t_span, y0, t_eval=np.linspace(0, 10, 500))

# Trajectoires des pendules
theta1 = sol.y[0]
theta2 = sol.y[1]
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Tracé des trajectoires
plt.plot(x1, y1, label='Barre 1')
plt.plot(x2, y2, label='Barre 2')
plt.xlabel('Position horizontale')
plt.ylabel('Position verticale')
plt.title('Trajectoires des pendules collés')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.savfig('Double Pendule.png')
plt.show()
