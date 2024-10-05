#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:32:32 2024

@author: philippeangot
"""

#
# Solve Cauchy initial-value problem (I.V.P.) for a first-order system of O.D.E.:
# Y' = f(t,Y), for t in [t0,T]
# with I.C. (t0,Y0)
#
#
# Numerical solution with 'solve_ivp' module of SciPy
# New version for 'odeint'
#
# solution = solve_ivp(fun, t_span, y0, method='RK45', max_step=None, t_eval=None, dense_output=False, events=None, vectorized=False, args=None)
# Here Runge-Kutta RK4-5 scheme with adaptive time step
#


# Packages and modules
# from numpy import *

# from math import floor
import numpy as np
import matplotlib.pyplot as plt

# from scipy.integrate import odeint
from scipy.integrate import solve_ivp



# Data
# Initial data (t0,Y0), time step, final time T, number of iterations niter

# t0 = 0         # Initial time
# Y0 = [1, 0]    # Initial values
# step = 0.01    # or dt = step
# T = 10         # Final time
# niter = floor((T-t0)/step)

#
# Solution of the simple pendulus system with no damping
#

t0 = 0
Y0 = [3*np.pi/4, 0]
step = 0.01  # or dt = step
T = 10

# Definition of function f for pendulus (pendulum) system with no damping
f_pend = lambda t, Y: [Y[1], np.sin(Y[0])]

# Approximate solution with Runge-Kutta RK4-5 scheme
solp = solve_ivp(f_pend, [0, 10], Y0, method='RK45', max_step=0.01)
tp = solp.t
xp = solp.y[0]
yp = solp.y[1]

# Plot the approximate solution
plt.figure(0)
plt.plot(tp, xp, '-', color='blue', label='Angle $X$ - $\delta t_{max}=0.01$')
plt.plot(tp, yp, '-', color='red', label='Angular velocity $Y=X^{\prime}$')

plt.xlabel('$t$')
plt.ylabel('Angle $X(t)$ or angular velocity $Y(t):=X^{\prime}(t)$')
plt.title('Simple pendulus system (no damping) with RK4-5 scheme')

plt.legend(loc = "best")
plt.savefig('pendulus_rk45.pdf', bbox_inches='tight')
plt.show()

# Phase diagram

plt.figure(1)
plt.plot(xp, yp, '-', color='blue', label='RK4-5 sol. - $\delta t_{max}=0.01$')
plt.plot(np.pi, 0, '*', color='blue', label='center $(\pi,0)$')
plt.plot(3*np.pi/4, 0, '*', color='green', label='$(3\pi/4,0)$: max. angle, zero velocity')
plt.plot(np.pi, 0.78, '*', color='magenta', label='$(\pi,0.78)$: zero angle, max. velocity')

plt.xlabel('Angle $X$')
plt.ylabel('Angular velocity $Y:=X^{\prime}$')
plt.title('Phase diagram of simple pendulus (no damping) for $T=10$')
plt.legend(loc = "best")
plt.savefig('pendulus_rk45_phase.pdf', bbox_inches='tight')
plt.show()



#
# Strange attractor of Lorenz (1963): a dynamical system for chaos and turbulence
# in fluid mechanics and meteorology
#
# model of Rayleigh-Bénard natural convection
# sigma = Prandtl number Pr
# rho = reduced Rayleigh number = Ra/Ra_c
# x(t): proportional to intensity of convective motion
# y(t): proportional to temperature difference between up- and down-currents
# z(t): proportional to difference of vertical temperature profile with linear profile
#

# Definition of Lorenz system
def Lorenz_system(t, Y, sigma, rho, beta):
    x = Y[0]
    y = Y[1]
    z = Y[2]
    dx_dt = sigma * (y-x)
    dy_dt = rho * x - y - x*z
    dz_dt = x*y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Parameters of Lorenz dynamical system
sigma = 10
beta = 8/3
#rho = 28      # rho_c in [24.29, 24.30] spawn threshold for occuring chaotic regime 
rho = 100
# rho = 22

# Initial conditions
x0 = 1
y0 = 1
z0 = 1

# Solution with Runge-Kutta RK4-5 scheme
sol = solve_ivp(Lorenz_system, [0, 40], [x0, y0, z0], method='RK45',
                args=(sigma, rho, beta), max_step=0.01)
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]

# Plot the phase diagram Z = Z(X, Y) in 3D
fig = plt.figure(2)
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z, ':', color='blue', label='RK4-5 sol. - $\delta t_{max}=0.01$')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Lorenz (1963) attractor with RK4-5 scheme - $T=40$')
# Lorenz attractor: $\sigma=10$, $\beta=8/3$, $\rho=28$
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('Lorentz_attractor.pdf', bbox_inches='tight')
plt.show()

