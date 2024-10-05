#
# Solve Cauchy initial-value problem (I.V.P.) for a first-order system of O.D.E.:
# Y' = f(t,Y), for t in [t0,T]
# with I.C. (t0,Y0)
#

# Packages
# from numpy import *

from math import floor
# from numpy import exp, sin, cos, sinh, cosh, linspace
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

# Data
# Initial data (t0,Y0), time step, final time T, number of iterations niter

t0 = 0
Y0 = [5, 3]
step = 0.001  # or dt = step
T = 10
niter = floor((T-t0)/step)

#
# Explicit Euler method for a nonlinear system of O.D.E.:
# Y' = f(t,Y), for t in [t0,T]
# with I.C. (t0,Y0)
#
# N.B. Basic implementation using lists of different data of heterogeneous types
#

# Compact version for a differential system
def euler_exp(f, t0, Y0, step, T):
    t = t0
    Y = Y0
    list_t = [t]
    list_Y = [Y]
    niter = floor((T-t0)/step)
    for n in range(niter):
        Y = [x + step * u for x, u in zip(Y, f(t, Y))]
        t += step
        list_t.append(t)
        list_Y.append(Y)
    return list_t, list_Y

# Definition of function f
# Lotka-Volterra nonlinear differential system
# modeling a "predator-prey" problem in population dynamics

f = lambda t, Y: [Y[0] * (3 - 2*Y[1]), Y[1] * (Y[0] - 4)]
# f = lambda t, Y: [Y[0] * (3 - Y[1]), Y[1] * (Y[0] - 4)]


# Solution with Euler scheme
list_t, list_Y = euler_exp(f, t0, Y0, step, T)
print(len(list_t))
X_ap = [Y[0] for Y in list_Y]
Y_ap = [Y[1] for Y in list_Y]

# list_t, X_ap, Y_ap = euler_exp_syst(f_syst0, f_syst1, t0, Y0, step, T)
print(len(X_ap), len(Y_ap))

L_t01, L_Y01 = euler_exp(f, t0, Y0, 0.01, T)
X_ap01 = [Y[0] for Y in L_Y01]
Y_ap01 = [Y[1] for Y in L_Y01]


# Plotting the approximate solutions
plt.figure(0)

# Plot the approximate solution
plt.plot(list_t, X_ap, '-', color='blue', label='Prey $X$ - $\delta t=0.001$')
plt.plot(list_t, Y_ap, '-', color='red', label='Predator $Y$ - $\delta t=0.001$' )
plt.plot(L_t01, X_ap01, '--', color='blue', label='$X$ - $\delta t=0.01$')
plt.plot(L_t01, Y_ap01, '--', color='red', label='$Y$ - $\delta t=0.01$' )
plt.xlabel('$t$')
plt.ylabel('Prey $X$ or Predator $Y$')
plt.title('Solution of Lotka-Volterra system with Euler scheme')

plt.legend(loc = "best")
plt.savefig('Lotka-Volterra_sol.pdf', bbox_inches='tight')
plt.show()


# Phase diagram

plt.figure(1)
plt.plot(X_ap, Y_ap, '-', color='blue', label='Euler scheme -- $\delta t=0.001$')
plt.plot(X_ap01, Y_ap01, '--', color='red', label='Euler scheme -- $\delta t=0.01$')
plt.xlabel('Prey $X$')
plt.ylabel('Predator $Y$')
plt.title('Lotka-Volterra phase diagram for $T=10$')
plt.legend(loc = "best")
plt.savefig('Lotka-Volterra_phase.pdf', bbox_inches='tight')
plt.show()


#
# Numerical solution with 'solve_ivp' module of SciPy
# New version for 'odeint'
#
# solution = solve_ivp(fun, t_span, y0, method='RK45', max_step=None, t_eval=None, dense_output=False, events=None, vectorized=False, args=None)
# Here Runge-Kutta RK45 scheme
#

stepmax = 0.01
sol_rk45 = solve_ivp(f, [t0, T], Y0, method='RK45', max_step=stepmax)
t_rk45 = sol_rk45.t
X_rk45 = sol_rk45.y[0]
Y_rk45 = sol_rk45.y[1]


# Plotting the approximate solutions
plt.figure(2)

# Plot the approximate solution
plt.plot(t_rk45, X_rk45, '-', color='blue', label='RK45 sol. $X_h$')
plt.plot(t_rk45, Y_rk45, '-', color='red', label='RK45 sol. $Y_h$')
plt.xlabel('$t$')
plt.ylabel('Prey $X(t)$ or Predator $Y(t)$')
plt.title('Solution of Lotka-Volterra system with RK45 scheme - $\delta t_{max}=0.01$')

plt.legend(loc = 'best')
plt.savefig('Lotka-Volterra_RK45.pdf', bbox_inches='tight')
plt.show()


# Phase diagram

plt.figure(3)
plt.plot(X_rk45, Y_rk45, '-', color='blue', label='RK45 scheme -- $\delta t_{max}=0.01$')

plt.xlabel('Prey $X$')
plt.ylabel('Predator $Y$')
plt.title('Lotka-Volterra phase diagram with RK45 scheme for $T=10$')
plt.legend(loc = "best")
plt.savefig('Lotka-Volterra_RK45_phase.pdf', bbox_inches='tight')
plt.show()



