#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 09:24:01 2021

@author: philippeangot
"""

# Packages
# from numpy import *

from math import floor
from numpy import exp, sin, cos, linspace
# import numpy as np
import matplotlib.pyplot as plt

# Data
# Initial data (t0,y0), time step, final time T, number of iterations niter

t0 = 0
y0 = 1.
step = 0.01  # or dt = step
T = 10
# niter = int((T-t0)/step)
niter = floor((T-t0)/step)

#
# Explicit Euler method to solve a Cauchy problem
# for a first-order scalar O.D.E.:
# y' = f(t,y), for t in [t0,T]
# with I.C.: (t0,y0)
#
# N.B. Basic implementation using lists of different data of heterogeneous types
#

def euler_exp(f, t0, y0, step, T):
    t = t0
    y = y0
    list_t = [t]
    list_y = [y]
    niter = floor((T-t0)/step)
#    for _ in range(niter):
    for n in range(niter):
        y += step * f(t, y)
        t += step
        list_t.append(t)
        list_y.append(y)
    return list_t, list_y

# Definition of function f (by anonymous function "lambda")
f = lambda t, y: cos(t) * y

# Solution with Euler scheme
list_t, list_y = euler_exp(f, t0, y0, step, T)
# with another step dt
list_t01, list_y01 = euler_exp(f, t0, y0, 0.1, T)


# Plotting the exact and approximate solutions
plt.figure(0)

# Plot the exact solution
# x = arange(t0, T+step, step)
x = linspace(t0, T, niter+1)
y = exp(sin(x))
plt.plot(x, y, '-', color='blue', label='Exact solution')


# Plot the approximate solution
plt.plot(list_t, list_y, '--', color='red', label='Approx. sol. - $\delta t=0.01$')
plt.plot(list_t01, list_y01, '-', color='red', label='Approx. sol. - $\delta t=0.1$')
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Solution of a scalar O.D.E. with Euler scheme')

plt.legend(loc = "best")
plt.savefig('edo_sol.pdf', bbox_inches='tight')
plt.show()


# Error in Linf(0,T) norm

list_err = abs(y - list_y)
err_Linf = max(list_err)
print('Error in Linf norm =', err_Linf, 'for step =', step)

# Error calculation in Linf(0,T) norm versus step dt for different T

dt = [1., 1.e-01, 1.e-02, 1.e-03, 1.e-04]
print('step =', dt)

Tf = 1
Err_Linf_Tf1 = []
for i in range(len(dt)):
    step = dt[i]
    L_t, Y_ap = euler_exp(f, t0, y0, step, Tf)
    Y_ex = exp(sin(L_t))
    err_max = max(abs(Y_ex - Y_ap))
    Err_Linf_Tf1.append(err_max)
print('Error in Linf(0,T) norm for final time T =', Tf)
print(Err_Linf_Tf1)

Tf = 5
Err_Linf_Tf5 = []
for i in range(len(dt)):
    step = dt[i]
    L_t, Y_ap = euler_exp(f, t0, y0, step, Tf)
    Y_ex = exp(sin(L_t))
    err_max = max(abs(Y_ex - Y_ap))
    Err_Linf_Tf5.append(err_max)
print('Error in Linf(0,T) norm for final time T =', Tf)
print(Err_Linf_Tf5)

Tf = 10
Err_Linf_Tf10 = []
for i in range(len(dt)):
    step = dt[i]
    L_t, Y_ap = euler_exp(f, t0, y0, step, Tf)
    Y_ex = exp(sin(L_t))
    err_max = max(abs(Y_ex - Y_ap))
    Err_Linf_Tf10.append(err_max)
print('Error in Linf(0,T) norm for final time T =', Tf)
print(Err_Linf_Tf10)


# Plot the error graphs in loglog scale

slope_1 = dt

plt.figure(1)
plt.loglog(dt, slope_1, '--', color='green', label='Order $1$ in $\delta t$: $\mathcal{O}(\delta t)$')
plt.loglog(dt, Err_Linf_Tf1, '*-', color='magenta', label='T = 1')
plt.loglog(dt, Err_Linf_Tf5, '*-', color='blue', label='T = 5')
plt.loglog(dt, Err_Linf_Tf10, '*-', color='red', label='T = 10')
plt.grid(True)
plt.xlabel('step $\delta t$')
plt.ylabel('$\|E_{h}\|_{L^{\infty}(0,T)}$')
plt.title('Error in $L^{\infty}(0,T)$ norm versus $\delta t$ for different $T$')

plt.legend(loc = "best")
plt.savefig('edo_error.pdf', bbox_inches='tight')
plt.show()

