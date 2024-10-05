#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 08:48:50 2022

@author: philippeangot
"""

#
# Solution of elliptic Boundary-value problem (B.V.P.)
#
# Finite Difference (F.D.) scheme with 3 points for the Poisson P.D.E.
# - u''(x) = f(x), x in ]0, L[
# with Dirichlet B.C.: u(0) = u_0 and u(L) = u_L
#
# with given source (or sink) term f in L^{2}(0,L)
#

# Packages and modules
# import time
from math import floor
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sp
import scipy.sparse.linalg as spla


# Data
L = 1
u_0 = 0
u_L = 0

dx = 0.01   # or mesh step h = dx

print('mesh step h = dx =', dx)

# Space domain of calculation: [0, L] and mesh
nx = floor(L/dx)
X_mesh = np.linspace(0, L, nx+1, dtype=np.float64)
print('Number of FV cells nx =', nx)
# print(X_mesh)
print('Number of mesh points (nx+1) =', len(X_mesh))
print('Number of interior points N = (nx-1) =', nx-1)

# Definition of several source functions f 
f_0 = lambda x: 0
f_1 = lambda x: 1
f_k = lambda x, k: np.sin(k * np.pi * x)

# Some exact solutions
uex_1 = lambda x: -0.5 * x**2 + (0.5 * L + (u_L-u_0)/L) * x + u_L
uex_2 = lambda x, k: np.sin(k * np.pi * x)/(k * np.pi)**2       # for u_0 = u_L = 0

# Choice of functions q, f and exact solution

# solution 1
f_h = np.ones(nx+1)
Uex_h = uex_1(X_mesh)

# solution 2 with k = 1
#f_h = f_k(X_mesh, 2)
#Uex_h = uex_2(X_mesh, 2)

#
# Assembly of linear system's sparse tridiagonal matrix: A_h
# size N x N with N = (nx-1) number of interior points
#
def Matrix_Lap(dx,nx):
    Tridiag = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(nx-1,nx-1), format='csr')
    Matrix = Tridiag/(dx*dx)
    return Matrix

A_h = Matrix_Lap(dx,nx)
# print('A_h =', A_h.toarray())

# Right-hand side vector with contribution of Dirichlet B.C.
b_h = np.copy(f_h[1:nx])
b_h[0] = b_h[0] + u_0/(dx*dx)
b_h[nx-2] = b_h[nx-2] + u_L/(dx*dx)

#
# Solution of linear system A_h u_h = b_h with 'scipy.sparse.linalg.spsolve'
#
# spsolve(A, b, permc_spec=None, use_umfpack=True)
# b: ndarray or sparse matrix = matrix or vector representing the right hand side
# If a vector, b.shape must be (n,) or (n, 1).
#

u_h = spla.spsolve(A_h, b_h)


# Residual r_h = b_h - A_h u_h in Linf norm
r_h = b_h - A_h.dot(u_h)
res_Linf = max(abs(r_h))
print('Residual r_h in Linf norm =', res_Linf, 'for step dx =', dx)


# Approximate solution with Dirichlet B.C.
U_h = np.zeros(nx+1)
U_h[1:nx] = u_h.copy()
U_h[0] = u_0
U_h[nx] = u_L


# Plot the exact and approximate solutions

plt.close(fig='all')

plt.figure(1)
# plt.clf()
plt.plot(X_mesh, Uex_h, '-', color='blue', label='Exact solution')
plt.plot(X_mesh, U_h, '-.', color='red', label='F.D. sol. - $\delta x=10^{-2}$')

plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.legend(loc = 'best')
# plt.legend(loc = 'upper center')
plt.title('Elliptic boundary-value problem of diffusion')
plt.savefig('Poisson_sol2.pdf', bbox_inches='tight')
plt.show()


# Error in Linf(0,L) norm
# err_Linf_Up = np.max(np.abs(Uex_h - U_h))
err_Linf = max(abs(Uex_h - U_h))
print('Error in Linf(0,L) norm =', err_Linf, 'for step h = dx =', dx)

# Error in L2(0,L) norm
err_L2 = np.sqrt(dx * np.sum(np.abs(Uex_h - U_h)**2))
print('Error in L2(0,L) norm =', err_L2, 'for step h = dx =', dx)


# Convergence speed and order 2 of precision

# Array of mesh steps h
step_h = np.array([1.e-01, 0.5e-01, 1.e-02, 0.5e-02, 1.e-03, 0.5e-03, 1.e-04], dtype=float)
print('space steps h = dx =', step_h)

# List of error norms to compute
Err_Linf = []
Err_L2 = []

for i in range(len(step_h)):
    dx = step_h[i]
    nx = floor(L/dx)
    X_mesh = np.linspace(0, L, nx+1)
# choice of data and solution
# solution 1
    # f_h = np.ones(nx+1)
    # Uex_h = uex_1(X_mesh)
# solution 2
    f_h = f_k(X_mesh, 1)
    Uex_h = uex_2(X_mesh, 1)
# Matrix and rhs
    A_h = Matrix_Lap(dx,nx)
    b_h = np.copy(f_h[1:nx])
    b_h[0] = b_h[0] + u_0/(dx*dx)
    b_h[nx-2] = b_h[nx-2] + u_L/(dx*dx)
# Solution
    u_h = spla.spsolve(A_h, b_h)
    U_h = np.zeros(nx+1)
    U_h[1:nx] = u_h.copy()
    U_h[0] = u_0
    U_h[nx] = u_L
# Residual norm
    r_h = b_h - A_h.dot(u_h)
    res_Linf = max(abs(r_h))
    print('Residual r_h in Linf norm =', res_Linf, 'for step dx =', dx)
# Error norms
    err_Linf = max(abs(Uex_h - U_h))
    err_L2 = np.sqrt(dx * np.sum(np.abs(Uex_h - U_h)**2))
    Err_Linf.append(err_Linf)
    Err_L2.append(err_L2)


# Plot the error graphs in loglog scale

slope_1 = step_h
slope_2 = step_h**2
# slope_2 = 10. * step_h**2

plt.figure(2)
plt.loglog(step_h, slope_2, '-.', color='magenta', label='$\mathcal{O}(\delta x^{2})$: order $2$ in $\delta x$')
plt.loglog(step_h, Err_Linf, 'x-', color='blue', label='$L^{\infty}(0,1)$ norm')
plt.loglog(step_h, Err_L2, 'x-', color='red', label='$L^{2}(0,1)$ norm')
plt.grid(True)
plt.xlabel('mesh step: $h=\delta x$')
plt.ylabel('Error norm: $\|\overline{u}_{h}-u_{h}\|$')
plt.title('Convergence rate: error norm versus step $\delta x$')
plt.legend(loc = "best")
plt.savefig('Poisson_error_sol2.pdf', bbox_inches='tight')
plt.show()



