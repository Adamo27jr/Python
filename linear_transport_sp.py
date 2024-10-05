#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 19:11:49 2022

@author: adamb
"""

#
# Finite Difference (F.D.) explicit schemes for the linear transport P.D.E.
# with given velocity c >0 or <0:
# u_t + c u_x = 0, t in ]0,T], x in [0, L]
# with I.C.: u(0,x) = u_0(x), x in [0, L]
#
#
# Explicit Upwind, Lax-Friedrichs (1954) and Lax-Wendroff (1960) schemes
# using optimized implementation with no loop in space
# and coded with a Sparse matrix for fast matrix-vector products in the right-hand side (rhs)
#
# providing CPU times per time step proportional to number of unknowns, i.e. O(1/h)
#
# tested for several regular or discontinuous initial conditions
#

# Packages and modules
# import time
from math import floor
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# import scipy as spy
# import scipy.linalg as la
import scipy.sparse as sp
# import scipy.sparse.linalg as spla


# Data
t0 = 0
c = 1
T = 2
L = 4

dx = 1.e-02  # or h = dx
print('space step h = dx =', dx)

#
# C.F.L. number fixed: nu := c dt/dx
# C.F.L. Linf or L2 stability condition: |c| dt/dx =< 1
#

nu = 0.5
dt = nu*dx/c
niter = floor((T-t0)/dt)
print('time step dt =', dt)
print('Number of time iterations niter =', niter)

# Practical space domain of calculation: [0, L] and mesh
nx = floor(L/dx)
X_mesh = np.linspace(0, L, nx+1)
print('Number of FV cells nx =', nx)
# print(X_mesh)
print('Number of mesh points (nx+1) =', len(X_mesh))

# Definition of function u_0 for several initial conditions (I.C.)
# and possible upstream or downstream B.C.

u0_1 = lambda x: np.exp(-5 * (x - 0.5)**2)        # u0_1 regular
u0_2 = lambda x: (1 + np.tanh(10 * (x - 0.5)))/2  # u0_2 regular
u0_3 = lambda x: H(x - 0.5)                       # u0_3 Heaviside step function : discontinuous
u0_4 = lambda x: H(x - 0.5) - H(x - 1.)           # Supp u0_4 = [0.5,1] compact

# Vector form of Heaviside function H(X) for arrays of floats with 'np.where'
def H(X):
    nX = np.size(X)
    H = np.zeros(nX)
    index = np.where(X > 0.)
    H[index] = 1.
    return H


#
# Definitions of explicit F.D. schemes using ndarray class of floats
# # from 'scipy.linalg' (alias 'la') compiled with BLAS/LAPACK
# # => faster than 'numpy.linalg'
# To avoid slower loop "for" in space...!
#
# Construction of RHS Sparse tridiagonal matrix in format 'csr_matrix'
# Use 'sp.diags' or 'sp.spdiags' functions from 'scipy.sparse' module
# N.B. Compressed Sparse Row (CSR) format specially suitable for fast matrix-vector products
#

# Upwind explicit scheme for c > 0
# N.B. Unstable downwind explicit scheme for c < 0
Mrhs_Up = sp.diags([nu, 1 - nu, 0], [-1, 0, 1], shape=(nx+1,nx+1), format='csr')

# Lax-Friedrichs scheme
Mrhs_LF = sp.diags([0.5*(1 + nu), 0, 0.5*(1 - nu)], [-1, 0, 1], shape=(nx+1,nx+1), format='csr')

# Lax-Wendroff scheme
Mrhs_LW = sp.diags([0.5*nu*(nu + 1), 1 - nu**2, 0.5*nu*(nu - 1)], [-1, 0, 1], shape=(nx+1,nx+1), format='csr')


#
# Approximate solutions with F.D. explicit schemes
#

# choice of I.C.
# Initial and exact solutions: u(t,x) = u_0(x - c t)

u0 = u0_3
U0 = np.copy(u0(X_mesh))
Uex = np.copy(u0(X_mesh - c * T))


#
# Upwind explicit scheme for c > 0
#
Un = np.copy(U0)            # Needs a copy to avoid changing U0 by later changing Un

# Main loop in time
## N.B.: U_up = np.dot(Mrhs_Upsp, Un)  # Not Ok for sparse, needs to convert to dense

for n in range(niter):
    U_up = Mrhs_Up.dot(Un)        # Sparse matrix-vector (row by column) product
#    Upstream B.C. at x=0 with exact sol. u(t_{n+1},0)
    U_up[0] = u0(- c * (n+1)*dt)
    Un = U_up

#
# Lax-Friedrichs (1954) scheme
#
Un = np.copy(U0)
# Main loop in time
for n in range(niter):
    U_LF = Mrhs_LF.dot(Un)
#    Upstream (x=0) and downstream (x=L) B.C. with exact sol. u(t_{n+1},x)
    U_LF[0] = u0(- c * (n+1)*dt)
    U_LF[nx] = u0(L - c * (n+1)*dt)
    Un = U_LF

#
# Lax-Wendroff (1960) scheme
#
Un = np.copy(U0)
# Main loop in time
for n in range(niter):
    U_LW = Mrhs_LW.dot(Un)
#    Upstream (x=0) and downstream (x=L) B.C. with exact sol. u(t_{n+1},x)
    U_LW[0] = u0(- c * (n+1)*dt)
    U_LW[nx] = u0(L - c * (n+1)*dt)
    Un = U_LW


# Plot the exact and approximate solutions

plt.close(fig='all')

plt.figure(1)
# plt.clf()
plt.plot(X_mesh, Uex, '-', color='blue', label='Exact sol. at $t=2$')
plt.plot(X_mesh, U_up, '-.', color='blue', label='Upwind')
plt.plot(X_mesh, U_LF, '-.', color='red', label='Lax-Friedrichs (1954)')
plt.plot(X_mesh, U_LW, ':', color='magenta', label='Lax-Wendroff (1960)')

plt.xlabel('$x$')
plt.ylabel('$u(t,x)$')
plt.legend(loc = 'best')
# plt.legend(loc = 'upper center')
plt.title('Linear transport with velocity $c=1$ - $\delta x=10^{-2}$, $CFL=0.5$')
# plt.savefig('transport_sol4_cfl-0.1.pdf', bbox_inches='tight')
plt.show()


# Discrete norms of the error
# Error in Linf(0,L) norm at t = T

# err_Linf_Up = np.max(np.abs(Uex - U_up))
err_Linf_Up = max(abs(Uex - U_up))
err_Linf_LF = max(abs(Uex - U_LF))
err_Linf_LW = max(abs(Uex - U_LW))
print('Error in Linf(0,L) norm: at T =', T, 'for dx =', dx, 'and CFL =', nu)
print('Upwind Error in Linf norm =', err_Linf_Up)
print('L-F Error in Linf norm =', err_Linf_LF)
print('L-W Error in Linf norm =', err_Linf_LW)

# Error in L1(0,L) norm at t = T

err_L1_Up = dx * np.sum(np.abs(Uex - U_up))
err_L1_LF = dx * np.sum(np.abs(Uex - U_LF))
err_L1_LW = dx * np.sum(np.abs(Uex - U_LW))
print('Error in L1(0,L) norm: at T =', T, 'for dx =', dx, 'and CFL =', nu)
print('Upwind Error in L1 norm =', err_L1_Up)
print('L-F Error in L1 norm =', err_L1_LF)
print('L-W Error in L1 norm =', err_L1_LW)

# Error in L2(0,L) norm at t = T

err_L2_Up = np.sqrt(dx * np.sum(np.abs(Uex - U_up)**2))
err_L2_LF = np.sqrt(dx * np.sum(np.abs(Uex - U_LF)**2))
err_L2_LW = np.sqrt(dx * np.sum(np.abs(Uex - U_LW)**2))
print('Error in L2(0,L) norm: at T =', T, 'for dx =', dx, 'and CFL =', nu)
print('Upwind Error in L2 norm =', err_L2_Up)
print('L-F Error in L2 norm =', err_L2_LF)
print('L-W Error in L2 norm =', err_L2_LW)



