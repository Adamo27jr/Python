#
# Finite Difference (F.D.) explicit schemes for the linear transport P.D.E.
# with given velocity c >0 or <0:
# u_t + c u_x = 0, t in [0,T], x in [0, L]
# with I.C.: u(0,x) = u_0(x), x in [0, L]
#
# Performance comparison of 4 different implementations of the Upwind scheme
#

# Packages and modules
import time
from math import floor
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sp

# Data
t0 = 0
c = 1
T = 2
L = 4

dx = 1.e-03  # or h = dx
print('space step h = dx =', dx)

#
# C.F.L. number fixed: nu := c dt/dx
# C.F.L. Linf or L2 stability condition: |c| dt/dx =< 1
#
nu = 0.5
dt = nu*dx/c
# niter = 10
niter = floor((T-t0)/dt)
print('time step dt =', dt)
print('Number of time iterations niter =', niter)

# Practical space domain of calculation: [0, L] and mesh
nx = floor(L/dx)
X_mesh = np.linspace(0, L, nx+1)
print('Number of FV cells nx =', nx)
print(len(X_mesh))

# Definition of function u_0 for several initial conditions (I.C.)
# and possible upstream or downstream B.C.
u0_1 = lambda x: np.exp(-5 * (x - 0.5)**2)
u0_2 = lambda x: (1 + np.tanh(10 * (x - 0.5)))/2


#
# Comparison test of execution elapse times
# between four different implementation versions of Upwind explicit scheme
#

# Upwind explicit scheme for c > 0
# Full right-hand side (rhs) tridiagonal matrix
#
def Matrhs_Upwind(nx, nu):
    Diag = (1 - nu) * np.ones(nx+1)
    Diag_m = nu * np.ones(nx)
    Mrhs_Up = np.diag(Diag) + np.diag(Diag_m, k=-1)
    return Mrhs_Up

#
# Approximate solutions with F.D. explicit schemes
#

# choice of I.C. u0
# Exact solution : u(t,x) = u_0(x - c t)

u0 = u0_2
U0 = np.copy(u0(X_mesh))
Uex = np.copy(u0(X_mesh - c * T))


# records the starting time
start = time.perf_counter()
#
# Upwind explicit scheme for c > 0: naive version with loop 'for' in space
#
Un = np.copy(U0)      # Needs a copy to avoid changing U0 by changing Un later
U_loop = np.copy(U0)

# for n in range(0,niter):
for n in range(niter):
    for j in range(1, nx+1):
        U_loop[j] = (1 - nu) * Un[j] + nu * Un[j-1]
    U_loop[0] = u0(- c * (n+1)*dt)
#    Upstream B.C. at x=0 with exact sol. u(t_{n+1},0)
    Un = np.copy(U_loop)

# records the final time
end = time.perf_counter()
print('Execution time of naive version with loops =', (end - start)/niter, 'seconds')


start = time.perf_counter()
#
# Upwind explicit scheme for c > 0: optimized version with NO loop 'for' in space
# using arrays shifting
#
Un = np.copy(U0)

for n in range(niter):
    Un_m = np.copy(Un)
    Un_m[1:nx+1] = Un[0:nx]
    U_up = (1 - nu) * Un + nu * Un_m
    U_up[0] = u0(- c * (n+1)*dt)
    Un = np.copy(U_up)

end = time.perf_counter()
print('Execution time of no-loop optimized version =', (end - start)/niter, 'seconds')


start = time.perf_counter()
#
# Upwind explicit scheme for c > 0: standard version with rhs full tridiagonal matrix
# Possible problem of memory swap to cache for h <= 10^-04 with RAM = 8 Go
# Possible optimization using the class of Sparse matrices in 'scipy.sparse'
#
Mrhs_Up = Matrhs_Upwind(nx, nu)
Un = np.copy(U0)

# Main loop in time
for n in range(niter):
    U_upwind = np.dot(Mrhs_Up, Un)
#    U_upwind = Mrhs_Up.dot(Un)     # Matrix-vector (row by column) product
#    Upwind B.C. at x=0 with exact sol. u(t_{n+1},0)
    U_upwind[0] = u0(- c * (n+1)*dt)
    Un = U_upwind

end = time.perf_counter()
print('Execution time of standard rhs version with full matrix =', (end - start)/niter, 'seconds')


start = time.perf_counter()
#
# Upwind explicit scheme for c > 0: version with rhs Sparse tridiagonal matrix
# N.B. Unstable downwind explicit scheme for c < 0
#
# Construction of RHS Sparse tridiagonal matrix in format 'csr_matrix'
# Use 'sp.diags' or 'sp.spdiags' functions from 'scipy.sparse' module
# N.B. Compressed Sparse Row (CSR) format specially suitable for fast matrix-vector products
#
Mrhs_Upsp = sp.diags([nu, 1. - nu, 0.], [-1, 0, 1], shape=(nx+1,nx+1), format='csr')
Un = np.copy(U0)

# Main loop in time
for n in range(niter):
    U_upsp = Mrhs_Upsp.dot(Un)   # Ok for Sparse matrix-vector (row by column) product
#    Upstream B.C. at x=0 with exact sol. u(t_{n+1},0)
    U_upsp[0] = u0(- c * (n+1)*dt)
    Un = U_upsp

end = time.perf_counter()
print('Execution time of rhs version with sparse matrix =', (end - start)/niter, 'seconds')


# Errors between versions
Err1 = max(abs(U_upwind - U_loop))
Err2 = max(abs(U_upwind - U_up))
Err3 = max(abs(U_upwind - U_upsp))
print('Error between two versions Err1 =', Err1)
print('Error between two versions Err2 =', Err2)
print('Error between two versions Err3 =', Err3)


# Plot the exact and approximate solutions

plt.close(fig='all')

plt.figure(1)
# plt.plot(X_mesh, u0(X_mesh - c * T), '-', color='blue', label='Exact sol. at $t=2$')
plt.plot(X_mesh, Uex, '-', color='blue', label='Exact sol. at $t=2$')
plt.plot(X_mesh, U_loop, '-.', color='magenta', label='Upwind - space loop')
# plt.plot(X_mesh, U_upwind, '-.', color='blue', label='Upwind - full rhs matrix')
plt.plot(X_mesh, U_up, '-.', color='red', label='Upwind - Array shift')
plt.plot(X_mesh, U_upsp, ':', color='red', label='Upwind - sparse rhs matrix')

plt.xlabel('$x$')
plt.ylabel('$u(t,x)$')
plt.legend(loc = 'best')
# plt.legend(loc = 'upper center')
plt.title('Linear transport with velocity $c=1$ - $\delta x=10^{-2}$, $CFL=0.5$')
#plt.title('Linear transport with velocity $c=1$ - $\delta x=10^{-3}$, $CFL=0.5$')
# plt.savefig('transport_sol2_test.pdf', bbox_inches='tight')
plt.show()


# Error in Linf(0,L) norm at t = T

# err_Linf_Up = np.max(np.abs(Uex - U_up))
err_Linf_Up = max(abs(Uex - U_up))
print('Error in Linf(0,L) norm: at T =', T, 'for dx =', dx, 'and CFL =', nu)
print('Upwind Error in Linf norm =', err_Linf_Up)


#
# Synthesis : comparison results of CPU times (s) until T=2 for decreasing mesh steps h=dx
#

step_h = np.array([1., 1.e-01, 1.e-02, 1.e-03], float)
# print('space steps h = dx =', step_h)

cpu_loop = np.array([1.00e-04, 2.42e-03, 1.25e-01, 11.2], float)
cpu_shift = np.array([1.13e-04, 9.87e-04, 6.18e-03, 0.105], float)
cpu_full = np.array([1.40e-04, 3.35e-04, 1.31e-02, 34.8], float)
cpu_sparse = np.array([1.06e-03, 1.23e-03, 6.52e-03, 0.121], float)

# Plot the cpu graphs in loglog scale
# Order 1 in number of unknowns (Ndof) : O(1/h)
slope_1 = 1.e-03/step_h

plt.figure(2)
plt.loglog(step_h, slope_1, '-.', color='green', label='Order $1$ in nb of unknowns: $\mathcal{O}(1/h)$')

plt.loglog(step_h, cpu_loop, 'x-', color='magenta', label='Space loop')
plt.loglog(step_h, cpu_shift, 'x-', color='blue', label='Array shift')
plt.loglog(step_h, cpu_full, 'x-.', color='red', label='Full rhs matrix')
plt.loglog(step_h, cpu_sparse, 'x-', color='red', label='Sparse rhs matrix')

plt.grid(True)
plt.xlabel('Space step: $h=\delta x$')
plt.ylabel('CPU time (s)')
plt.title('CPU time for Upwind scheme - $T=2$, $CFL=0.5$')
plt.legend(loc = "best")
# plt.savefig('cpu_upwind.pdf', bbox_inches='tight')
plt.show()


#
# Synthesis : comparison results of CPU times (s) per time step for decreasing mesh steps h=dx
#

step_hs = np.array([1., 1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06], float)
# print('space steps h = dx =', step_h)

cpu_loops = np.array([4.10e-05, 1.12e-04, 6.18e-04, 2.64e-03, 2.71e-02, 2.76e-01, 2.82], float)
cpu_shifts = np.array([3.53e-05, 1.41e-04, 5.78e-04, 4.18e-05, 4.84e-04, 7.77e-03, 7.87e-02], float)
cpu_fulls = np.array([8.09e-05, 6.23e-04, 2.00e-03, 2.33e-01, 32.0], float)
cpu_sparses = np.array([6.36e-04, 9.31e-04, 1.12e-03, 1.18e-03, 7.32e-03, 4.69e-02, 5.25e-01], float)

# Plot the cpu graphs in loglog scale
# Order 1 in number of unknowns (Ndof) : O(1/h)
slope_1 = 1.e-05/step_hs

plt.figure(3)
plt.loglog(step_hs, slope_1, '-.', color='green', label='Order $1$ in nb of unknowns: $\mathcal{O}(1/h)$')

plt.loglog(step_hs, cpu_loops, 'x-', color='magenta', label='Space loop')
plt.loglog(step_hs, cpu_shifts, 'x-', color='blue', label='Array shift')
# plt.loglog(step_hs, cpu_fulls, 'x-.', color='red', label='Full rhs matrix')
plt.loglog(step_hs, cpu_sparses, 'x-', color='red', label='Sparse rhs matrix')

plt.grid(True)
plt.xlabel('Space step: $h=\delta x$')
plt.ylabel('CPU time (s)')
plt.title('CPU time per time step for Upwind scheme')
plt.legend(loc = "best")
# plt.savefig('cpu_per_time-step_upwind_h10-6.pdf', bbox_inches='tight')
plt.show()




