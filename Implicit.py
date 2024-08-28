
import jax.numpy as jnp
import jax
from jax.scipy.linalg import solve
import matplotlib.pyplot as plt

# Domain setup
x_min, x_max = -1.0, 1.0  # Domain 
N = 100  # Number of grid points
dx = (x_max - x_min) / (N - 1) 
x = jnp.linspace(x_min, x_max, N) # array for domaim of graph

# Narrow Gaussian centered at x = 0 as initial condition
P_initial = jnp.exp(-50 * (x ** 2))

# Formula to nomralize the distribution
area = jnp.sum(P_initial) * dx
P_initial /= area

# Parameters
dt = 0.0001  # dt has to be small 
T = 0.1  # time increments
num_steps = int(T / dt) 
D = 0.1 # diffusion constant

# First derivative matrix using Central-Diff 
D1 = jnp.diag(-jnp.ones(N-1), -1) + jnp.diag(jnp.ones(N-1), 1) # Using a built in funciton of jax/numpy
D1 = D1.at[0, 0].set(0).at[-1, -1].set(0) # setting the no flux boundary conditions
D1 /= (2 * dx)

# Second derivative matrix using Central-Diffs
D2 = jnp.diag(-2 * jnp.ones(N), 0) + jnp.diag(jnp.ones(N-1), 1) + jnp.diag(jnp.ones(N-1), -1)
D2 = D2.at[0, 0].set(0).at[-1, -1].set(0)  # No-flux boundary conditions
D2 /= dx**2

# Initializeing the Matrices
I = jnp.eye(N) # identity matrix of size N * N 
A = I - dt * (D * D2 + jnp.diag(x) @ D1 + I) # Implicit Method formula

# Initializing probability distribution
P = P_initial.copy()

# Storing results for graphing
results = [P_initial.copy()]

# Time-stepping loop using the implicit method
for _ in range(num_steps):
    P = solve(A, P) # solves linear system A * P_new = P_old
    P /= jnp.sum(P) * dx # normalizing distrubition to avoid numerical drift
    
    if _ % (num_steps // 10) == 0: # storing results as intervals for graph
        results.append(P.copy())

# Plotting evolution of P(x, t) over time
plt.figure(figsize=(8, 6))
for i, P in enumerate(results):
    plt.plot(x, P, label=f't={i * T / len(results):.2f}')

plt.xlabel('x')
plt.ylabel('P(x, t)')
plt.title('Evolution of P(x, t) over time (Implicit Method)')
plt.legend()
plt.show()




