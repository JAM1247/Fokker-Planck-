import jax.numpy as jnp
import jax
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

# Construct the second derivative matrix (central difference)
D2 = jnp.diag(-2 * jnp.ones(N), 0) + jnp.diag(jnp.ones(N-1), 1) + jnp.diag(jnp.ones(N-1), -1)
D2 = D2.at[0, 0].set(0).at[-1, -1].set(0)  # No-flux boundary conditions
D2 /= dx**2

# Initializing probability distribution
P = P_initial.copy()

# Forward Euler function 
@jax.jit
def step(P, D, D1, D2, x, dt):
    flux = D * (D2 @ P) + x * (D1 @ P) # Calculate the flux terms using matrix multiplication
    P_new = P + dt * (-x * (D1 @ P) + flux) # Updating probability distribution
    return P_new

# Stoeing results for graphing 
results = [P_initial.copy()] # first element intzliaed as guassian 

# Time-stepping loop
for _ in range(num_steps):
    P = step(P, D, D1, D2, x, dt) # updating probability distribution
    P = P.at[0].set(0).at[-1].set(0) # reapplying boudnary conditions
    if _ % (num_steps // 10) == 0: # storing results as intervals for graph
        results.append(P.copy())

# Plotting evolution of P(x, t) over time
plt.figure(figsize=(8, 6))
for i, P in enumerate(results):
    plt.plot(x, P, label=f't={i * T / len(results):.2f}')

plt.xlabel('x')
plt.ylabel('P(x, t)')
plt.title('Evolution of P(x, t) Over Time')
plt.legend()
plt.show()
