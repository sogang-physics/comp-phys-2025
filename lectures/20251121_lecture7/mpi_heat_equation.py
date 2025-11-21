from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Physical parameters
L = 1.0          # Length of domain
alpha = 0.01     # Thermal diffusivity
T_final = 0.5    # Final time

# Numerical parameters
N_global = 1004  # Total number of grid points (divisible by 4)
dt = 0.00001     # Time step (reduced for stability)
N_steps = int(T_final / dt)

# Domain decomposition
N_local = N_global // size
dx = L / (N_global - 1)

# CFL condition check
cfl = alpha * dt / dx**2
if rank == 0:
    print(f"CFL number: {cfl:.4f} (should be < 0.5 for stability)")
    if cfl >= 0.5:
        print("WARNING: Unstable parameters!")

# Local grid (with ghost cells for boundaries)
u = np.zeros(N_local + 2)
u_new = np.zeros(N_local + 2)

# Initialize temperature distribution
x_local = np.linspace(rank * N_local * dx, (rank + 1) * N_local * dx, N_local + 2)
u[:] = np.sin(np.pi * x_local / L)  # Initial condition: sin wave

# Determine neighbors
left_neighbor = rank - 1 if rank > 0 else MPI.PROC_NULL
right_neighbor = rank + 1 if rank < size - 1 else MPI.PROC_NULL

# Time evolution
for step in range(N_steps):
    # Exchange boundary values with neighbors
    # Send right boundary to right neighbor, receive left boundary from left neighbor
    comm.Sendrecv(u[-2:-1], dest=right_neighbor, sendtag=0,
                  recvbuf=u[0:1], source=left_neighbor, recvtag=0)
    
    # Send left boundary to left neighbor, receive right boundary from right neighbor
    comm.Sendrecv(u[1:2], dest=left_neighbor, sendtag=1,
                  recvbuf=u[-1:], source=right_neighbor, recvtag=1)
    
    # Update interior points using finite difference
    u_new[1:-1] = u[1:-1] + cfl * (u[2:] - 2*u[1:-1] + u[:-2])
    
    # Boundary conditions (fixed at 0)
    if rank == 0:
        u_new[1] = 0.0
    if rank == size - 1:
        u_new[-2] = 0.0
    
    # Swap arrays
    u, u_new = u_new, u

# Gather results for plotting
u_global = None
if rank == 0:
    u_global = np.zeros(N_global)

# Remove ghost cells before gathering
u_local = u[1:-1]
comm.Gather(u_local, u_global, root=0)

if rank == 0:
    # Plot results
    x = np.linspace(0, L, N_global)
    plt.figure(figsize=(10, 6))
    plt.plot(x, np.sin(np.pi * x / L), 'b--', label='Initial', linewidth=2)
    plt.plot(x, u_global, 'r-', label=f't = {T_final}', linewidth=2)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Temperature', fontsize=12)
    plt.title(f'1D Heat Equation (Parallel with {size} processes)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('heat_equation_parallel.png', dpi=150)
    print(f"\nSimulation complete! Plot saved as 'heat_equation_parallel.png'")
    print(f"Steps computed: {N_steps:,}")
    print(f"Grid points per process: {N_local}")
