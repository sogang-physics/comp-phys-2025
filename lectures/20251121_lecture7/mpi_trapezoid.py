from mpi4py import MPI
import numpy as np
import time

def f(x):
    """Function to integrate: sqrt(1-x^2)"""
    return np.sqrt(1 - x**2)

def trapezoidal_rule(a, b, n, func):
    """Compute integral using trapezoidal rule"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Integration parameters
a = 0.0  # Lower bound
b = 1.0  # Upper bound
n_total = 10_000_000  # Total number of trapezoids

# Divide work among processes
n_local = n_total // size

# Each process handles a portion of the domain
local_a = a + rank * (b - a) / size
local_b = a + (rank + 1) * (b - a) / size

# Start timing
start_time = time.time()

# Compute local integral
local_integral = trapezoidal_rule(local_a, local_b, n_local, f)

# Sum all local integrals
total_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)

end_time = time.time()

if rank == 0:
    # The integral equals π/4, so multiply by 4
    pi_estimate = 4.0 * total_integral
    print(f"{'='*60}")
    print(f"Parallel Trapezoidal Integration")
    print(f"{'='*60}")
    print(f"Number of processes: {size}")
    print(f"Total trapezoids: {n_total:,}")
    print(f"Trapezoids per process: {n_local:,}")
    print(f"\nIntegral value: {total_integral:.10f}")
    print(f"Estimated π: {pi_estimate:.10f}")
    print(f"Actual π:    {np.pi:.10f}")
    print(f"Error:       {abs(pi_estimate - np.pi):.10f}")
    print(f"\nTime taken: {end_time - start_time:.4f} seconds")
