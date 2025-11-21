from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Total number of samples
N_total = 100_000_000

# Each process handles a portion
N_local = N_total // size

# Start timing
start_time = time.time()

# Generate random points in unit square
np.random.seed(rank)  # Different seed for each process
x = np.random.random(N_local)
y = np.random.random(N_local)

# Count points inside quarter circle
inside = np.sum(x**2 + y**2 <= 1.0)

# Sum results from all processes
total_inside = comm.reduce(inside, op=MPI.SUM, root=0)

end_time = time.time()

if rank == 0:
    pi_estimate = 4.0 * total_inside / N_total
    print(f"{'='*60}")
    print(f"Parallel Monte Carlo Estimation of π")
    print(f"{'='*60}")
    print(f"Number of processes: {size}")
    print(f"Total samples: {N_total:,}")
    print(f"Samples per process: {N_local:,}")
    print(f"\nEstimated π: {pi_estimate:.10f}")
    print(f"Actual π:    {np.pi:.10f}")
    print(f"Error:       {abs(pi_estimate - np.pi):.10f}")
    print(f"\nTime taken: {end_time - start_time:.4f} seconds")
    print(f"Samples/sec: {N_total / (end_time - start_time):,.0f}")
