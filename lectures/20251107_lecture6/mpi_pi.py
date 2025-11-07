#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import time

def monte_carlo_pi(n_samples):
    """Estimate pi using Monte Carlo"""
    np.random.seed(int(time.time() * 1000) % 2**31 + MPI.COMM_WORLD.Get_rank())
    x = np.random.random(n_samples)
    y = np.random.random(n_samples)
    inside = np.sum((x**2 + y**2) <= 1.0)
    return inside

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Total samples
n_total = 100_000_000

# Divide work among processes
n_local = n_total // size

# Start timing
comm.Barrier()
start_time = MPI.Wtime()

# Each process computes its portion
local_inside = monte_carlo_pi(n_local)

# Reduce: sum all local counts
total_inside = comm.reduce(local_inside, op=MPI.SUM, root=0)

# Stop timing
comm.Barrier()
elapsed_time = MPI.Wtime() - start_time

# Root process computes and prints result
if rank == 0:
    pi_estimate = 4.0 * total_inside / n_total
    print(f"Number of processes: {size}")
    print(f"Total samples: {n_total:,}")
    print(f"Samples per process: {n_local:,}")
    print(f"\nEstimated π: {pi_estimate:.8f}")
    print(f"Actual π:    {np.pi:.8f}")
    print(f"Error:       {abs(pi_estimate - np.pi):.8f}")
    print(f"\nTime: {elapsed_time:.4f} seconds")
    print(f"Samples/sec: {n_total/elapsed_time:,.0f}")
