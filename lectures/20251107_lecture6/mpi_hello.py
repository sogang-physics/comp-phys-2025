#!/usr/bin/env python
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process ID
size = comm.Get_size()  # Total number of processes

print(f"Hello from process {rank} of {size}")

# Synchronize
comm.Barrier()

if rank == 0:
    print(f"\nAll {size} processes completed!")
