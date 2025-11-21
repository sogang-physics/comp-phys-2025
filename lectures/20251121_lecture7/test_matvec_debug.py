from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Rank {rank} starting", flush=True)

# Matrix dimensions
N = 100  # Small size for testing
rows_per_process = N // size

# Create matrix and vector on root
if rank == 0:
    A = np.random.randn(N, N)
    x = np.random.randn(N)
    print(f"Rank 0: Created A({A.shape}) and x({x.shape})", flush=True)
else:
    A = None
    x = None

print(f"Rank {rank}: Before scatter", flush=True)

# Scatter rows of A to all processes
if rank == 0:
    A_chunks = [A[i*rows_per_process:(i+1)*rows_per_process] for i in range(size)]
    print(f"Rank 0: Created {len(A_chunks)} chunks, each shape {A_chunks[0].shape}", flush=True)
else:
    A_chunks = None

local_A = comm.scatter(A_chunks, root=0)
print(f"Rank {rank}: After scatter, local_A shape: {local_A.shape}", flush=True)

# Broadcast x
x_local = comm.bcast(x if rank == 0 else None, root=0)
print(f"Rank {rank}: After bcast, x_local shape: {x_local.shape}", flush=True)

# Compute
local_y = local_A @ x_local
print(f"Rank {rank}: Computed local_y shape: {local_y.shape}", flush=True)

# Gather
y_parallel = comm.gather(local_y, root=0)
print(f"Rank {rank}: After gather", flush=True)

if rank == 0:
    y_parallel = np.concatenate(y_parallel)
    print(f"Rank 0: Final y_parallel shape: {y_parallel.shape}", flush=True)
    print("SUCCESS!")
