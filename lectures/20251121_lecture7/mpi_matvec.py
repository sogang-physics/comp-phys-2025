from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix dimensions
N = 1000  # Size of square matrix (reduced from 10000 for faster execution)

# Number of rows per process
rows_per_process = N // size

# Create matrix and vector on root
if rank == 0:
    A = np.random.randn(N, N)
    x = np.random.randn(N)
    y_serial = np.zeros(N)
    
    # Serial computation for comparison
    start_serial = time.time()
    y_serial = A @ x
    end_serial = time.time()
    serial_time = end_serial - start_serial
    
    print(f"{'='*60}")
    print(f"Parallel Matrix-Vector Multiplication")
    print(f"{'='*60}")
    print(f"Matrix size: {N} × {N}")
    print(f"Number of processes: {size}")
    print(f"Rows per process: {rows_per_process}")
    print(f"\nSerial time: {serial_time:.4f} seconds")
else:
    A = None
    x = None
    y_serial = None
    serial_time = 0.0

# Allocate local storage
# (No longer needed - removed these lines)

# Start parallel timing
comm.Barrier()
start_parallel = time.time()

# Scatter rows of A to all processes
if rank == 0:
    # Split A into chunks for each process
    A_chunks = [A[i*rows_per_process:(i+1)*rows_per_process] for i in range(size)]
else:
    A_chunks = None

local_A = comm.scatter(A_chunks, root=0)

# Broadcast x to all processes
x_local = comm.bcast(x if rank == 0 else None, root=0)

# Each process computes its portion of y
local_y = local_A @ x_local

# Gather results
y_parallel = comm.gather(local_y, root=0)
if rank == 0:
    y_parallel = np.concatenate(y_parallel)

comm.Barrier()
end_parallel = time.time()

if rank == 0:
    parallel_time = end_parallel - start_parallel
    
    print(f"\nParallel time: {parallel_time:.4f} seconds")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")
    print(f"Efficiency: {100 * serial_time / (parallel_time * size):.1f}%")
    
    # Verify correctness
    error = np.linalg.norm(y_serial - y_parallel) / np.linalg.norm(y_serial)
    print(f"\nRelative error: {error:.2e}")
    if error < 1e-10:
        print("✓ Results match!")
    else:
        print("✗ Results differ!")
