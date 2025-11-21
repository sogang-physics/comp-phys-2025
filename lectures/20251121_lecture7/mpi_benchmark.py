from mpi4py import MPI
import numpy as np
import time

def benchmark_computation(size):
    """Perform some computation to benchmark"""
    N = size
    A = np.random.randn(N, N)
    B = np.random.randn(N, N)
    C = A @ B
    return np.sum(C)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Problem sizes to test
problem_sizes = [500, 1000, 2000, 4000]

if rank == 0:
    print(f"{'='*70}")
    print(f"MPI Scaling Benchmark")
    print(f"Number of processes: {size}")
    print(f"{'='*70}")

for N in problem_sizes:
    comm.Barrier()
    start_time = time.time()
    
    # Each process does some work
    local_result = benchmark_computation(N // size)
    
    # Combine results
    global_result = comm.reduce(local_result, op=MPI.SUM, root=0)
    
    comm.Barrier()
    elapsed = time.time() - start_time
    
    if rank == 0:
        print(f"\nProblem size: {N}×{N}")
        print(f"  Time: {elapsed:.4f} seconds")
        print(f"  Time per process: {elapsed:.4f} s")
        
print(f"\nRank {rank} finished benchmark")
