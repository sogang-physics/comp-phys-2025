from mpi4py import MPI
import socket

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

# Each process prints its information
print(f"Hello from rank {rank} of {size} processes on host {hostname}")

# Synchronize all processes
comm.Barrier()

# Only rank 0 prints the summary
if rank == 0:
    print(f"\n{'='*50}")
    print(f"MPI program completed with {size} processes")
    print(f"{'='*50}")
