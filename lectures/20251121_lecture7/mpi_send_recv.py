from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    print("This example requires at least 2 processes")
    exit()

# Example 1: Send/receive Python objects (lowercase)
if rank == 0:
    data = {'message': 'Hello from rank 0', 'value': 42, 'array': [1, 2, 3]}
    comm.send(data, dest=1, tag=11)
    print(f"Rank 0 sent: {data}")
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    print(f"Rank 1 received: {data}")

comm.Barrier()

# Example 2: Send/receive NumPy arrays (uppercase - FASTER!)
if rank == 0:
    array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    comm.Send(array, dest=1, tag=22)
    print(f"\nRank 0 sent NumPy array: {array}")
elif rank == 1:
    array = np.empty(5, dtype=np.float64)
    comm.Recv(array, source=0, tag=22)
    print(f"Rank 1 received NumPy array: {array}")
