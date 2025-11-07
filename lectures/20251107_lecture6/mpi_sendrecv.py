#!/usr/bin/env python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    print("This example requires at least 2 processes")
    exit(1)

if rank == 0:
    # Process 0 sends data to process 1
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Process {rank}: Sending {data} to process 1")
    comm.send(data, dest=1, tag=11)
    
    # Receive result back
    result = comm.recv(source=1, tag=22)
    print(f"Process {rank}: Received result {result} from process 1")
    
elif rank == 1:
    # Process 1 receives data from process 0
    data = comm.recv(source=0, tag=11)
    print(f"Process {rank}: Received {data} from process 0")
    
    # Process and send back
    result = np.sum(data)
    print(f"Process {rank}: Computed sum = {result}, sending back to process 0")
    comm.send(result, dest=0, tag=22)

print(f"Process {rank}: Done!")
