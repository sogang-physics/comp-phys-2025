from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    print("This example requires at least 2 processes")
    exit()

N = 1_000_000
data_to_send = np.arange(N, dtype=np.float64) * (rank + 1)
data_to_recv = np.empty(N, dtype=np.float64)

# Non-blocking send/receive
if rank == 0:
    # Send to rank 1, receive from rank 1
    req_send = comm.Isend(data_to_send, dest=1, tag=0)
    req_recv = comm.Irecv(data_to_recv, source=1, tag=1)
    
    # Do some computation while communication happens
    start = time.time()
    local_result = np.sum(data_to_send ** 2)
    computation_time = time.time() - start
    
    # Wait for communication to complete
    req_send.wait()
    req_recv.wait()
    
    print(f"Rank 0: Communication overlapped with computation!")
    print(f"Computation took: {computation_time:.6f} seconds")
    print(f"Received data sum: {np.sum(data_to_recv):.2e}")
    
elif rank == 1:
    # Receive from rank 0, send to rank 0
    req_recv = comm.Irecv(data_to_recv, source=0, tag=0)
    req_send = comm.Isend(data_to_send, dest=0, tag=1)
    
    # Do some computation while communication happens
    local_result = np.sum(data_to_send ** 2)
    
    # Wait for communication to complete
    req_send.wait()
    req_recv.wait()
    
    print(f"Rank 1: Received data sum: {np.sum(data_to_recv):.2e}")
