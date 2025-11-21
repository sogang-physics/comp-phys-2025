from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"\n{'='*60}")
print(f"Process {rank} starting collective communication examples")
print(f"{'='*60}")

# Example 1: Broadcast
if rank == 0:
    data = {'param': 'simulation_config', 'timesteps': 1000, 'dt': 0.01}
    print(f"\nRank 0 broadcasting: {data}")
else:
    data = None

data = comm.bcast(data, root=0)
print(f"Rank {rank} received broadcast: {data}")

comm.Barrier()

# Example 2: Scatter
if rank == 0:
    # Split work among processes
    work = np.arange(size * 4).reshape(size, 4)
    print(f"\nRank 0 scattering work:\n{work}")
else:
    work = None

local_work = comm.scatter(work, root=0)
print(f"Rank {rank} received: {local_work}")

comm.Barrier()

# Example 3: Gather
# Each process computes something
local_result = (rank + 1) ** 2
print(f"\nRank {rank} computed: {local_result}")

all_results = comm.gather(local_result, root=0)
if rank == 0:
    print(f"Rank 0 gathered all results: {all_results}")

comm.Barrier()

# Example 4: Reduce (sum)
local_value = np.array([rank + 1], dtype=np.float64)
total = np.zeros(1, dtype=np.float64)

comm.Reduce(local_value, total, op=MPI.SUM, root=0)

if rank == 0:
    print(f"\nSum of all ranks (1+2+...+{size}): {total[0]}")
    print(f"Expected: {size * (size + 1) // 2}")

# Example 5: Allreduce (result available on all processes)
local_array = np.ones(3) * (rank + 1)
global_sum = np.zeros(3)

comm.Allreduce(local_array, global_sum, op=MPI.SUM)
print(f"Rank {rank} sees global sum: {global_sum}")
