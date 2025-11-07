#!/usr/bin/env python
from mpi4py import MPI
import numpy as np


def initialize_temperature(nx, ny):
    """Initialize temperature field with hot spot in center"""
    T = np.zeros((nx, ny))
    center_x, center_y = nx // 2, ny // 2
    radius = min(nx, ny) // 10
    for i in range(nx):
        for j in range(ny):
            if (i - center_x)**2 + (j - center_y)**2 < radius**2:
                T[i, j] = 100.0
    return T


def solve_heat_equation_mpi(nx, ny, nt, alpha=0.01):
    """Solve 2D heat equation using MPI with domain decomposition"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Grid parameters
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dt = dx * dy / (4 * alpha)
    factor = alpha * dt / dx**2

    # Domain decomposition: split interior points among processes
    interior_rows = nx - 2
    rows_per_process = interior_rows // size
    remainder = interior_rows % size

    # Determine local domain (interior rows only)
    if rank < remainder:
        local_interior_rows = rows_per_process + 1
        start_row = rank * local_interior_rows + 1
    else:
        local_interior_rows = rows_per_process
        start_row = rank * rows_per_process + remainder + 1

    end_row = start_row + local_interior_rows

    # Local array size includes ghost cells
    local_nx = local_interior_rows + 2  # +2 for ghost cells (top and bottom)

    # Local array size includes ghost cells
    local_nx = local_interior_rows + 2  # +2 for ghost cells (top and bottom)

    # Initialize local temperature array
    T_local = np.zeros((local_nx, ny))
    T_new = np.zeros((local_nx, ny))

    # Root initializes and distributes
    if rank == 0:
        T_full = initialize_temperature(nx, ny)
        # Set boundary conditions
        T_full[0, :] = T_full[-1, :] = T_full[:, 0] = T_full[:, -1] = 0.0

        # Extract local portion for rank 0 (include one ghost cell below)
        T_local[0, :] = T_full[start_row - 1, :]  # Top ghost (boundary)
        T_local[1:-1, :] = T_full[start_row:end_row, :]  # Interior
        if size > 1:
            T_local[-1, :] = T_full[end_row, :]  # Bottom ghost (from neighbor)
        else:
            T_local[-1, :] = T_full[end_row, :]  # Bottom ghost (boundary)

        # Send to other processes
        for r in range(1, size):
            r_interior = rows_per_process + (1 if r < remainder else 0)
            r_start = r * rows_per_process + min(r, remainder) + 1
            r_end = r_start + r_interior

            # Send: top ghost, interior rows, bottom ghost
            temp = np.zeros((r_interior + 2, ny))
            temp[0, :] = T_full[r_start - 1, :]  # Top ghost
            temp[1:-1, :] = T_full[r_start:r_end, :]  # Interior
            if r < size - 1:
                temp[-1, :] = T_full[r_end, :]  # Bottom ghost
            else:
                temp[-1, :] = T_full[r_end, :]  # Bottom ghost (boundary)

            comm.send(temp, dest=r, tag=0)
    else:
        T_local = comm.recv(source=0, tag=0)

    # Time stepping
    for n in range(nt):
        # Exchange ghost cells with neighbors using non-blocking communication
        requests = []

        # Exchange with upper neighbor
        if rank > 0:
            # Send first interior row up, receive ghost row from above
            req1 = comm.isend(T_local[1, :].copy(), dest=rank-1, tag=1)
            req2 = comm.irecv(source=rank-1, tag=2)
            requests.extend([req1, req2])

        # Exchange with lower neighbor
        if rank < size - 1:
            # Send last interior row down, receive ghost row from below
            req3 = comm.isend(T_local[-2, :].copy(), dest=rank+1, tag=2)
            req4 = comm.irecv(source=rank+1, tag=1)
            requests.extend([req3, req4])

        # Wait for all communications to complete
        if rank > 0:
            T_local[0, :] = requests[1].wait()
        if rank < size - 1:
            T_local[-1, :] = requests[3 if rank > 0 else 1].wait()

        # Wait for sends to complete
        for i, req in enumerate(requests):
            if i % 2 == 0:  # Send requests
                req.wait()

        # Update interior points
        for i in range(1, local_nx - 1):
            for j in range(1, ny - 1):
                T_new[i, j] = T_local[i, j] + factor * (
                    T_local[i+1, j] + T_local[i-1, j] +
                    T_local[i, j+1] + T_local[i, j-1] - 4*T_local[i, j]
                )

        # Keep side boundaries at zero
        T_new[:, 0] = T_new[:, -1] = 0.0

        # Swap arrays
        T_local, T_new = T_new, T_local

    # Gather results to root
    if rank == 0:
        T_result = np.zeros((nx, ny))
        # Set boundaries
        T_result[0, :] = T_result[-1, :] = T_result[:,
                                                    0] = T_result[:, -1] = 0.0

        # Copy rank 0's interior rows
        T_result[start_row:end_row, :] = T_local[1:-1, :]

        # Receive from other processes
        for r in range(1, size):
            r_interior = rows_per_process + (1 if r < remainder else 0)
            r_start = r * rows_per_process + min(r, remainder) + 1
            r_end = r_start + r_interior

            data = comm.recv(source=r, tag=3)
            T_result[r_start:r_end, :] = data[1:-1, :]  # Extract interior only

        return T_result
    else:
        comm.send(T_local, dest=0, tag=3)
        return None


# Main execution
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
nx = ny = 200
nt = 500

if rank == 0:
    print(f"MPI 2D Heat Equation")
    print(f"Grid: {nx} × {ny}")
    print(f"Time steps: {nt}")
    print(f"Processes: {size}")

comm.Barrier()
start_time = MPI.Wtime()

T_final = solve_heat_equation_mpi(nx, ny, nt)

comm.Barrier()
elapsed = MPI.Wtime() - start_time

if rank == 0:
    print(f"\nComputation time: {elapsed:.3f} seconds")
    print(f"Max temperature: {np.max(T_final):.2f}")
    print(f"Min temperature: {np.min(T_final):.2f}")
