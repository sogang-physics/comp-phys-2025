#!/usr/bin/env python
import numpy as np
import time


def heat_equation_2d_serial(nx, ny, nt, alpha=0.01):
    """
    Solve 2D heat equation serially
    """
    # Grid spacing
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dt = dx * dy / (4 * alpha)

    # Initialize temperature field
    T = np.zeros((nx, ny))
    T_new = np.zeros((nx, ny))

    # Initial condition: hot spot in center
    center_x, center_y = nx // 2, ny // 2
    radius = min(nx, ny) // 10
    for i in range(nx):
        for j in range(ny):
            if (i - center_x)**2 + (j - center_y)**2 < radius**2:
                T[i, j] = 100.0

    # Boundary conditions (fixed at 0)
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0.0

    # Time stepping
    for n in range(nt):
        # Update interior points
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                T_new[i, j] = T[i, j] + alpha * dt / dx**2 * (
                    T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] - 4*T[i, j]
                )

        # Keep boundaries fixed
        T_new[0, :] = T_new[-1, :] = T_new[:, 0] = T_new[:, -1] = 0.0

        # Swap arrays
        T, T_new = T_new, T

    return T


# Main execution
nx = ny = 200
nt = 500

print(f"Serial 2D Heat Equation")
print(f"Grid: {nx} × {ny}")
print(f"Time steps: {nt}")

start = time.time()
T_final = heat_equation_2d_serial(nx, ny, nt)
elapsed = time.time() - start

print(f"\nComputation time: {elapsed:.3f} seconds")
print(f"Max temperature: {np.max(T_final):.2f}")
print(f"Min temperature: {np.min(T_final):.2f}")

# Save result for comparison
np.save('heat_serial_result.npy', T_final)
print(f"\nResult saved to heat_serial_result.npy")
