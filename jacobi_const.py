import numpy as np
import matplotlib.pyplot as plt

from c3brp_eq_of_motions import mu_earth_moon, cr3bp_ode, rk4_step


def jacobi_constant(state, mu):
    """Compute Jacobi constant for a given state."""
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)
    return x**2 + y**2 + 2*((1 - mu)/r1 + mu/r2) - (vx**2 + vy**2)


# System parameters (Earth–Moon)
mu = mu_earth_moon #0.01215  # Earth–Moon mass ratio

# Initial condition (near Moon, low-energy)
state0 = np.array([1.1, 0.0, 0.0, 0.2])  # [x, y, vx, vy]

# Integration settings
dt = 0.01
t_final = 6.2832  # ~2π (1 synodic period)
n_steps = int(t_final / dt)

# Storage
trajectory = np.zeros((n_steps + 1, 4))
trajectory[0] = state0

# Integrate
state = state0.copy()
for i in range(n_steps):
    state = rk4_step(cr3bp_ode, state, dt, mu)
    trajectory[i + 1] = state

# Compute Jacobi constant over time
C_vals = np.array([jacobi_constant(s, mu) for s in trajectory])
C0 = C_vals[0]
C_error = np.abs(C_vals - C0)
print(f"Initial Jacobi constant: {C0:.8f}")
print(f"Max deviation: {C_error.max():.2e}")

# Extract trajectory
x_traj = trajectory[:, 0]
y_traj = trajectory[:, 1]

# Plot trajectory and zero-velocity curve
x_grid = np.linspace(-2.0, 2.0, 500)
y_grid = np.linspace(-2.0, 2.0, 500)
X, Y = np.meshgrid(x_grid, y_grid)

R1 = np.sqrt((X + mu)**2 + Y**2)
R2 = np.sqrt((X - (1 - mu))**2 + Y**2)
# Avoid division by zero
R1 = np.where(R1 == 0, 1e-12, R1)
R2 = np.where(R2 == 0, 1e-12, R2)

V = X**2 + Y**2 + 2*((1 - mu)/R1 + mu/R2)

plt.figure(figsize=(8, 8))
# Zero-velocity curve (contour where V = C)
contour = plt.contour(X, Y, V, levels=[C0], colors='red', linestyles='--', linewidths=1.2)
plt.clabel(contour, fmt={C0: f"C = {C0:.4f}"}, fontsize=9)

# Trajectory
plt.plot(x_traj, y_traj, 'b-', linewidth=1.0, label='Trajectory')

# Primaries
plt.plot(-mu, 0, 'bo', markersize=12, label='Earth (Primary)')
plt.plot(1 - mu, 0, 'ko', markersize=8, label='Moon (Secondary)')

plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('CR3BP: Trajectory + Zero-Velocity Curve (Day 3)')
plt.xlabel('x (nondim)')
plt.ylabel('y (nondim)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('jacobi_constant.png')