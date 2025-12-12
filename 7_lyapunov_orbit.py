from matplotlib import pyplot as plt
import numpy as np

from const.const import get_mu
from cr3bp.collinear import compute_L1
from cr3bp.jacobian import cr3bp_jacobian
from numeric.differential_correction import differential_correction_Lyapunov

from numba import njit, config

config.DISABLE_JIT = True

if __name__ == "__main__":
    mu = get_mu("earth_moon")
    L1 = compute_L1(mu)
    print(f"L1 location: x = {L1:.10f}")

    # Linearized guess: offset from L1 along unstable direction
    J = cr3bp_jacobian(L1, 0.0, mu)
    
    # Eigenvalues give unstable direction — we approximate with small x offset
    x0_guess = L1 - 0.005  # Slightly left of L1
    vy0_guess = 0.05       # Initial guess for vy

    state0, T, traj = differential_correction_Lyapunov(mu, x0_guess, vy0_guess)

    print(f"Orbit period: {T:.6f}")
    print(f"Final state (half): {traj[-1]}")

    # Optionally plot using matplotlib (not required for computation)
    # import matplotlib.pyplot as plt
    plt.plot(traj[:,0], traj[:,1])
    plt.scatter([L1], [0], c='red')
    plt.axis('equal')
    # plt.show()
    plt.savefig("earth_moon_lyapunov_orbit.png")

