import numpy as np
import matplotlib.pyplot as plt

from c3brp.c3brp_eq_of_motion import mu_earth_moon, cr3bp_ode, rk4_step


def jacobi_constant(state, mu):
    """Compute Jacobi constant for a given state."""
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)
    return x**2 + y**2 + 2*((1 - mu)/r1 + mu/r2) - (vx**2 + vy**2)

