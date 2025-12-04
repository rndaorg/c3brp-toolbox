import numpy as np
import matplotlib.pyplot as plt


def jacobi_constant(state, mu):
    """Compute Jacobi constant for a given state."""
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)
    return x**2 + y**2 + 2*((1 - mu)/r1 + mu/r2) - (vx**2 + vy**2)


def omega(x, y, mu):
    x1, x2 = -mu, 1 - mu
    r1 = np.sqrt((x - x1)**2 + y**2)
    r2 = np.sqrt((x - x2)**2 + y**2)
    return 0.5*(x**2 + y**2) + (1 - mu)/r1 + mu/r2