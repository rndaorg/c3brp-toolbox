# Let's implement the CR3BP ODE function and a basic RK4 stepper in pure Python/NumPy
import numpy as np
import matplotlib.pyplot as plt
from const.const import mu_earth_moon ,mu_jupiter_europa


def cr3bp_ode(state, mu):
    """
    Compute the time derivatives for the planar CR3BP in rotating frame.
    
    Parameters:
    - state: array-like [x, y, vx, vy]
    - mu: mass ratio (m2 / (m1 + m2))
    
    Returns:
    - dsdt: array [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = state
    
    # Distance to primary (m1 at -mu, 0)
    r1 = np.sqrt((x + mu)**2 + y**2)
    # Distance to secondary (m2 at 1 - mu, 0)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)
    
    # Avoid singularity (optional safety)
    # if r1 < 1e-6 or r2 < 1e-6:
    #     raise ValueError("Collision with primary or secondary!")
    # if r1 < 1e-12 or r2 < 1e-12:
    #     return np.array([vx, vy, 0, 0])
    
    # Equations of motion
    dxdt = vx
    dydt = vy
    dvxdt = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3
    dvydt = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    
    return np.array([dxdt, dydt, dvxdt, dvydt])