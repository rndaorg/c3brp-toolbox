import numpy as np
from numeric.newton import newton_raphson

def collinear_lagrange_points(mu, tol=1e-12, max_iter=100):
    """
    Compute L1, L2, L3 for the circular restricted three-body problem
    using only NumPy and vanilla Python.
    
    Parameters
    ----------
    mu : float
        Mass ratio mu = m2 / (m1 + m2), where m2 <= m1, so 0 < mu <= 0.5.
    tol : float
        Desired absolute tolerance on x.
    max_iter : int
        Maximum iterations for Newton-Raphson.
    
    Returns
    -------
    L1, L2, L3 : float
        x-coordinates of the three collinear Lagrange points (y = 0).
    """
    if not (0 < mu <= 0.5):
        raise ValueError("mu must be in (0, 0.5]")
    
    # Positions of primaries in normalized synodic frame:
    # m1 at x = -mu, m2 at x = 1 - mu
    # Note: distance between primaries = 1, total mass = 1, G = 1, omega = 1
    
    def f(x):
        """Effective acceleration in rotating frame (set to zero at equilibrium)."""
        # Avoid singularities
        r1 = x + mu          # distance from m1 (at -mu)
        r2 = x - (1 - mu)    # distance from m2 (at 1 - mu)
        # Gravitational + centrifugal
        return x - (1 - mu) * r1 / np.abs(r1)**3 - mu * r2 / np.abs(r2)**3
    
    
    # --- L1: between the two primaries: -mu < x < 1 - mu
    # Initial guess: closer to smaller mass; use series approx if mu small
    if mu < 0.1:
        # Approx: x_L1 ≈ 1 - (mu/3)**(1/3)
        x_L1_guess = 1.0 - (mu / 3.0)**(1/3)
    else:
        x_L1_guess = (1 - mu - mu) / 2.0  # rough midpoint
    L1 = newton_raphson(f, x_L1_guess, x_min=-mu + 1e-6, x_max=1 - mu - 1e-6)
    
    # --- L2: beyond the smaller mass (m2 at 1 - mu), so x > 1 - mu
    if mu < 0.1:
        # Approx: x_L2 ≈ 1 + (mu/3)**(1/3)
        x_L2_guess = 1.0 + (mu / 3.0)**(1/3)
    else:
        x_L2_guess = 1.0 + 0.1
    # Use bisection with safe bracket: [1 - mu + eps, 2]
    L2 = newton_raphson(f, x_L2_guess, x_min=1 - mu + 1e-6, x_max=2.0)
    
    # --- L3: beyond the larger mass (m1 at -mu), so x < -mu
    if mu < 0.1:
        # Approx: x_L3 ≈ -1 - (5/12)*mu
        x_L3_guess = -1.0 - (5.0/12.0)*mu
    else:
        x_L3_guess = -1.5
    # Bracket: [-2, -mu - eps]
    L3 = newton_raphson(f, x_L3_guess, x_min=-2.0, x_max=-mu - 1e-6)
    
    return L1, L2, L3
