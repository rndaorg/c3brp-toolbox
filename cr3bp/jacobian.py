import numpy as np
from numba import njit
 

def jacobian(x, y, mu):
    """
    Compute the Jacobian matrix of the linearized CR3BP equations of motion
    evaluated at (x, y) in the rotating frame.
    
    State vector: [x, y, vx, vy]
    Equations:
        x'' - 2*vy = ∂U/∂x
        y'' + 2*vx = ∂U/∂y
    where U = (x² + y²)/2 + (1-μ)/r1 + μ/r2
    """
    # Positions of primaries
    x1 = -mu
    x2 = 1 - mu

    # Distances
    r1 = np.sqrt((x - x1)**2 + y**2)
    r2 = np.sqrt((x - x2)**2 + y**2)

    # Avoid singularity
    if r1 == 0 or r2 == 0:
        raise ValueError("Evaluation point coincides with a primary mass.")

    # Second derivatives of potential
    U_xx = 1 - (1 - mu) * ( (y**2 + (x - x1)**2 - 3*(x - x1)**2) / r1**5 ) \
             - mu * ( (y**2 + (x - x2)**2 - 3*(x - x2)**2) / r2**5 )
    U_yy = 1 - (1 - mu) * ( ( (x - x1)**2 + y**2 - 3*y**2 ) / r1**5 ) \
             - mu * ( ( (x - x2)**2 + y**2 - 3*y**2 ) / r2**5 )
    U_xy =     (1 - mu) * ( 3*(x - x1)*y / r1**5 ) \
             + mu * ( 3*(x - x2)*y / r2**5 )

    # Jacobian of the first-order system:
    # Let state = [x, y, vx, vy]
    # Then:
    # dx/dt = vx
    # dy/dt = vy
    # dvx/dt = 2*vy + U_x
    # dvy/dt = -2*vx + U_y
    #
    # Linearized: dδ/dt = A δ, where A is:
    J = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [U_xx, U_xy, 0, 2],
        [U_xy, U_yy, -2, 0]
    ])
    return J


@njit
def cr3bp_jacobian(x, y, mu):
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)

    Uxx = 1 - (1 - mu) * (1/r1**3 - 3*(x + mu)**2 / r1**5) - mu * (1/r2**3 - 3*(x - 1 + mu)**2 / r2**5)
    Uyy = 1 - (1 - mu) * (1/r1**3 - 3*y**2 / r1**5) - mu * (1/r2**3 - 3*y**2 / r2**5)
    Uxy = - (1 - mu) * ( -3*(x + mu)*y / r1**5 ) - mu * ( -3*(x - 1 + mu)*y / r2**5 )

    # Jacobian (state = [x, y, vx, vy])
    J = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [Uxx, Uxy, 0, 2],
        [Uxy, Uyy, -2, 0]
    ])
    return J

