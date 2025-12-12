from numba import njit

def rk4_step(func, state, dt, mu):
    """
    Perform one RK4 integration step.
    
    Parameters:
    - func: ODE function (e.g., cr3bp_ode)
    - state: current state vector
    - dt: time step
    - mu: mass parameter
    
    Returns:
    - new_state: state after one step
    """
    k1 = func(state, mu)
    k2 = func(state + 0.5*dt*k1, mu)
    k3 = func(state + 0.5*dt*k2, mu)
    k4 = func(state + dt*k3, mu)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

#@njit
def rk4_step_xyz(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt*k1/2)
    k3 = func(t + dt/2, y + dt*k2/2)
    k4 = func(t + dt, y + dt*k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
