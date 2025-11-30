# Let's implement the CR3BP ODE function and a basic RK4 stepper in pure Python/NumPy
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Equations of motion
    dxdt = vx
    dydt = vy
    dvxdt = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3
    dvydt = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    
    return np.array([dxdt, dydt, dvxdt, dvydt])

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

# Test with Earth-Moon system
mu_earth_moon = 0.012150585609624044

# Initial condition near L2 (rough guess)
x0 = 1.1  # beyond Moon
y0 = 0.0
vx0 = 0.0
vy0 = 0.2

state0 = np.array([x0, y0, vx0, vy0])

# Integrate for a few steps
dt = 0.01
steps = 10000
state_iter = []
state = state0.copy()

print("Step | x        | y        | vx       | vy")
print("-" * 45)
for i in range(steps + 1):
    print(f"{i:4d} | {state[0]:8.5f} | {state[1]:8.5f} | {state[2]:8.5f} | {state[3]:8.5f}")
    state_iter.append(state)
    state = rk4_step(cr3bp_ode, state, dt, mu_earth_moon)

states = np.array(state_iter)

plt.plot(states[:, 0], states[:, 1])
plt.savefig('c3brp_state')