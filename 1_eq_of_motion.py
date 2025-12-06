import numpy as np
import matplotlib.pyplot as plt
from cr3bp.cr3bp_eq_of_motion import cr3bp_ode
from const.const import mu_earth_moon ,mu_jupiter_europa
from numeric.rk4 import rk4_step

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
plt.savefig('cr3bp_state.png')
