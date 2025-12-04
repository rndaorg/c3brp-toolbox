import numpy as np
from c3brp.c3brp_eq_of_motion import cr3bp_ode

def integrate_cr3bp(z0, mu, t_max, n_steps):
    dt = t_max / n_steps
    traj = np.empty((n_steps + 1, 5))
    traj[0, :4] = z0
    traj[0, 4] = 0.0 
    z = np.array(z0, dtype=float)
    for i in range(n_steps):
        k1 = cr3bp_ode(z, mu)
        k2 = cr3bp_ode(z + 0.5*dt*k1, mu)
        k3 = cr3bp_ode(z + 0.5*dt*k2, mu)
        k4 = cr3bp_ode(z + dt*k3, mu)
        z = z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        traj[i+1, :4] = z
        traj[i+1, 4] = traj[i, 4] + dt
        if np.abs(z[0]) > 5 or np.abs(z[1]) > 5:
            traj = traj[:i+2]
            break
    return traj