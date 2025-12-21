import numpy as np

from cr3bp.collinear import compute_L1
from numba import njit

from numeric.integrate import integrate_cr3bp_xyz


#@njit
def differential_correction_Lyapunov_old(mu, x0_guess, vy0_guess, dt=1e-3, max_iter=30, tol=1e-12):
    x0 = x0_guess
    vy0 = vy0_guess
    L1 = compute_L1(mu)
    half_period_guess = 2.0  # Non-dimensional time

    for it in range(max_iter):
        # Initial state (Lyapunov orbits are symmetric about x-axis)
        state0 = np.array([x0, 0.0, 0.0, vy0, 0.0, 0.0])  # [x, y, z, vx, vy, vz]
        t_half = half_period_guess
        traj = integrate_cr3bp_xyz(state0, 0.0, t_half, dt)
        xf, yf, zf, vxf, vyf, vzf = traj[-1]

        # Residuals: we want y = 0 and vx = 0 at half-period (symmetry)
        res = np.array([yf, vxf])

        if np.linalg.norm(res) < tol:
            print(f"Converged in {it} iterations.")
            return state0, t_half * 2, traj

        # Numerical state transition (finite differences)
        eps = 1e-8
        # Perturb x0
        state0_dx = state0.copy()
        state0_dx[0] += eps
        traj_dx = integrate_cr3bp_xyz(state0_dx, 0.0, t_half, dt)
        yf_dx, vxf_dx = traj_dx[-1][1], traj_dx[-1][3]
        # Perturb vy0
        state0_dvy = state0.copy()
        state0_dvy[4] += eps
        traj_dvy = integrate_cr3bp_xyz(state0_dvy, 0.0, t_half, dt)
        yf_dvy, vxf_dvy = traj_dvy[-1][1], traj_dvy[-1][3]

        # Build 2×2 Jacobian: d(res)/d(var)
        J = np.empty((2, 2))
        J[0, 0] = (yf_dx - yf) / eps
        J[0, 1] = (yf_dvy - yf) / eps
        J[1, 0] = (vxf_dx - vxf) / eps
        J[1, 1] = (vxf_dvy - vxf) / eps

        try:
            delta = np.linalg.solve(J, -res)
        except:
            print("Jacobian singular.")
            break

        x0 += delta[0]
        vy0 += delta[1]

        # Optional: update half-period using y-crossing (not implemented here for simplicity)

    print("Did not converge.")
    return state0, t_half * 2, traj



#@njit
def differential_correction_Lyapunov(mu, x0_guess, vy0_guess, dt=1e-4, max_iter=20, tol=1e-10):
    x0 = x0_guess
    vy0 = vy0_guess
    L1 = compute_L1(mu)
    t_half = 1.5  # Better initial half-period guess

    for it in range(max_iter):
        state0 = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        traj = integrate_cr3bp_xyz(state0, 0.0, t_half, dt)
        xf, yf, zf, vxf, vyf, vzf = traj[-1]

        # Residuals: want y=0 and vx=0 at half-period (symmetry)
        res = np.array([yf, vxf])
        err = np.linalg.norm(res)
        if err < tol:
            print(f"Converged in {it} iterations. Residual: {err:.2e}")
            return state0, 2*t_half, traj

        # Finite differences
        eps_x = 1e-8
        eps_vy = 1e-8

        # Perturb x0
        state_dx = state0.copy()
        state_dx[0] += eps_x
        traj_dx = integrate_cr3bp_xyz(state_dx, 0.0, t_half, dt)
        yf_dx, vxf_dx = traj_dx[-1][1], traj_dx[-1][3]

        # Perturb vy0
        state_dvy = state0.copy()
        state_dvy[4] += eps_vy
        traj_dvy = integrate_cr3bp_xyz(state_dvy, 0.0, t_half, dt)
        yf_dvy, vxf_dvy = traj_dvy[-1][1], traj_dvy[-1][3]

        # Jacobian
        J = np.zeros((2, 2))
        J[0, 0] = (yf_dx - yf) / eps_x
        J[0, 1] = (yf_dvy - yf) / eps_vy
        J[1, 0] = (vxf_dx - vxf) / eps_x
        J[1, 1] = (vxf_dvy - vxf) / eps_vy

        # Check determinant
        detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        if abs(detJ) < 1e-15:
            print("Jacobian singular (det ≈ 0).")
            break

        try:
            delta = np.linalg.solve(J, -res)
        except:
            print("Linear solve failed.")
            break

        # Dampen correction to avoid overshoot
        damp = 0.8
        x0 += damp * delta[0]
        vy0 += damp * delta[1]

        # Optional: adjust t_half based on y-crossing (advanced)

    print(f"Failed to converge. Final residual: {err:.2e}")
    return state0, 2*t_half, traj