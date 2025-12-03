* 1: Derive the equations of motion for the planar CR3BP in rotating frame. Non-dimensionalize using characteristic quantities.
* 2: Implement a Python function for CR3BP equations of motion (state = [x, y, vx, vy]).
 3: Integrate CR3BP trajectories using RK4 or scipy.integrate.solve_ivp. Visualize sample trajectories.
 4: Compute and plot zero-velocity curves (Jacobi constant contours).
 5: Locate and compute collinear Lagrange points (L1–L3) numerically via root-finding.
 6: Compute and plot triangular Lagrange points (L4, L5) and analyze stability via linearization.
 7: Build a simple interactive CR3BP trajectory viewer (e.g., using Matplotlib sliders or Plotly).
📅  2: Invariant Manifolds and Low-Energy Pathways
Goal: Generate and visualize manifolds around libration point orbits.

 8: Compute a Lyapunov orbit around L1 using differential correction (single-shooting).
 9: Linearize dynamics around a periodic orbit. Compute monodromy matrix.
 10: Extract stable/unstable eigenvectors. Propagate them to generate manifolds.
 11: Visualize stable/unstable manifolds in rotating and inertial frames.
 12: Repeat manifold generation for L2 and compare Earth–Moon vs. Jupiter–Europa.
 13: Design a heteroclinic connection between L1 and L2 manifolds (conceptual + basic code).
 14: Add a simple event detection for manifold-to-orbit intersections.
📅  3: Multi-System Scaling & Flyby Sequencing
Goal: Generalize to Jovian/Saturnian systems and include moon flybys.

 15: Parameterize CR3BP for Europa–Jupiter, Titan–Saturn. Compare μ and distance scales.
 16: Build a system selector: input primary/secondary bodies → auto-set μ, distance, time units.
 17: Implement patched-conic flybys near moons using 2-body approximations.
 18: Detect and avoid collisions with moons or primaries during propagation.
 19: Design a “moon-hopping” trajectory using manifold-guided flybys (e.g., Ganymede → Europa).
 20: Add Jacobi constant continuity checks across system transitions.
 21: Visualize multi-moon trajectories in 3D (optional: convert to inertial J2000).
📅  4: Mission Design Interface & Optimization
Goal: Build a usable mission design tool with basic optimization and export.

 22: Wrap CR3BP solver into a class: CR3BPSystem(μ, Lstar, Tstar).
 23: Add JSON/YAML input for mission specification (bodies, duration, initial guess).
 24: Implement a simple direct collocation or single-shooting optimizer for orbit targeting.
 25: Add Lambert solver for inter-moon transfers outside CR3BP regions.
 26: Create a “low-energy transfer designer” CLI that chains manifold arcs + flybys.
 27: Export trajectories to CSV or GMAT/STK-compatible format.
 28: Add error handling for failed integrations or Lambert solvers.
 29: Document the code with docstrings and example Jupyter notebooks.
 30: Package as a minimal Python library (e.g., lowfimission) with pip installability.