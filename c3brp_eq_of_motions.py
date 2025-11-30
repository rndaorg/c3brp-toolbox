# Derive and define CR3BP equations of motion in rotating frame
# Non-dimensionalized using characteristic quantities:
# - Distance unit: distance between primaries (e.g., Earth-Moon = d_EM)
# - Time unit: sqrt(d_EM^3 / (G*(M1+M2)))
# - Mass unit: M1 + M2
# - mu = M2 / (M1 + M2)

import sympy as sp

# Define symbols
x, y, vx, vy = sp.symbols('x y vx vy')
mu = sp.symbols('mu', real=True, positive=True)

# Positions of primaries in rotating frame
x1 = -mu      # Primary 1 (e.g., Earth)
x2 = 1 - mu   # Primary 2 (e.g., Moon)

# Distances from particle to each primary
r1 = sp.sqrt((x - x1)**2 + y**2)
r2 = sp.sqrt((x - x2)**2 + y**2)

# Pseudo-potential (Omega)
Omega = (x**2 + y**2)/2 + (1 - mu)/r1 + mu/r2

# Equations of motion: 
# x'' = 2*vy + dOmega/dx
# y'' = -2*vx + dOmega/dy

dOmega_dx = sp.diff(Omega, x)
dOmega_dy = sp.diff(Omega, y)

# Accelerations
ax = 2*vy + dOmega_dx
ay = -2*vx + dOmega_dy

# Display equations
ax_simplified = sp.simplify(ax)
ay_simplified = sp.simplify(ay)

print(ax_simplified) 
print(ay_simplified)