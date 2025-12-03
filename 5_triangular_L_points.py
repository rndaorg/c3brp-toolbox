import numpy as np
import matplotlib.pyplot as plt

from c3brp.jacobian import jacobian
from const.const import mu_earth_moon

def compute_L4_L5(mu):
    """
    Compute triangular Lagrange points L4 and L5.
    In normalized CR3BP coordinates.
    """
    x_L4 = 0.5 - mu
    y_L4 = np.sqrt(3) / 2
    x_L5 = 0.5 - mu
    y_L5 = -np.sqrt(3) / 2
    return (x_L4, y_L4), (x_L5, y_L5)


def analyze_stability(mu):
    (x4, y4), (x5, y5) = compute_L4_L5(mu)

    J4 = jacobian(x4, y4, mu)
    J5 = jacobian(x5, y5, mu)

    eigvals4 = np.linalg.eigvals(J4)
    eigvals5 = np.linalg.eigvals(J5)

    # Due to symmetry, L4 and L5 have identical eigenvalues
    return eigvals4, (x4, y4), (x5, y5)


def plot_system_and_eigenvalues(mu):
    eigvals, L4, L5 = analyze_stability(mu)
    x4, y4 = L4
    x5, y5 = L5

    # Plot configuration
    plt.figure(figsize=(12, 5))

    # --- Left: Geometry ---
    #plt.subplot(1, 2, 1)
    x1, y1 = -mu, 0
    x2, y2 = 1 - mu, 0
    xbnd, ybnd = -1, 0

    plt.scatter([x1], [y1], s=300, label=f'$m_1 = {1 - mu:.3f}$', c='orange')
    plt.scatter([x2], [y2], s=150, label=f'$m_2 = {mu:.3f}$', c='blue')
    plt.scatter([x4, x5], [y4, y5], c='red', marker='^', s=100, label='$L_4, L_5$')

    # Draw equilateral triangle
    plt.plot([x1, x2, x4, x1], [y1, y2, y4, y1], 'k--', alpha=0.5)
    plt.plot([x1, x2, x5, x1], [y1, y2, y5, y1], 'k--', alpha=0.5)

    plt.plot([xbnd], [ybnd])

    plt.gca().set_aspect('equal')
    plt.title('Triangular Lagrange Points (L4, L5)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle=':')
    plt.legend()

    # --- Right: Eigenvalues ---
    '''
    plt.subplot(1, 2, 2)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.scatter(eigvals.real, eigvals.imag, c='purple', s=60, label='Eigenvalues')

    plt.title('Eigenvalues of Linearized System at L4/L5')
    plt.xlabel('Re(λ)')
    plt.ylabel('Im(λ)')
    plt.grid(True, linestyle=':')
    plt.legend()
    '''


    # Stability criterion:
    # For L4/L5 to be linearly stable: μ < μ_crit = (1 - sqrt(23/27))/2 ≈ 0.03852
    mu_crit = (1 - np.sqrt(23/27)) / 2
    stable = mu < mu_crit
    plt.title(f'Triangular Lagrange Points (L4, L5) | μ = {mu:.4f} → L4/L5 are {"linearly stable" if stable else "unstable"}\n'
                 f'(Critical μ = {mu_crit:.5f})')
    

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.savefig("tri_L_points.png")

    return eigvals, stable

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Try Earth-Moon system: μ ≈ 0.01215 < 0.0385 → stable
    eigvals, stable = plot_system_and_eigenvalues(mu_earth_moon)

    # Try Sun-Jupiter: μ ≈ 0.00095 → stable
    # mu_sun_jup = 0.0009537
    # plot_system_and_eigenvalues(mu_sun_jup)

    # Try μ = 0.1 (> critical) → unstable
    # plot_system_and_eigenvalues(0.1)