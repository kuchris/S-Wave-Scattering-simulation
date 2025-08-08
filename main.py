import numpy as np

def solve_s_wave_scattering_numerov(
    energy,
    well_depth,
    well_radius,
    r_max=50.0,
    step=0.001,
    mu=1.0,
    hbar=1.0,
):
    """
    Solve the radial Schrödinger equation for s-wave (l=0) scattering
    off a square-well potential using Numerov's method.

    Equation (l = 0):
        u''(r) + k^2(r) u(r) = 0,
        where k^2(r) = (2*mu/hbar^2) * (E - V(r))
        and u(r) = r * R(r) is the reduced radial wavefunction.

    Potential:
        V(r) = -well_depth, for r < well_radius
        V(r) = 0,            for r >= well_radius

    Asymptotic (outside the potential, r >> well_radius):
        u(r) ~ sin(k*r + delta)
        with k = sqrt((2*mu/hbar^2) * E)

    Phase shift extraction:
        Let ϕ = k*r + delta. Then:
            u(r) = sin(ϕ),  u'(r) = k*cos(ϕ)
            => tan(ϕ) = k*u(r)/u'(r)
            => delta = atan(k*u/u') - k*r
        We evaluate at a matching point r_match outside the well.

    Parameters
    ----------
    energy : float
        Scattering energy E (> 0).
    well_depth : float
        Square well depth (positive number, potential is -well_depth inside).
    well_radius : float
        Square well radius a.
    r_max : float, optional
        Maximum radius to integrate to (must be > well_radius).
    step : float, optional
        Radial grid spacing (Numerov step size).
    mu : float, optional
        Reduced mass.
    hbar : float, optional
        Reduced Planck's constant.

    Returns
    -------
    r : np.ndarray
        Radial grid, shape (N,).
    u : np.ndarray
        Reduced radial wavefunction u(r) on the grid (unnormalized), shape (N,).
    delta : float
        s-wave phase shift (in radians).
    k_out : float
        Asymptotic wavenumber outside the potential: sqrt((2*mu/hbar^2)*E).

    Notes
    -----
    - Units: By default mu = hbar = 1 gives the common choice hbar^2/(2*mu) = 0.5.
      You can set units so that hbar^2/(2*mu) = 1 by choosing (mu=0.5, hbar=1).
    - For very small energy, choose a larger r_max to ensure a good asymptotic region.
    """

    if energy <= 0.0:
        raise ValueError("Energy must be positive for scattering.")
    if r_max <= well_radius + 20 * step:
        raise ValueError("r_max should be sufficiently larger than well_radius for asymptotics.")

    # Precompute constant factor: (2*mu / hbar^2)
    s = 2.0 * mu / (hbar * hbar)

    # Asymptotic wavenumber outside the potential (V=0)
    k_out = np.sqrt(s * energy)

    # Radial grid
    r = np.arange(0.0, r_max + step, step)
    n_points = r.size

    # Define the square-well potential on the grid: V(r) = -well_depth inside, 0 outside
    V = np.where(r < well_radius, -well_depth, 0.0)

    # k^2(r) = s * (E - V(r))
    k2 = s * (energy - V)

    # Allocate solution array for u(r)
    u = np.zeros_like(r)

    # Initial conditions near r=0 for l=0:
    # u(r) ~ r, so choose u(0)=0 and u(step)=step (arbitrary normalization).
    u[0] = 0.0
    u[1] = step

    # Numerov coefficients reused
    h2 = step * step
    one_over_twelve = 1.0 / 12.0
    five_over_six = 5.0 / 6.0

    # Numerov integration outward:
    # y_{n+1} = [ 2*(1 - 5 h^2 g_n / 12) y_n - (1 + h^2 g_{n-1} / 12) y_{n-1} ] / (1 + h^2 g_{n+1} / 12)
    # where g_n = k^2(r_n).
    for i in range(1, n_points - 1):
        g_im1 = k2[i - 1]
        g_i = k2[i]
        g_ip1 = k2[i + 1]

        denom = 1.0 + h2 * g_ip1 * one_over_twelve
        term1 = 2.0 * (1.0 - h2 * g_i * (5.0 * one_over_twelve)) * u[i]
        term2 = (1.0 + h2 * g_im1 * one_over_twelve) * u[i - 1]
        u[i + 1] = (term1 - term2) / denom

    # Choose a matching point in the asymptotic region.
    # Use a point safely outside the well and away from the boundary to allow central differences.
    # We pick the largest index m such that r[m] > well_radius + 10*step and m in [1, n_points-2]
    candidates = np.where(r > well_radius + 10.0 * step)[0]
    if candidates.size == 0 or candidates[-1] >= n_points - 1:
        # Fallback: pick a point near the end but not at the boundary
        m = max(1, n_points - 2)
    else:
        # Prefer a point fairly close to r_max but not at the boundary
        m = min(candidates[-1], n_points - 2)

    # Numerical derivative u'(r_m) via central difference
    up = (u[m + 1] - u[m - 1]) / (2.0 * step)

    # Extract s-wave phase shift:
    # delta = atan( k_out * u / u' ) - k_out * r
    delta = np.arctan2(k_out * u[m], up) - k_out * r[m]

    # Optionally wrap to (-pi, pi] for cleanliness
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi

    return r, u, delta, k_out


# Example usage:
# E = 1.0
# V0 = 5.0
# a = 1.0
# r, u, delta, k = solve_s_wave_scattering_numerov(E, V0, a, r_max=50.0, step=0.001)
# total_cross_section_swave = 4.0 * np.pi * (np.sin(delta) ** 2) / (k ** 2)

import matplotlib.pyplot as plt

# Example parameters
E = 1.0
V0 = 5.0
a = 1.0

r, u, delta, k = solve_s_wave_scattering_numerov(E, V0, a, r_max=50.0, step=0.001)

plt.figure(figsize=(8, 5))
plt.plot(r, u, label="Radial wavefunction $u(r)$")
plt.xlabel("$r$")
plt.ylabel("$u(r)$")
plt.title("S-wave radial wavefunction for $E=1.0$, $V_0=5.0$, $a=1.0$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()