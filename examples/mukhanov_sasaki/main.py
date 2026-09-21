"""Solve Mukhanov-Sasaki modes for a quadratic inflation example.

The Mukhanov-Sasaki equation describes the gauge-invariant scalar perturbation
that seeds the CMB temperature anisotropies and the large-scale distribution of
matter.  For a single inflaton field on an FLRW background, the canonical
perturbation variable v obeys

    v_k'' + (k^2 - z'' / z) v_k = 0,

where primes denote derivatives with respect to conformal time, k is the
comoving wavenumber, z = a dphi/dN in reduced Planck units, a is the scale
factor, phi is the homogeneous inflaton, and N = log(a) is e-fold time.  The
quantity z'' / z is a time-dependent effective mass that couples each quantum
fluctuation to the evolving background geometry.

The standard initial condition is the Bunch-Davies vacuum.  Deep inside the
horizon, k >> aH, curvature and expansion are negligible over the wavelength,
so each mode behaves like a flat-spacetime oscillator:

    v_k ~= exp(-i k tau) / sqrt(2 k).

The equation follows from expanding the Einstein-Hilbert plus scalar-field
action to second order around an FLRW background.  Metric and inflaton
perturbations are combined into one gauge-invariant variable, v, and applying
the Euler-Lagrange equations to the quadratic action gives the linear
Mukhanov-Sasaki equation.  Spatial homogeneity and isotropy make the background
depend only on time; after a Fourier transform, the linearized PDE splits into
independent ODEs labelled by |k|.  That independence is why this example maps
many k modes onto a batched ensemble solve.

The final frozen curvature perturbation is R_k = v_k / z.  Its dimensionless
power spectrum is P_R(k) = k^3 |R_k|^2 / (2 pi^2).  Einstein-Boltzmann solvers
commonly take this primordial spectrum through its amplitude A_s at a pivot
scale and its scalar spectral index n_s = d log(P_R) / d log(k).

In the code below the mode equation is solved in e-fold time N, not conformal
time.  With epsilon = -d log(H) / dN, q = (z'' / z) / (aH)^2, and
x = k / (aH), the complex mode equation becomes

    d^2 v_k/dN^2 + (1 - epsilon) dv_k/dN + (x^2 - q) v_k = 0.

The solver state stores real and imaginary parts separately:

    y = [Re(v_k), Im(v_k), Re(dv_k/dN), Im(dv_k/dN)].

Each trajectory is integrated over a normalized time s in [0, 1], where
N = N_start + s (N_stop - N_start), so the right-hand side returned to the
solver is multiplied by dN/ds = N_stop - N_start.

References:
    D. Baumann, "TASI Lectures on Inflation", arXiv:0907.5424.

Usage:
    uv run python examples/mukhanov_sasaki/main.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.dual_backend import Forms, build_fn, build_rhs
from solvers.tsit5 import solve as tsit5_solve

jax.config.update("jax_enable_x64", True)

N_MODES = 41  # Number of Fourier modes in the ensemble solve.
K_MIN_MPC = 1.0e-4  # Smallest physical comoving wavenumber, in Mpc^-1.
K_MAX_MPC = 1.0  # Largest physical comoving wavenumber, in Mpc^-1.
K_PIVOT_MPC = 0.05  # Pivot scale used for A_s and n_s, in Mpc^-1.

TARGET_A_S = 2.1e-9  # Observed scalar amplitude used to set the mass scale.
N_PIVOT_BEFORE_END = 55.0  # E-folds between pivot horizon exit and inflation end.

PHI_INITIAL = 17.5  # Initial homogeneous inflaton value, in reduced Planck units.
D_PHI_DN_INITIAL = -2.0 / PHI_INITIAL  # Slow-roll initial dphi/dN estimate.
N_BACKGROUND_MAX = 80.0  # Maximum e-fold time for the background integration.
N_BACKGROUND_SAMPLES = 2000  # Save points used to build interpolation tables.

X_INITIAL = 80.0  # Initial subhorizon ratio k/(aH) for each mode.
X_FINAL = 1.0e-3  # Final superhorizon ratio k/(aH) for each mode.

BACKGROUND_RTOL = 1.0e-9  # Relative tolerance for the background solve.
BACKGROUND_ATOL = 1.0e-11  # Absolute tolerance for the background solve.
MODE_RTOL = 1.0e-7  # Relative tolerance for the Mukhanov-Sasaki mode solve.
MODE_ATOL = 1.0e-9  # Absolute tolerance for the Mukhanov-Sasaki mode solve.
MODE_MAX_STEPS = 200000  # Step cap per mode trajectory.


def quadratic_mass_from_slow_roll(
    target_a_s: float = TARGET_A_S,
    n_star: float = N_PIVOT_BEFORE_END,
) -> float:
    """Return m for V = 0.5 m^2 phi^2 from the slow-roll amplitude."""
    phi_star_sq = 4.0 * n_star + 2.0
    epsilon_v = 2.0 / phi_star_sq
    return float(np.sqrt(target_a_s * 48.0 * np.pi**2 * epsilon_v / phi_star_sq))


MASS = quadratic_mass_from_slow_roll()


# The background solve needs its right-hand side in two forms: ``tsit5``
# compiles one with ``numba_cuda_mlir``, and the Diffrax reference backend
# traces the other.  Both are built from one body by
# ``examples/dual_backend.py``; ``potential`` goes the same way because the
# device callback and the NumPy table building below both call it.


def _make_potential():
    def potential(phi, mass):
        """Evaluate the quadratic inflaton potential in reduced Planck units."""
        return 0.5 * mass**2 * phi**2

    return potential


def _make_dpotential_dphi():
    def dpotential_dphi(phi, mass):
        """Evaluate dV/dphi for the quadratic inflaton potential."""
        return mass**2 * phi

    return dpotential_dphi


POTENTIAL = build_fn(_make_potential)
DPOTENTIAL_DPHI = build_fn(_make_dpotential_dphi)
potential = POTENTIAL.jax
dpotential_dphi = DPOTENTIAL_DPHI.jax


def _make_background_ode(*, potential, dpotential_dphi):
    def background_ode(y, n_efolds, params):
        """Return dphi/dN and d2phi/dN^2 for the homogeneous inflaton background.

        Autonomous, so ``n_efolds`` goes unused -- and unmentioned: numba
        rejects a ``del`` of an argument in a device function.
        """
        mass = params[0]
        phi = y[0]
        dphi_dn = y[1]
        epsilon = 0.5 * dphi_dn**2
        h_sq = potential(phi, mass) / (3.0 - epsilon)
        d2phi_dn2 = -(3.0 - epsilon) * dphi_dn - dpotential_dphi(phi, mass) / h_sq
        return (dphi_dn, d2phi_dn2)

    return background_ode


BACKGROUND_ODE = build_rhs(
    _make_background_ode, potential=POTENTIAL, dpotential_dphi=DPOTENTIAL_DPHI
)
background_ode = BACKGROUND_ODE.jax
background_ode_device = BACKGROUND_ODE.device


def solve_background():
    """Integrate the homogeneous inflationary background over e-fold time."""
    times = jnp.linspace(0.0, N_BACKGROUND_MAX, N_BACKGROUND_SAMPLES)
    y0 = jnp.array([PHI_INITIAL, D_PHI_DN_INITIAL], dtype=jnp.float64)
    params = jnp.array([MASS], dtype=jnp.float64)
    solution = tsit5_solve(
        background_ode_device,
        y0,
        times,
        params,
        rtol=BACKGROUND_RTOL,
        atol=BACKGROUND_ATOL,
        first_step=1.0e-4,
    )[0]
    return np.asarray(times), np.asarray(solution)


def first_crossing(x, y, value):
    """Linearly interpolate the first x location where y crosses value upward."""
    below = np.nonzero(y >= value)[0]
    if below.size == 0 or below[0] == 0:
        raise RuntimeError(f"Could not find crossing for value {value}.")
    i = below[0]
    x0, x1 = x[i - 1], x[i]
    y0, y1 = y[i - 1], y[i]
    return float(x0 + (value - y0) * (x1 - x0) / (y1 - y0))


def build_background_tables(n_grid, background):
    """Build inflationary interpolation tables used by the mode equations."""
    phi_full = background[:, 0]
    dphi_dn_full = background[:, 1]
    epsilon_full = 0.5 * dphi_dn_full**2
    n_end = first_crossing(n_grid, epsilon_full, 1.0)
    n_pivot = n_end - N_PIVOT_BEFORE_END
    if n_pivot <= n_grid[0]:
        raise RuntimeError("Background does not start early enough for the pivot.")

    inflationary = n_grid <= n_end
    n_grid = n_grid[inflationary]
    phi = phi_full[inflationary]
    dphi_dn = dphi_dn_full[inflationary]
    epsilon = 0.5 * dphi_dn**2

    h = np.sqrt(potential(phi, MASS) / (3.0 - epsilon))
    log_a_h = n_grid + np.log(h)
    z = np.exp(n_grid) * dphi_dn
    dz_dn = np.gradient(z, n_grid, edge_order=2)
    d2z_dn2 = np.gradient(dz_dn, n_grid, edge_order=2)
    q = d2z_dn2 / z + (1.0 - epsilon) * dz_dn / z

    return {
        "n": n_grid,
        "phi": phi,
        "dphi_dn": dphi_dn,
        "epsilon": epsilon,
        "h": h,
        "log_a_h": log_a_h,
        "z": z,
        "q": q,
        "n_end": n_end,
        "n_pivot": n_pivot,
    }


def interp_np(x, xp, fp):
    """Return one-dimensional NumPy linear interpolation as a Python float."""
    return float(np.interp(x, xp, fp))


def prepare_mode_problem(tables, n_modes=N_MODES):
    """Create k values, per-mode integration windows, and Bunch-Davies y0."""
    physical_k = np.geomspace(K_MIN_MPC, K_MAX_MPC, n_modes)
    log_a_h_pivot = interp_np(tables["n_pivot"], tables["n"], tables["log_a_h"])
    code_k_pivot = np.exp(log_a_h_pivot)
    code_k = code_k_pivot * physical_k / K_PIVOT_MPC
    log_code_k = np.log(code_k)

    n_start = np.interp(log_code_k - np.log(X_INITIAL), tables["log_a_h"], tables["n"])
    n_stop = np.interp(log_code_k - np.log(X_FINAL), tables["log_a_h"], tables["n"])

    phase = X_INITIAL
    norm = 1.0 / np.sqrt(2.0 * code_k)
    v_re = norm * np.cos(phase)
    v_im = norm * np.sin(phase)
    dv_dn_re = X_INITIAL * v_im
    dv_dn_im = -X_INITIAL * v_re
    y0 = np.stack([v_re, v_im, dv_dn_re, dv_dn_im], axis=1)
    params = np.stack([code_k, n_start, n_stop], axis=1)
    return physical_k, code_k, y0, params


def _make_mode_ode(*, exp, maximum, minimum, index_of, table, grid):
    """The Mukhanov-Sasaki right-hand side over a uniform background grid.

    ``jnp.interp`` has no device equivalent, so the three background lookups
    are done by hand.  ``build_background_tables`` keeps the uniform
    ``np.linspace`` grid that ``solve_background`` produced (it only trims a
    trailing slice), so the bracketing index is arithmetic rather than a
    search, and clamping at both ends reproduces ``np.interp``'s behaviour
    outside the table.
    """
    n_first, dn, last = grid

    def mode_ode(y, s, params):
        tab = table  # bind once: see the constant-memory note in make_mode_ode
        code_k = params[0]
        n_start = params[1]
        n_stop = params[2]
        delta_n = n_stop - n_start
        n_now = n_start + s * delta_n

        pos = minimum(maximum((n_now - n_first) / dn, 0.0), float(last))
        i = minimum(index_of(pos), last - 1)
        frac = pos - i

        epsilon = tab[i, 0] + frac * (tab[i + 1, 0] - tab[i, 0])
        log_a_h = tab[i, 1] + frac * (tab[i + 1, 1] - tab[i, 1])
        q = tab[i, 2] + frac * (tab[i + 1, 2] - tab[i, 2])

        k_over_a_h = code_k * exp(-log_a_h)
        omega_sq = k_over_a_h * k_over_a_h - q

        v_re = y[0]
        v_im = y[1]
        dv_re = y[2]
        dv_im = y[3]
        d2v_re = -(1.0 - epsilon) * dv_re - omega_sq * v_re
        d2v_im = -(1.0 - epsilon) * dv_im - omega_sq * v_im
        return (
            delta_n * dv_re,
            delta_n * dv_im,
            delta_n * d2v_re,
            delta_n * d2v_im,
        )

    return mode_ode


def make_mode_ode(tables):
    """Build the Mukhanov-Sasaki RHS over background interpolation tables.

    The tables are closed over, which numba lowers into CUDA *constant* memory
    -- a hard 64 KiB per module.  Two details keep the footprint at one copy of
    the data: the three columns are packed into a single row-major ``(n, 3)``
    array, and it is bound to a local before indexing.  Indexing a closed-over
    array directly emits one constant copy per reference site, which for three
    separate tables read twice each came to ~135 KiB and would not load.
    Row-major packing also puts the three values for a given ``n`` adjacent,
    so one bracket costs two cache lines rather than six.
    """
    n_table = np.ascontiguousarray(tables["n"], dtype=np.float64)
    spacing = np.diff(n_table)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=0.0):
        raise ValueError("the mode RHS requires a uniformly spaced N grid")

    table = np.ascontiguousarray(
        np.stack(
            [
                np.asarray(tables["epsilon"], dtype=np.float64),
                np.asarray(tables["log_a_h"], dtype=np.float64),
                np.asarray(tables["q"], dtype=np.float64),
            ]
        ).T
    )
    grid = (float(n_table[0]), float(spacing[0]), n_table.size - 1)
    return build_rhs(
        _make_mode_ode,
        # The traced form indexes a device array; the device form closes over
        # the NumPy one, which is what lands in constant memory.
        table=Forms(table, jnp.asarray(table), table),
        grid=grid,
    )


class ModeODE:
    """Picklable traced mode RHS, for the scipy backend's worker processes.

    ``make_mode_ode(...).jax`` is a closure, and stdlib pickle carries a
    function by name; ``multiprocessing`` therefore cannot send it to a
    worker.  This carries the background tables instead -- plain arrays --
    and rebuilds the closure on the other side.
    """

    _TABLES = ("n", "epsilon", "log_a_h", "q")

    def __init__(self, tables):
        self.tables = {key: np.asarray(tables[key]) for key in self._TABLES}
        self._rhs = make_mode_ode(self.tables).jax

    def __call__(self, y, s, params):
        return self._rhs(y, s, params)

    def __getstate__(self):
        return self.tables

    def __setstate__(self, tables):
        self.__init__(tables)


def make_solver(backend):
    """Return a uniform ``solve(ode_fn, y0, s_span, params)`` for a backend.

    The mode equation is non-stiff and oscillatory, so the science uses the
    explicit modax Tsit5 solver.  Two reference backends integrate the
    identical complex mode equation for a like-for-like timing comparison:
      * "scipy"   -- serial CPU integration with scipy.solve_ivp (RK45), the
                     no-GPU baseline.
      * "diffrax" -- GPU integration with plain Diffrax Tsit5 (jax.vmap), the
                     explicit analogue of the Kvaerno5 baseline used for the
                     stiff examples (Kvaerno5 is an implicit method and is a
                     poor match for this non-stiff oscillatory problem).
    """
    if backend == "modax":
        return lambda f, y0, ts, p: tsit5_solve(
            f,
            y0,
            ts,
            p,
            rtol=MODE_RTOL,
            atol=MODE_ATOL,
            first_step=1.0e-5,
            max_steps=MODE_MAX_STEPS,
        )
    if backend == "diffrax":
        from reference.solvers.python.diffrax_tsit5 import solve as diffrax_solve

        return lambda f, y0, ts, p: diffrax_solve(
            f,
            y0,
            ts,
            p,
            rtol=MODE_RTOL,
            atol=MODE_ATOL,
            first_step=1.0e-5,
            max_steps=MODE_MAX_STEPS,
        )
    if backend == "scipy":
        from reference.solvers.python.scipy_solve_ivp import solve as scipy_solve

        return lambda f, y0, ts, p: scipy_solve(
            f,
            y0,
            ts,
            p,
            method="RK45",
            rtol=MODE_RTOL,
            atol=MODE_ATOL,
            first_step=1.0e-5,
        )
    raise ValueError(f"unknown backend: {backend}")


def solve_modes(tables, backend="modax", n_modes=N_MODES):
    """Solve all uncoupled Mukhanov-Sasaki Fourier modes as one ensemble."""
    physical_k, code_k, y0, params = prepare_mode_problem(tables, n_modes)
    solve_fn = make_solver(backend)
    # The modax kernel solver compiles its RHS with numba-cuda; the reference
    # backend traces the jnp form built from the same body.
    ode_fn = make_mode_ode(tables).device if backend == "modax" else ModeODE(tables)
    solution = solve_fn(
        ode_fn,
        jnp.asarray(y0, dtype=jnp.float64),
        jnp.array([0.0, 1.0], dtype=jnp.float64),
        jnp.asarray(params, dtype=jnp.float64),
    )
    return physical_k, code_k, params[:, 2], np.asarray(solution[:, -1, :])


def compute_power_spectrum(tables, code_k, n_stop, final_state):
    """Convert final mode amplitudes into the dimensionless curvature spectrum."""
    z_stop = np.interp(n_stop, tables["n"], tables["z"])
    v_abs_sq = final_state[:, 0] ** 2 + final_state[:, 1] ** 2
    return code_k**3 * v_abs_sq / z_stop**2 / (2.0 * np.pi**2)


def local_spectral_index(k, power, pivot):
    """Estimate n_s from a local log-log slope of the numerical spectrum."""
    width = 7 if k.size >= 7 else k.size
    center = int(np.argmin(np.abs(np.log(k / pivot))))
    lo = max(0, center - width // 2)
    hi = min(k.size, lo + width)
    lo = max(0, hi - width)
    slope, _ = np.polyfit(np.log(k[lo:hi]), np.log(power[lo:hi]), deg=1)
    return float(1.0 + slope)


def log_interp(x, xp, fp):
    """Interpolate a positive quantity linearly in log-log space."""
    return float(np.exp(np.interp(np.log(x), np.log(xp), np.log(fp))))


def slow_roll_estimates(tables):
    """Compute approximate quadratic-inflation slow-roll values at the pivot."""
    n_star = tables["n_end"] - tables["n_pivot"]
    phi_pivot = interp_np(tables["n_pivot"], tables["n"], tables["phi"])
    epsilon_v = 1.0 / (2.0 * n_star + 1.0)
    eta_v = epsilon_v
    v_pivot = potential(phi_pivot, MASS)
    a_s = v_pivot / (24.0 * np.pi**2 * epsilon_v)
    n_s = 1.0 - 6.0 * epsilon_v + 2.0 * eta_v
    n_s_compact = 1.0 - 2.0 / n_star
    return {
        "n_star": n_star,
        "epsilon_v": epsilon_v,
        "eta_v": eta_v,
        "A_s": a_s,
        "n_s": n_s,
        "n_s_compact": n_s_compact,
    }


def print_results(results):
    """Print numerical and slow-roll primordial-spectrum outputs side by side."""
    a_s = results["A_s"]
    n_s = results["n_s"]
    a_s_sr = results["A_s_slow_roll"]
    n_s_sr = results["n_s_slow_roll"]
    print("Mukhanov-Sasaki scalar spectrum from quadratic inflation")
    print(f"m                 = {MASS:.6e} M_pl")
    print(f"N modes           = {N_MODES}")
    print(f"k pivot           = {K_PIVOT_MPC:.6g} Mpc^-1")
    print(f"N_end             = {results['background']['N_end']:.6f}")
    print(f"N_*               = {results['background']['N_star']:.6f}")
    print()
    print("quantity    numerical MS        slow-roll estimate    diff")
    print(
        f"A_s         {a_s: .8e}       {a_s_sr: .8e}       "
        f"{(a_s - a_s_sr) / a_s_sr: .3e} rel"
    )
    print(f"n_s         {n_s: .8f}       {n_s_sr: .8f}       {n_s - n_s_sr: .3e} abs")
    print(
        f"n_s compact slow-roll sanity estimate: {results['n_s_slow_roll_compact']:.8f}"
    )


def time_solve(fn, repeats):
    """Return (mean seconds excluding compile, result) over ``repeats`` runs."""
    result = fn()
    jax.block_until_ready(result)
    t0 = time.perf_counter()
    for _ in range(repeats):
        result = fn()
        jax.block_until_ready(result)
    return (time.perf_counter() - t0) / repeats, result


def run_benchmark(n_modes, backends, repeats):
    n_grid, background = solve_background()
    tables = build_background_tables(n_grid, background)
    physical_k, code_k, y0, params = prepare_mode_problem(tables, n_modes)
    # The modax kernel solver compiles its RHS with numba-cuda; the reference
    # backends trace the jnp form of the same equations.
    mode_ode_device = make_mode_ode(tables).device
    mode_ode = ModeODE(tables)
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    params = jnp.asarray(params, dtype=jnp.float64)
    s_span = jnp.array([0.0, 1.0], dtype=jnp.float64)
    print(f"Mukhanov-Sasaki benchmark: N = {n_modes:,} uncoupled k-modes\n")
    print(f"{'backend':>10}  {'wall (s)':>10}  {'per solve':>12}  A_s(pivot)")
    print("-" * 56)
    for backend in backends:
        solve_fn = make_solver(backend)
        f = mode_ode_device if backend == "modax" else mode_ode
        run = lambda sf=solve_fn, fn=f: sf(fn, y0, s_span, params)
        try:
            secs, sol = time_solve(run, repeats)
        except Exception as exc:  # noqa: BLE001
            print(f"{backend:>10}  FAILED: {exc}")
            continue
        final_state = np.asarray(sol)[:, -1, :]
        power = compute_power_spectrum(tables, code_k, params[:, 2], final_state)
        a_s = log_interp(K_PIVOT_MPC, physical_k, power)
        print(f"{backend:>10}  {secs:10.3f}  {secs / n_modes * 1e3:9.4f} ms  {a_s:.4e}")


def main():
    """Run the background solve, mode solve, spectrum extraction, and reporting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="time the batched mode solve across solver backends",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["modax", "diffrax", "scipy"],
        choices=["modax", "diffrax", "scipy"],
    )
    parser.add_argument("--n", type=int, default=4096, help="number of k-modes")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.n, args.backends, args.repeats)
        return

    n_grid, background = solve_background()
    tables = build_background_tables(n_grid, background)
    physical_k, code_k, n_stop, final_state = solve_modes(tables)
    power = compute_power_spectrum(tables, code_k, n_stop, final_state)
    a_s = log_interp(K_PIVOT_MPC, physical_k, power)
    n_s = local_spectral_index(physical_k, power, K_PIVOT_MPC)
    slow_roll = slow_roll_estimates(tables)

    results = {
        "A_s": a_s,
        "n_s": n_s,
        "A_s_slow_roll": slow_roll["A_s"],
        "n_s_slow_roll": slow_roll["n_s"],
        "n_s_slow_roll_compact": slow_roll["n_s_compact"],
        "k_pivot": K_PIVOT_MPC,
        "k": physical_k,
        "power_spectrum": power,
        "background": {
            "N_end": tables["n_end"],
            "N_pivot": tables["n_pivot"],
            "N_star": slow_roll["n_star"],
            "epsilon_V": slow_roll["epsilon_v"],
            "eta_V": slow_roll["eta_v"],
        },
    }
    print_results(results)
    return results


if __name__ == "__main__":
    main()
