"""Inverse Parameter Estimation: Big Bang Nucleosynthesis (BBN) Reaction Networks.

If you want a rigorous test of a solver's ability to handle stiffness across a
massive batch, BBN is an excellent benchmark.

The Physics: During the first few minutes of the universe, the temperature and
density dropped rapidly as the universe expanded. The production of light
elements (Deuterium, Helium-3, Helium-4, Lithium-7) is governed by a network
of nuclear reaction rates. This is modeled as a system of coupled, highly stiff
ODEs tracking the mass fractions of these isotopes as a function of time (or
temperature).

The Inverse Problem: We have highly precise modern observations of the
primordial abundances of these elements. The inverse problem is to find the
exact cosmological parameters—specifically the baryon-to-photon ratio (eta) and
the effective number of neutrino species (N_eff)—that, when plugged into the
ODEs, produce a final state that matches those observations.

Why it's a Massively Batched, Uncoupled Problem:
To fit these parameters, you typically use an optimization algorithm (like
L-BFGS) or a Bayesian inference method (like Hamiltonian Monte Carlo).
If you are mapping the parameter space via a grid search to visualize the chi^2
surface, you might evaluate a 1000 x 1000 grid of (eta, N_eff).
This means you have 1,000,000 distinct ODE systems to integrate from
t=10^-2 seconds to t=10^4 seconds.
Each integration represents an entirely independent universe with slightly
different initial conditions or expansion rates. There is zero cross-talk
between the batches, allowing you to parallelize the solver massively across
GPU threads to calculate the loss landscape in seconds.

This example integrates a 4-species BBN network (n, p, D, 4He) with x=Q/T as
the independent variable, then fits cosmological parameters (log10(eta_10),
N_eff) to observed primordial abundances using nested sampling from
handley-lab/blackjax.

Usage:
    uv run python examples/bbn_estimation/main.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from solvers.rodas5P import solve as rodas5P_solve

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Physical constants (natural units, MeV)
# ---------------------------------------------------------------------------

Q = 1.293  # MeV, n-p mass difference
B_D = 2.225  # MeV, deuterium binding energy
M_N = 938.272  # MeV, nucleon mass
M_PL = 1.2209e22  # MeV, unreduced Planck mass
ZETA3 = 1.2020569  # Riemann zeta(3)
N_EFF_SM = 3.044  # Standard-model N_eff

# Natural-unit neutron lifetime: tau_n [s] -> tau_n / hbar [MeV^-1]
# hbar = 6.582119e-22 MeV*s
TAU_N_MEV = 879.4 / 6.582119e-22

# Rate cross-sections in MeV^-2
# SIGMA_NP = 4e-11 MeV^-2 from Bernstein et al. (1989) 4.55e-22 cm^3/s via hbar*c = 197.3 MeV*fm
# SIGMA_DD = 3.4e-5 MeV^-2 from NACRE (Angulo et al. 1999) at T ~ 0.1 MeV
SIGMA_NP = 4.0e-11  # MeV^-2, <sigma*v>_{np->D}
SIGMA_DD = 3.4e-5  # MeV^-2, effective <sigma*v>_{DD->4He}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def g_star(T, N_eff):
    """Effective relativistic dof; step through e+e- annihilation at m_e=0.511 MeV."""
    M_E = 0.511  # MeV
    g_sm = jnp.where(T > M_E, 10.75, 3.91)
    return g_sm + (7.0 / 4.0) * (N_eff - N_EFF_SM)


def hubble(x, N_eff):
    """H(T) in MeV with T = Q/x (Friedmann, radiation domination)."""
    T = Q / x
    g = g_star(T, N_eff)
    return jnp.sqrt(4.0 * jnp.pi**3 * g / 45.0) * T**2 / M_PL


def n_photon(T):
    """Photon number density in MeV^3: (2*zeta3/pi^2) T^3."""
    return 2.0 * ZETA3 / jnp.pi**2 * T**3


def weak_rate_np(x):
    """Total n->p rate [MeV]: Bernstein polynomial + free neutron decay."""
    return (255.0 / TAU_N_MEV) * (12.0 + 6.0 * x + x**2) / x**5 + 1.0 / TAU_N_MEV


def deuterium_eq_ratio(T, eta):
    """K_D(T,eta) = Y_d^eq / (Y_n * Y_p): Saha equation for D formation."""
    n_b = eta * n_photon(T)
    return n_b * (3.0 / 4.0) * (4.0 * jnp.pi / (M_N * T)) ** 1.5 * jnp.exp(B_D / T)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

# Independent variable: x = Q/T (dimensionless, runs from ~0.13 to ~129.3).
# Since dx/dt = H*x, the equation of motion becomes dY/dx = rates / (H*x).
# State: Y = [Y_n, Y_p, Y_d, Y_4He] (nucleon number fractions).
# Conservation: Y_n + Y_p + 2*Y_d + 4*Y_4He = 1.
# Parameters: [log10(eta_10), N_eff] where eta_10 = eta * 1e10.


def bbn_ode(y, x, params):
    log_eta10, N_eff = params[0], params[1]
    eta = 10.0 ** (log_eta10 - 10.0)
    T = Q / x

    H = hubble(x, N_eff)
    n_b = eta * n_photon(T)

    Gamma_np = weak_rate_np(x)
    Gamma_pn = Gamma_np * jnp.exp(-x)  # detailed balance

    K_D = deuterium_eq_ratio(T, eta)
    rate_np = n_b * SIGMA_NP * (y[0] * y[1] - y[2] / K_D)  # n+p<->D net rate
    rate_dd = n_b * SIGMA_DD * y[2] ** 2  # D+D->4He rate

    dYn = (Gamma_pn * y[1] - Gamma_np * y[0] - rate_np) / (H * x)
    dYp = (Gamma_np * y[0] - Gamma_pn * y[1] - rate_np) / (H * x)
    dYd = (rate_np - 2.0 * rate_dd) / (H * x)
    dYHe = rate_dd / (H * x)
    return jnp.array([dYn, dYp, dYd, dYHe])


# ---------------------------------------------------------------------------
# CUDA-device callbacks for the modax kernel solver
# ---------------------------------------------------------------------------
#
# ``rodas5P`` compiles its right-hand side, Jacobian and time-derivative
# with ``numba.cuda``, so these mirror ``bbn_ode`` above using ``math`` scalars
# and fixed-size tuples instead of ``jnp`` arrays.  ``tests`` in
# ``tests/test_examples.py`` checks all three against ``bbn_ode`` and its JAX
# derivatives, so the duplication cannot drift silently.  Device code cannot
# call a plain Python helper, so the shared term block is repeated inline.


def bbn_ode_device(y, x, p):
    """Device RHS: same equations as :func:`bbn_ode`, as a 4-tuple."""
    log_eta10 = p[0]
    N_eff = p[1]
    eta = 10.0 ** (log_eta10 - 10.0)
    T = Q / x

    g_sm = 10.75 if T > 0.511 else 3.91
    g = g_sm + (7.0 / 4.0) * (N_eff - N_EFF_SM)
    H = math.sqrt(4.0 * math.pi**3 * g / 45.0) * T * T / M_PL

    n_b = eta * (2.0 * ZETA3 / math.pi**2) * T**3

    Gamma_np = (255.0 / TAU_N_MEV) * (12.0 + 6.0 * x + x * x) / x**5 + 1.0 / TAU_N_MEV
    Gamma_pn = Gamma_np * math.exp(-x)

    K_D = n_b * (3.0 / 4.0) * (4.0 * math.pi / (M_N * T)) ** 1.5 * math.exp(B_D / T)

    rate_np = n_b * SIGMA_NP * (y[0] * y[1] - y[2] / K_D)
    rate_dd = n_b * SIGMA_DD * y[2] * y[2]
    denom = H * x
    return (
        (Gamma_pn * y[1] - Gamma_np * y[0] - rate_np) / denom,
        (Gamma_np * y[0] - Gamma_pn * y[1] - rate_np) / denom,
        (rate_np - 2.0 * rate_dd) / denom,
        rate_dd / denom,
    )


def bbn_jac_device(y, x, p):
    """Analytic df/dy for the four-species network, as a 4x4 nested tuple.

    Only ``rate_np`` and ``rate_dd`` carry state dependence; with
    ``a = n_b*SIGMA_NP``, ``b = n_b*SIGMA_DD`` and ``c = a/K_D``:

        d(rate_np)/dy = ( a*y1,  a*y0, -c, 0 )
        d(rate_dd)/dy = (    0,     0,  2*b*y2, 0 )
    """
    log_eta10 = p[0]
    N_eff = p[1]
    eta = 10.0 ** (log_eta10 - 10.0)
    T = Q / x

    g_sm = 10.75 if T > 0.511 else 3.91
    g = g_sm + (7.0 / 4.0) * (N_eff - N_EFF_SM)
    H = math.sqrt(4.0 * math.pi**3 * g / 45.0) * T * T / M_PL

    n_b = eta * (2.0 * ZETA3 / math.pi**2) * T**3

    Gamma_np = (255.0 / TAU_N_MEV) * (12.0 + 6.0 * x + x * x) / x**5 + 1.0 / TAU_N_MEV
    Gamma_pn = Gamma_np * math.exp(-x)

    K_D = n_b * (3.0 / 4.0) * (4.0 * math.pi / (M_N * T)) ** 1.5 * math.exp(B_D / T)

    denom = H * x
    a = n_b * SIGMA_NP
    b = n_b * SIGMA_DD
    c = a / K_D
    return (
        ((-Gamma_np - a * y[1]) / denom, (Gamma_pn - a * y[0]) / denom, c / denom, 0.0),
        ((Gamma_np - a * y[1]) / denom, (-Gamma_pn - a * y[0]) / denom, c / denom, 0.0),
        (
            a * y[1] / denom,
            a * y[0] / denom,
            (-c - 4.0 * b * y[2]) / denom,
            0.0,
        ),
        (0.0, 0.0, 2.0 * b * y[2] / denom, 0.0),
    )


def bbn_time_jac_device(y, x, p):
    """Analytic df/dx, needed for Rodas5P to hold fifth order here.

    Every coefficient (H, n_b, the weak rates and K_D) depends on x, so the
    network is strongly non-autonomous.  With ``T = Q/x`` and g locally
    constant (it steps only at ``T = m_e``), ``dT/dx = -T/x`` gives

        dH/dx = -2H/x,   d(n_b)/dx = -3 n_b/x,   d(H*x)/dx = -H,
        d(K_D)/dx = K_D * (B_D/T - 1.5) / x.
    """
    log_eta10 = p[0]
    N_eff = p[1]
    eta = 10.0 ** (log_eta10 - 10.0)
    T = Q / x

    g_sm = 10.75 if T > 0.511 else 3.91
    g = g_sm + (7.0 / 4.0) * (N_eff - N_EFF_SM)
    H = math.sqrt(4.0 * math.pi**3 * g / 45.0) * T * T / M_PL

    n_b = eta * (2.0 * ZETA3 / math.pi**2) * T**3

    Gamma_np = (255.0 / TAU_N_MEV) * (12.0 + 6.0 * x + x * x) / x**5 + 1.0 / TAU_N_MEV
    Gamma_pn = Gamma_np * math.exp(-x)

    K_D = n_b * (3.0 / 4.0) * (4.0 * math.pi / (M_N * T)) ** 1.5 * math.exp(B_D / T)

    rate_np = n_b * SIGMA_NP * (y[0] * y[1] - y[2] / K_D)
    rate_dd = n_b * SIGMA_DD * y[2] * y[2]
    denom = H * x

    # d(Gamma_np)/dx from the Bernstein polynomial; the constant decay term drops.
    A = 255.0 / TAU_N_MEV
    dGamma_np = A * ((6.0 + 2.0 * x) / x**5 - 5.0 * (12.0 + 6.0 * x + x * x) / x**6)
    dGamma_pn = math.exp(-x) * (dGamma_np - Gamma_np)

    d_rate_np = SIGMA_NP * (
        -3.0 * n_b / x * (y[0] * y[1] - y[2] / K_D)
        + n_b * y[2] * (B_D / T - 1.5) / (x * K_D)
    )
    d_rate_dd = -3.0 * rate_dd / x

    dN0 = dGamma_pn * y[1] - dGamma_np * y[0] - d_rate_np
    dN1 = dGamma_np * y[0] - dGamma_pn * y[1] - d_rate_np
    dN2 = d_rate_np - 2.0 * d_rate_dd
    dN3 = d_rate_dd

    # f_i = N_i / (H*x) and d(H*x)/dx = -H, so df_i/dx = dN_i/(H*x) + f_i/x.
    f0 = (Gamma_pn * y[1] - Gamma_np * y[0] - rate_np) / denom
    f1 = (Gamma_np * y[0] - Gamma_pn * y[1] - rate_np) / denom
    f2 = (rate_np - 2.0 * rate_dd) / denom
    f3 = rate_dd / denom
    return (
        dN0 / denom + f0 / x,
        dN1 / denom + f1 / x,
        dN2 / denom + f2 / x,
        dN3 / denom + f3 / x,
    )


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

X_SPAN = jnp.array([Q / 10.0, Q / 0.01])  # x: 0.1293 -> 129.3 (T = 0.01 MeV)
X_SAVE = X_SPAN  # save at start and end only

SOLVER_RTOL = 1e-3
SOLVER_ATOL = 1e-7
SOLVER_FIRST_STEP = 0.1
SOLVER_MAX_STEPS = 256


def initial_conditions():
    """Weak-equilibrium initial state at T = 10 MeV (x = Q/10 ~ 0.13)."""
    x0 = X_SPAN[0]
    yn0 = jnp.exp(-x0) / (1.0 + jnp.exp(-x0))
    yp0 = 1.0 / (1.0 + jnp.exp(-x0))
    return jnp.array([yn0, yp0, 1e-20, 0.0])


def predict_abundances(params):
    """Integrate BBN network and return [Y_P, D/H] for given params."""
    sol = rodas5P_solve(
        bbn_ode_device,
        bbn_jac_device,
        initial_conditions(),
        X_SAVE,
        params,
        time_jac_fn=bbn_time_jac_device,
        lu_precision="fp32",
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
        first_step=SOLVER_FIRST_STEP,
        max_steps=SOLVER_MAX_STEPS,
    )
    _, Yp, Yd, YHe = sol[0, -1]
    Y_P = 4.0 * YHe  # helium mass fraction
    D_H = Yd / Yp  # D/H number ratio
    return jnp.array([Y_P, D_H])


# ---------------------------------------------------------------------------
# Solver backends and benchmark
# ---------------------------------------------------------------------------
#
# The science of the example uses the GPU-batched modax Rodas5P solver.  For a
# like-for-like timing comparison we also expose two reference backends with the
# identical four-species stiff network:
#   * "scipy"   -- serial CPU integration with scipy.solve_ivp (LSODA), the
#                  no-GPU baseline used by codes such as the original ECHO21.
#   * "diffrax" -- GPU integration with plain Diffrax Kvaerno5 (jax.vmap).
# The "chi^2 grid" use case of the docstring batches N independent (eta, N_eff)
# universes into a single ensemble solve.


def make_solver(backend):
    """Return a uniform ``solve(ode_fn, y0, t_span, params)`` for a backend."""
    if backend == "modax":
        # The kernel solver takes its own CUDA-device callbacks rather than the
        # traced ``bbn_ode``; ``f`` is ignored so every backend shares one
        # ``solve(f, y0, ts, p)`` signature.
        return lambda f, y0, ts, p: rodas5P_solve(
            bbn_ode_device,
            bbn_jac_device,
            y0,
            ts,
            p,
            time_jac_fn=bbn_time_jac_device,
            lu_precision="fp32",
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            first_step=SOLVER_FIRST_STEP,
            max_steps=SOLVER_MAX_STEPS,
        )
    if backend == "diffrax":
        from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_solve

        return lambda f, y0, ts, p: diffrax_solve(
            f,
            y0,
            ts,
            p,
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            first_step=SOLVER_FIRST_STEP,
            max_steps=8192,
        )
    if backend == "scipy":
        from reference.solvers.python.scipy_solve_ivp import solve as scipy_solve

        # LSODA with an automatic initial step is what serial codes such as
        # ECHO21 use; an imposed first_step of 0.1 destabilises it here.
        return lambda f, y0, ts, p: scipy_solve(
            f,
            y0,
            ts,
            p,
            method="LSODA",
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            first_step=None,
        )
    raise ValueError(f"unknown backend: {backend}")


def sample_grid_params(n):
    """A near-square (eta_10, N_eff) grid covering the prior box, flattened."""
    side = int(np.ceil(np.sqrt(n)))
    eta = np.linspace(LO[0], HI[0], side)
    neff = np.linspace(LO[1], HI[1], side)
    ee, nn = np.meshgrid(eta, neff)
    grid = np.column_stack([ee.ravel(), nn.ravel()])[:n]
    return jnp.asarray(grid, dtype=jnp.float64)


def time_solve(fn, repeats):
    """Return (mean seconds excluding compile, result) over ``repeats`` runs."""
    result = fn()
    jax.block_until_ready(result)
    t0 = time.perf_counter()
    for _ in range(repeats):
        result = fn()
        jax.block_until_ready(result)
    return (time.perf_counter() - t0) / repeats, result


def run_benchmark(n, backends, repeats):
    params = sample_grid_params(n)
    y0 = initial_conditions()
    print(
        f"BBN forward-solve benchmark: N = {n:,} stiff 4-species universes\n",
        flush=True,
    )
    print(
        f"{'backend':>10}  {'wall (s)':>10}  {'per solve':>12}  Y_P(eta~6,Neff~3)",
        flush=True,
    )
    print("-" * 60, flush=True)
    for backend in backends:
        solve_fn = make_solver(backend)
        run = lambda sf=solve_fn: sf(bbn_ode, y0, X_SAVE, params)
        try:
            secs, sol = time_solve(run, repeats)
        except Exception as exc:  # noqa: BLE001
            print(f"{backend:>10}  FAILED: {exc}", flush=True)
            continue
        sol = np.asarray(sol)
        # mid-grid sample for a sanity check on agreement across backends
        yp_mid = 4.0 * sol[n // 2, -1, 3]
        per = secs / n
        print(
            f"{backend:>10}  {secs:10.3f}  {per * 1e3:9.4f} ms  {yp_mid:.5f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Observational data and statistical model
# ---------------------------------------------------------------------------

# Simplified-model predictions calibrated to CMB Planck 2018 parameters
# (eta_10 ~ 6.1, N_eff = 3.044).  The 4-species network (n+p<->D, D+D->He4)
# gives Y_P ~ 0.106 rather than the real-network value 0.245, because the full
# chain (He3, T, He4 via multiple paths) is absent.  Uncertainties are scaled
# to be ~4% relative, matching the real observational precision.
Y_P_OBS = 0.106
SIGMA_YP = 0.004

DH_OBS = 4.2e-6
SIGMA_DH = 0.3e-6

# Uniform box prior: log10(eta_10) in [0.5, 1.0], N_eff in [2.0, 4.0]
LO = jnp.array([0.5, 2.0])
HI = jnp.array([1.0, 4.0])


def log_prior(theta):
    return jnp.where(jnp.all((theta >= LO) & (theta <= HI)), 0.0, -jnp.inf)


def log_likelihood(theta):
    # Select rather than branch. The sampler always evaluates this under
    # jax.vmap, where lax.cond with a batched predicate is rewritten into a
    # select that runs both branches anyway -- so this costs nothing extra --
    # and the solver's custom_vmap rule cannot be traced inside a batched
    # lax.cond (JAX asserts that a custom_vmap's captured constants are
    # unbatched, which cond's batching rule violates).
    in_prior = jnp.all((theta >= LO) & (theta <= HI))
    preds = predict_abundances(theta)
    chi2 = ((preds[0] - Y_P_OBS) / SIGMA_YP) ** 2 + (
        (preds[1] - DH_OBS) / SIGMA_DH
    ) ** 2
    ll = -0.5 * chi2
    return jnp.where(in_prior & jnp.isfinite(ll), ll, -jnp.inf)


# ---------------------------------------------------------------------------
# Nested sampling
# ---------------------------------------------------------------------------

N_LIVE = 128
N_INNER = 3  # slice steps per dead-particle replacement
NUM_DEL = 64  # replacement chains run in parallel inside BlackJAX NSS (vmap batch)
TOL_LOGZ = 5.0  # convergence: stop when logZ - logZ_live > tol (remaining live contribution < exp(-tol) of accumulated)
NS_CHUNK_SIZE = 4
STATUS_EVERY_CHUNKS = 2
MAX_NS_STEPS = 5 * N_LIVE
SLICE_MAX_STEPS = 2
SLICE_MAX_SHRINKAGE = 5


def run_nested_sampling():
    import blackjax.ns.nss as nss
    from blackjax.ns import utils
    from blackjax.ns.base import NSInfo

    key = jax.random.key(42)
    key, init_key, pos_key = jax.random.split(key, 3)
    positions = jax.random.uniform(pos_key, (N_LIVE, 2)) * (HI - LO) + LO

    kernel = nss.as_top_level_api(
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
        num_inner_steps=N_INNER,
        num_delete=NUM_DEL,
        max_steps=SLICE_MAX_STEPS,
        max_shrinkage=SLICE_MAX_SHRINKAGE,
    )
    state = kernel.init(positions, rng_key=init_key)
    step = kernel.step

    def _strip_update_info(info):
        return NSInfo(info.particles, None)

    def _flatten_dead_info(info):
        particles = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:]),
            info.particles,
        )
        return NSInfo(particles, None)

    def _chunk_body(carry, _):
        key, state = carry
        key, subkey = jax.random.split(key)
        state, info = step(subkey, state)
        return (key, state), _strip_update_info(info)

    @jax.jit
    def run_chunk(key, state):
        return jax.lax.scan(_chunk_body, (key, state), None, length=NS_CHUNK_SIZE)

    dead_info_chunks = []
    max_chunks = (MAX_NS_STEPS + NS_CHUNK_SIZE - 1) // NS_CHUNK_SIZE
    steps_done = 0
    for chunk in range(max_chunks):
        (key, state), chunk_info = run_chunk(key, state)
        dead_info_chunks.append(_flatten_dead_info(chunk_info))
        steps_done = min((chunk + 1) * NS_CHUNK_SIZE, MAX_NS_STEPS)
        logZ = state.integrator.logZ
        logZ_live = state.integrator.logZ_live
        if chunk % STATUS_EVERY_CHUNKS == 0:
            print(
                f"  step {steps_done:5d}  "
                f"logZ={float(logZ):.3f}  logZ_live={float(logZ_live):.3f}",
                flush=True,
            )
        if jnp.isfinite(logZ) and (float(logZ) - float(logZ_live)) > TOL_LOGZ:
            print(f"  Converged by step {steps_done}", flush=True)
            break

    # Combine dead particles + final live points into one NSInfo object
    final_info = utils.finalise(state, dead_info_chunks, update_info=False)
    dead_positions = final_info.particles.position  # shape (N_total, 2)

    # Compute posterior log-weights; shape=1 gives a single MC estimate
    key, w_key = jax.random.split(key)
    log_w = utils.log_weights(w_key, final_info, shape=1)  # (N_total, 1)
    log_w = log_w[:, 0]  # (N_total,)
    w = jnp.exp(log_w - jnp.max(log_w))
    w = w / w.sum()
    w_np = np.array(w)

    eta10_samples = 10.0 ** np.array(dead_positions[:, 0])
    neff_samples = np.array(dead_positions[:, 1])

    eta10_mean = float(np.average(eta10_samples, weights=w_np))
    eta10_std = float(
        np.sqrt(np.average((eta10_samples - eta10_mean) ** 2, weights=w_np))
    )
    neff_mean = float(np.average(neff_samples, weights=w_np))
    neff_std = float(np.sqrt(np.average((neff_samples - neff_mean) ** 2, weights=w_np)))

    print(f"\nlog Z = {float(state.integrator.logZ):.3f}")
    print(f"eta_10 = {eta10_mean:.3f} +/- {eta10_std:.3f}")
    print(f"N_eff  = {neff_mean:.3f} +/- {neff_std:.3f}")

    _plot_posterior(dead_positions, w_np, eta10_samples, neff_samples)


def _plot_posterior(dead_positions, w, eta10_samples, neff_samples):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 2D scatter coloured by weight
    sc = axes[0].scatter(
        eta10_samples,
        neff_samples,
        c=w,
        s=2,
        cmap="viridis",
        alpha=0.6,
    )
    plt.colorbar(sc, ax=axes[0], label="weight")
    axes[0].set_xlabel(r"$\eta_{10}$")
    axes[0].set_ylabel(r"$N_\mathrm{eff}$")
    axes[0].set_title("Posterior samples")

    # eta_10 marginal
    axes[1].hist(eta10_samples, weights=w, bins=40, color="steelblue", density=True)
    axes[1].set_xlabel(r"$\eta_{10}$")
    axes[1].set_ylabel("density")
    axes[1].set_title(r"Marginal $\eta_{10}$")

    # N_eff marginal
    axes[2].hist(neff_samples, weights=w, bins=40, color="coral", density=True)
    axes[2].set_xlabel(r"$N_\mathrm{eff}$")
    axes[2].set_ylabel("density")
    axes[2].set_title(r"Marginal $N_\mathrm{eff}$")

    fig.tight_layout()
    out = Path(__file__).parent / "posterior.png"
    fig.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="time the batched forward solve across solver backends",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["modax", "diffrax", "scipy"],
        choices=["modax", "diffrax", "scipy"],
    )
    parser.add_argument("--n", type=int, default=10_000, help="ensemble size")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.n, args.backends, args.repeats)
        return

    print("Integrating BBN network (4 species, x = Q/T)", flush=True)
    print("Running nested sampling ...", flush=True)
    run_nested_sampling()


if __name__ == "__main__":
    main()
