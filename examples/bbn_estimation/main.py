"""Bayesian parameter estimation from primordial light-element abundances.

A four-species Big Bang Nucleosynthesis network (n, p, D, 4He) is integrated
with the Rodas5P kernel solver, with x = Q/T as the independent variable, and
the baryon-to-photon ratio log10(eta_10) and N_eff are fitted to the observed
abundances by nested sampling (handley-lab/blackjax). Every likelihood
evaluation is an independent stiff universe, so the sampler's population is
one batched ensemble solve. The physics, the statistical model and the
benchmark are laid out in README.md next to this file.

Usage:
    uv run python examples/bbn_estimation/main.py [--benchmark]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import Backends, parse_args, run_benchmark
from examples.dual_backend import build_fn, build_rhs
from solvers.rodas5P import solve as rodas5P_solve

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Physical constants (natural units, MeV)
# ---------------------------------------------------------------------------

Q = 1.293  # MeV, n-p mass difference
B_D = 2.225  # MeV, deuterium binding energy
M_N = 938.272  # MeV, nucleon mass
M_E = 0.511  # MeV, electron mass (where g_star steps)
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
#
# The solvers want the network in two forms: ``rodas5P`` compiles its
# right-hand side with ``numba_cuda_mlir`` and differentiates it with Enzyme,
# while the Diffrax reference backend traces ``jnp`` arrays.  Each piece below
# is therefore written once, as a factory over the names the two spell
# differently, and built in both forms by ``examples/dual_backend.py``.  The
# ``.device`` member of each is a CUDA device function; ``.jax`` is the traced
# one; ``.host`` is the device arithmetic in plain Python, which is what
# ``tests/test_examples.py`` compares against ``.jax`` without needing a GPU.


def _make_g_star():
    def g_star(T, N_eff):
        """Effective relativistic dof; step through e+e- annihilation at m_e."""
        # Branchless step: the comparison is a 0/1 factor, which a traced
        # value and a device scalar both multiply by.  ``jnp.where`` would not
        # compile on the device and ``if`` would not trace.
        g_sm = 3.91 + (10.75 - 3.91) * (T > M_E)
        return g_sm + (7.0 / 4.0) * (N_eff - N_EFF_SM)

    return g_star


def _make_hubble(*, sqrt, g_star):
    def hubble(x, N_eff):
        """H(T) in MeV with T = Q/x (Friedmann, radiation domination)."""
        T = Q / x
        return sqrt(4.0 * math.pi**3 * g_star(T, N_eff) / 45.0) * T**2 / M_PL

    return hubble


def _make_n_photon():
    def n_photon(T):
        """Photon number density in MeV^3: (2*zeta3/pi^2) T^3."""
        return 2.0 * ZETA3 / math.pi**2 * T**3

    return n_photon


def _make_weak_rate_np():
    def weak_rate_np(x):
        """Total n->p rate [MeV]: Bernstein polynomial + free neutron decay."""
        return (255.0 / TAU_N_MEV) * (12.0 + 6.0 * x + x**2) / x**5 + 1.0 / TAU_N_MEV

    return weak_rate_np


def _make_deuterium_eq_ratio(*, exp, n_photon):
    def deuterium_eq_ratio(T, eta):
        """K_D(T,eta) = Y_d^eq / (Y_n * Y_p): Saha equation for D formation."""
        n_b = eta * n_photon(T)
        return n_b * (3.0 / 4.0) * (4.0 * math.pi / (M_N * T)) ** 1.5 * exp(B_D / T)

    return deuterium_eq_ratio


G_STAR = build_fn(_make_g_star)
N_PHOTON = build_fn(_make_n_photon)
WEAK_RATE_NP = build_fn(_make_weak_rate_np)
HUBBLE = build_fn(_make_hubble, g_star=G_STAR)
DEUTERIUM_EQ_RATIO = build_fn(_make_deuterium_eq_ratio, n_photon=N_PHOTON)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

# Independent variable: x = Q/T (dimensionless, runs from ~0.13 to ~129.3).
# Since dx/dt = H*x, the equation of motion becomes dY/dx = rates / (H*x).
# State: Y = [Y_n, Y_p, Y_d, Y_4He] (nucleon number fractions).
# Conservation: Y_n + Y_p + 2*Y_d + 4*Y_4He = 1.
# Parameters: [log10(eta_10), N_eff] where eta_10 = eta * 1e10.


def _make_bbn_ode(*, exp, hubble, n_photon, weak_rate_np, deuterium_eq_ratio):
    def bbn_ode(y, x, params):
        log_eta10, N_eff = params[0], params[1]
        eta = 10.0 ** (log_eta10 - 10.0)
        T = Q / x

        H = hubble(x, N_eff)
        n_b = eta * n_photon(T)

        Gamma_np = weak_rate_np(x)
        Gamma_pn = Gamma_np * exp(-x)  # detailed balance

        K_D = deuterium_eq_ratio(T, eta)
        rate_np = n_b * SIGMA_NP * (y[0] * y[1] - y[2] / K_D)  # n+p<->D net rate
        rate_dd = n_b * SIGMA_DD * y[2] ** 2  # D+D->4He rate

        denom = H * x
        return (
            (Gamma_pn * y[1] - Gamma_np * y[0] - rate_np) / denom,
            (Gamma_np * y[0] - Gamma_pn * y[1] - rate_np) / denom,
            (rate_np - 2.0 * rate_dd) / denom,
            rate_dd / denom,
        )

    return bbn_ode


# dY/dx: ``.device`` is the 4-tuple for the menodax kernel, ``.jax`` the array
# for the Diffrax and scipy backends.
BBN_ODE = build_rhs(
    _make_bbn_ode,
    hubble=HUBBLE,
    n_photon=N_PHOTON,
    weak_rate_np=WEAK_RATE_NP,
    deuterium_eq_ratio=DEUTERIUM_EQ_RATIO,
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

# The science uses the GPU-batched menodax Rodas5P solver. For a like-for-like
# timing, ``--benchmark`` also runs the identical four-species stiff network
# on Diffrax Kvaerno5 (GPU, jax.vmap) and on serial scipy.solve_ivp LSODA, the
# no-GPU baseline used by codes such as the original ECHO21. LSODA runs with an
# automatic initial step, as those serial codes do; an imposed first_step of
# 0.1 destabilises it here.
MENODAX_KWARGS = dict(
    lu_precision="fp32",
    rtol=SOLVER_RTOL,
    atol=SOLVER_ATOL,
    first_step=SOLVER_FIRST_STEP,
    max_steps=SOLVER_MAX_STEPS,
)
BACKENDS = Backends(
    menodax_solve=rodas5P_solve,
    menodax_kwargs=MENODAX_KWARGS,
    diffrax_method="kvaerno5",
    diffrax_kwargs=dict(
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
        first_step=SOLVER_FIRST_STEP,
        max_steps=8192,
    ),
    scipy_kwargs=dict(
        method="LSODA", rtol=SOLVER_RTOL, atol=SOLVER_ATOL, first_step=None
    ),
)


def initial_conditions():
    """Weak-equilibrium initial state at T = 10 MeV (x = Q/10 ~ 0.13)."""
    x0 = X_SPAN[0]
    yn0 = jnp.exp(-x0) / (1.0 + jnp.exp(-x0))
    yp0 = 1.0 / (1.0 + jnp.exp(-x0))
    return jnp.array([yn0, yp0, 1e-20, 0.0])


def predict_abundances(params):
    """Integrate BBN network and return [Y_P, D/H] for given params."""
    sol = rodas5P_solve(
        BBN_ODE.device, initial_conditions(), X_SAVE, params, **MENODAX_KWARGS
    )
    _, Yp, Yd, YHe = sol[0, -1]
    Y_P = 4.0 * YHe  # helium mass fraction
    D_H = Yd / Yp  # D/H number ratio
    return jnp.array([Y_P, D_H])


# ---------------------------------------------------------------------------
# Benchmark: the chi^2-grid use case, N independent (eta, N_eff) universes
# ---------------------------------------------------------------------------


def sample_grid_params(n):
    """A near-square (eta_10, N_eff) grid covering the prior box, flattened."""
    side = int(np.ceil(np.sqrt(n)))
    eta = np.linspace(LO[0], HI[0], side)
    neff = np.linspace(LO[1], HI[1], side)
    ee, nn = np.meshgrid(eta, neff)
    grid = np.column_stack([ee.ravel(), nn.ravel()])[:n]
    return jnp.asarray(grid, dtype=jnp.float64)


def benchmark(n, backends, repeats):
    params = sample_grid_params(n)
    run_benchmark(
        BACKENDS,
        BBN_ODE,
        initial_conditions(),
        X_SAVE,
        params,
        backend_names=backends,
        repeats=repeats,
        title=f"BBN forward-solve benchmark: N = {n:,} stiff 4-species universes",
        column="Y_P(eta~6,Neff~3)",
        # mid-grid sample for a sanity check on agreement across backends
        metric=lambda sol: f"{4.0 * sol[n // 2, -1, 3]:.5f}",
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

    _plot_posterior(w_np, eta10_samples, neff_samples)


def _plot_posterior(w, eta10_samples, neff_samples):
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
    args = parse_args(__doc__, n_default=10_000, n_help="ensemble size")
    if args.benchmark:
        benchmark(args.n, args.backends, args.repeats)
        return

    print("Integrating BBN network (4 species, x = Q/T)", flush=True)
    print("Running nested sampling ...", flush=True)
    run_nested_sampling()


if __name__ == "__main__":
    main()
