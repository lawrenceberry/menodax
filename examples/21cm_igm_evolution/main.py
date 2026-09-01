"""Monte Carlo uncertainty quantification for a global 21-cm IGM surrogate.

The sky-averaged 21-cm brightness temperature tracks the thermal, ionization,
and spin history of neutral hydrogen.  During Cosmic Dawn and the Epoch of
Reionization, the relevant source functions are dominated by poorly known
properties of the first galaxies: star-formation efficiency, X-ray luminosity,
and the minimum halo virial temperature needed for star formation.

This example draws many independent astrophysical parameter sets and integrates
their uncoupled IGM histories in one batched Rodas5P solve.  The model is a
lightweight, pedagogical surrogate inspired by the standard global-signal
equations and by ECHO21.  It is intended to demonstrate massively batched
uncertainty quantification with modax, not to replace precision tools such as
ECHO21 or 21cmFAST.

References:
    S. Mittal et al., "ECHO21", arXiv:2503.11762 / RASTI 5 (2026) rzag001.
    S. Furlanetto, S. P. Oh, F. Briggs, Phys. Rep. 433, 181 (2006).
    J. Pritchard, A. Loeb, Rep. Prog. Phys. 75, 086901 (2012).
    S. Wouthuysen, AJ 57, 31 (1952); G. Field, Proc. IRE 46, 240 (1958).

Usage:
    uv run python examples/21cm_igm_evolution/main.py
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

Z_INITIAL = 100.0
Z_FINAL = 5.0
N_SAMPLES = 50_000
N_SAVE = 180
RANDOM_SEED = 21

NU_21_MHZ = 1420.40575177
T_CMB0 = 2.7255
H0_KM_S_MPC = 67.4
OMEGA_M = 0.315
OMEGA_L = 0.685
OMEGA_B = 0.049
HUBBLE_LITTLE_H = H0_KM_S_MPC / 100.0
X_HE = 0.079

MPC_KM = 3.0856775814913673e19
H0_S = H0_KM_S_MPC / MPC_KM
N_H0_CM3 = 1.88e-7 * OMEGA_B * HUBBLE_LITTLE_H**2 / 0.022

LOGIT_EPS = 1.0e-8

PARAM_LO = jnp.array([-3.0, -2.0, 3.5], dtype=jnp.float64)
PARAM_HI = jnp.array([-0.5, 2.0, 5.5], dtype=jnp.float64)

SOLVER_RTOL = 2.0e-4
SOLVER_ATOL = 1.0e-7
SOLVER_FIRST_STEP = 2.0e-3
SOLVER_MAX_STEPS = 4096


def redshift_from_u(u):
    """Map integration coordinate u to redshift."""
    return (1.0 + Z_INITIAL) * jnp.exp(-u) - 1.0


def u_from_redshift(z):
    """Map redshift to integration coordinate u."""
    return jnp.log((1.0 + Z_INITIAL) / (1.0 + z))


def hubble_s(z):
    """Flat-LambdaCDM Hubble rate in s^-1."""
    return H0_S * jnp.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def cmb_temperature(z):
    return T_CMB0 * (1.0 + z)


def _clip_unit_interval(x):
    return jnp.clip(x, LOGIT_EPS, 1.0 - LOGIT_EPS)


def logit(x):
    x = _clip_unit_interval(x)
    return jnp.log(x / (1.0 - x))


def sigmoid_from_logit(x):
    return jax.nn.sigmoid(jnp.clip(x, -40.0, 40.0))


def source_history(z, params):
    """Smooth dimensionless star-formation history surrogate.

    Higher star-formation efficiency raises the amplitude.  Higher virial
    temperature delays the source turn-on by requiring rarer, more massive
    haloes before star formation begins.
    """
    log10_f_star, _, log10_tvir = params
    f_star = 10.0**log10_f_star
    turn_on_z = 31.0 - 7.0 * (log10_tvir - 4.0)
    turn_on = jax.nn.sigmoid((turn_on_z - z) / 2.0)
    low_z_decline = jax.nn.sigmoid((z - 5.8) / 0.7)
    growth = ((1.0 + z) / 20.0) ** -1.7
    return 10.0 * f_star * turn_on * low_z_decline * growth


def couplings(z, T_k, params):
    """Return collisional and Ly-alpha coupling coefficients."""
    source = source_history(z, params)
    x_coll = 0.12 * ((1.0 + z) / 30.0) ** 2 * jnp.sqrt(jnp.maximum(T_k, 1.0) / 10.0)
    x_alpha = 180.0 * source
    return x_coll, x_alpha


def spin_temperature(z, T_k, params):
    """Field equilibrium spin temperature with T_color ~= T_k."""
    T_gamma = cmb_temperature(z)
    x_coll, x_alpha = couplings(z, T_k, params)
    x_tot = x_coll + x_alpha
    inv_Ts = (1.0 / T_gamma + x_tot / T_k) / (1.0 + x_tot)
    return 1.0 / inv_Ts


def brightness_temperature_mk(z, T_k, Q, params):
    """Sky-averaged 21-cm brightness temperature relative to the CMB."""
    T_s = spin_temperature(z, T_k, params)
    x_hi = 1.0 - Q
    cosmo = jnp.sqrt((0.15 / (OMEGA_M * HUBBLE_LITTLE_H**2)) * ((1.0 + z) / 10.0))
    baryon = OMEGA_B * HUBBLE_LITTLE_H**2 / 0.023
    return 27.0 * x_hi * baryon * cosmo * (1.0 - cmb_temperature(z) / T_s)


def igm_ode(y, u, params):
    """Thermal and ionization-history surrogate in d/du form."""
    z = redshift_from_u(u)
    H = hubble_s(z)
    T_gamma = cmb_temperature(z)

    log_Tk, logit_xe, logit_Q = y
    T_k = jnp.exp(jnp.clip(log_Tk, jnp.log(0.5), jnp.log(5.0e4)))
    x_e = sigmoid_from_logit(logit_xe)
    Q = sigmoid_from_logit(logit_Q)

    log10_f_star, log10_f_X, _ = params
    f_X = 10.0**log10_f_X
    source = source_history(z, params)

    compton_rate = 8.0e-20 * (1.0 + z) ** 4 * x_e / (1.0 + X_HE + x_e)
    xray_heating = 3.5e-16 * f_X * source
    lya_heating = 1.2e-17 * (10.0**log10_f_star / 0.01) * source

    dTdt = -2.0 * H * T_k + compton_rate * (T_gamma - T_k) + xray_heating + lya_heating

    n_h = N_H0_CM3 * (1.0 + z) ** 3
    alpha_b = 2.6e-13 * (jnp.maximum(T_k, 10.0) / 1.0e4) ** -0.7
    xray_ionization = 1.8e-17 * f_X * source * (1.0 - x_e)
    dxedt = -alpha_b * n_h * x_e**2 + xray_ionization

    ionizing_source = 1.1e-16 * source * (10.0**log10_f_star / 0.01) * (1.0 - Q)
    recombinations = 2.0e-17 * ((1.0 + z) / 8.0) ** 3 * Q**2
    dQdt = ionizing_source - recombinations

    dlogT_du = dTdt / (H * T_k)
    dlogit_xe_du = dxedt / (H * jnp.maximum(x_e * (1.0 - x_e), LOGIT_EPS))
    dlogit_Q_du = dQdt / (H * jnp.maximum(Q * (1.0 - Q), LOGIT_EPS))
    return jnp.array([dlogT_du, dlogit_xe_du, dlogit_Q_du])


# ---------------------------------------------------------------------------
# CUDA-device callbacks for the modax kernel solver
# ---------------------------------------------------------------------------
#
# ``rodas5P`` compiles its right-hand side, Jacobian and time derivative
# with ``numba.cuda``, so the block below mirrors ``igm_ode`` using ``math``
# scalars and fixed-size tuples instead of ``jnp`` arrays.  The clip and
# maximum guards are reproduced exactly, including their zero derivatives
# outside the active range.  Device code cannot call a plain Python helper, so
# the shared background/state block is repeated inline in each callback;
# ``tests/test_examples.py`` checks all three against ``igm_ode`` and its JAX
# derivatives so the duplication cannot drift silently.
#
# Background terms depend on u only through z = (1+Z_INITIAL) exp(-u) - 1, so
# dz/du = -(1+z) and every (1+z)^n factor contributes -n times itself per unit
# u.  The state enters only through the decoded (T_k, x_e, Q).

_LOG_TK_MIN = math.log(0.5)
_LOG_TK_MAX = math.log(5.0e4)


def igm_ode_device(y, u, p):
    """Device RHS: same equations as :func:`igm_ode`, as a 3-tuple."""
    log10_f_star = p[0]
    log10_f_X = p[1]
    log10_tvir = p[2]

    z = (1.0 + Z_INITIAL) * math.exp(-u) - 1.0
    opz = 1.0 + z
    H = H0_S * math.sqrt(OMEGA_M * opz**3 + OMEGA_L)
    T_gamma = T_CMB0 * opz

    f_star = 10.0**log10_f_star
    f_X = 10.0**log10_f_X
    turn_on_z = 31.0 - 7.0 * (log10_tvir - 4.0)
    turn_on = 1.0 / (1.0 + math.exp(-(turn_on_z - z) / 2.0))
    decline = 1.0 / (1.0 + math.exp(-(z - 5.8) / 0.7))
    growth = (opz / 20.0) ** -1.7
    source = 10.0 * f_star * turn_on * decline * growth

    log_Tk = y[0]
    clipped = min(max(log_Tk, _LOG_TK_MIN), _LOG_TK_MAX)
    T_k = math.exp(clipped)
    x_e = 1.0 / (1.0 + math.exp(-min(max(y[1], -40.0), 40.0)))
    Q = 1.0 / (1.0 + math.exp(-min(max(y[2], -40.0), 40.0)))

    compton_rate = 8.0e-20 * opz**4 * x_e / (1.0 + X_HE + x_e)
    dTdt = (
        -2.0 * H * T_k
        + compton_rate * (T_gamma - T_k)
        + 3.5e-16 * f_X * source
        + 1.2e-17 * (f_star / 0.01) * source
    )

    n_h = N_H0_CM3 * opz**3
    alpha_b = 2.6e-13 * (max(T_k, 10.0) / 1.0e4) ** -0.7
    dxedt = -alpha_b * n_h * x_e * x_e + 1.8e-17 * f_X * source * (1.0 - x_e)

    dQdt = (
        1.1e-16 * source * (f_star / 0.01) * (1.0 - Q)
        - 2.0e-17 * (opz / 8.0) ** 3 * Q * Q
    )

    return (
        dTdt / (H * T_k),
        dxedt / (H * max(x_e * (1.0 - x_e), LOGIT_EPS)),
        dQdt / (H * max(Q * (1.0 - Q), LOGIT_EPS)),
    )


def igm_jac_device(y, u, p):
    """Analytic df/dy as a 3x3 nested tuple.

    The background carries no state dependence, so the only couplings are
    T_k -> the thermal and recombination terms and x_e -> Compton heating; Q
    decouples entirely.  Physical-space derivatives are converted to
    state-space ones by the chain factors of the log/logit encoding, which are
    zero wherever the corresponding clip is saturated.
    """
    log10_f_star = p[0]
    log10_f_X = p[1]
    log10_tvir = p[2]

    z = (1.0 + Z_INITIAL) * math.exp(-u) - 1.0
    opz = 1.0 + z
    H = H0_S * math.sqrt(OMEGA_M * opz**3 + OMEGA_L)
    T_gamma = T_CMB0 * opz

    f_star = 10.0**log10_f_star
    f_X = 10.0**log10_f_X
    turn_on_z = 31.0 - 7.0 * (log10_tvir - 4.0)
    turn_on = 1.0 / (1.0 + math.exp(-(turn_on_z - z) / 2.0))
    decline = 1.0 / (1.0 + math.exp(-(z - 5.8) / 0.7))
    growth = (opz / 20.0) ** -1.7
    source = 10.0 * f_star * turn_on * decline * growth

    log_Tk = y[0]
    clipped = min(max(log_Tk, _LOG_TK_MIN), _LOG_TK_MAX)
    T_k = math.exp(clipped)
    dTk = T_k if _LOG_TK_MIN < log_Tk < _LOG_TK_MAX else 0.0
    x_e = 1.0 / (1.0 + math.exp(-min(max(y[1], -40.0), 40.0)))
    dxe = x_e * (1.0 - x_e) if -40.0 < y[1] < 40.0 else 0.0
    Q = 1.0 / (1.0 + math.exp(-min(max(y[2], -40.0), 40.0)))
    dQd = Q * (1.0 - Q) if -40.0 < y[2] < 40.0 else 0.0

    compton_rate = 8.0e-20 * opz**4 * x_e / (1.0 + X_HE + x_e)
    dTdt = (
        -2.0 * H * T_k
        + compton_rate * (T_gamma - T_k)
        + 3.5e-16 * f_X * source
        + 1.2e-17 * (f_star / 0.01) * source
    )
    n_h = N_H0_CM3 * opz**3
    alpha_b = 2.6e-13 * (max(T_k, 10.0) / 1.0e4) ** -0.7
    dxedt = -alpha_b * n_h * x_e * x_e + 1.8e-17 * f_X * source * (1.0 - x_e)
    recomb_factor = (opz / 8.0) ** 3
    dQdt = (
        1.1e-16 * source * (f_star / 0.01) * (1.0 - Q) - 2.0e-17 * recomb_factor * Q * Q
    )

    w_x = x_e * (1.0 - x_e)
    w_q = Q * (1.0 - Q)
    den_T = H * T_k
    den_x = H * max(w_x, LOGIT_EPS)
    den_q = H * max(w_q, LOGIT_EPS)

    d_compton_dxe = 8.0e-20 * opz**4 * (1.0 + X_HE) / (1.0 + X_HE + x_e) ** 2
    f0 = dTdt / den_T
    # f0 = dTdt/(H*T_k), so d/dT_k also picks up the -f0/T_k denominator term.
    df0_dTk = (-2.0 * H - compton_rate) / den_T - f0 / T_k
    df0_dxe = d_compton_dxe * (T_gamma - T_k) / den_T

    # alpha_b freezes below T_k = 10, killing the only T_k path into dxedt.
    dalpha_dTk = -0.7 * alpha_b / T_k if T_k > 10.0 else 0.0
    f1 = dxedt / den_x
    df1_dTk = -dalpha_dTk * n_h * x_e * x_e / den_x
    df1_dxe = (-2.0 * alpha_b * n_h * x_e - 1.8e-17 * f_X * source) / den_x
    if w_x > LOGIT_EPS:
        # The denominator itself depends on x_e once the guard is inactive.
        df1_dxe -= f1 * (1.0 - 2.0 * x_e) / w_x

    f2 = dQdt / den_q
    df2_dQ = (
        -1.1e-16 * source * (f_star / 0.01) - 2.0 * 2.0e-17 * recomb_factor * Q
    ) / den_q
    if w_q > LOGIT_EPS:
        df2_dQ -= f2 * (1.0 - 2.0 * Q) / w_q

    return (
        (df0_dTk * dTk, df0_dxe * dxe, 0.0),
        (df1_dTk * dTk, df1_dxe * dxe, 0.0),
        (0.0, 0.0, df2_dQ * dQd),
    )


def igm_time_jac_device(y, u, p):
    """Analytic df/du, needed for Rodas5P to hold fifth order here.

    This is the dominant correction: the state is frozen and only H, T_gamma,
    n_h, the Compton prefactor, ``source`` and the recombination factor vary
    with u.  Each ``(1+z)^n`` factor contributes ``-n`` times itself.
    """
    log10_f_star = p[0]
    log10_f_X = p[1]
    log10_tvir = p[2]

    z = (1.0 + Z_INITIAL) * math.exp(-u) - 1.0
    opz = 1.0 + z
    H = H0_S * math.sqrt(OMEGA_M * opz**3 + OMEGA_L)
    # dH/dz = 3*OMEGA_M*(1+z)^2 * H0_S^2 / (2H), then dz/du = -(1+z).
    dH_du = -(3.0 * OMEGA_M * opz**3 * H0_S * H0_S) / (2.0 * H)
    T_gamma = T_CMB0 * opz
    dT_gamma_du = -T_gamma

    f_star = 10.0**log10_f_star
    f_X = 10.0**log10_f_X
    turn_on_z = 31.0 - 7.0 * (log10_tvir - 4.0)
    turn_on = 1.0 / (1.0 + math.exp(-(turn_on_z - z) / 2.0))
    decline = 1.0 / (1.0 + math.exp(-(z - 5.8) / 0.7))
    growth = (opz / 20.0) ** -1.7
    source = 10.0 * f_star * turn_on * decline * growth
    dsource_dz = (
        10.0
        * f_star
        * (
            -0.5 * turn_on * (1.0 - turn_on) * decline * growth
            + turn_on * decline * (1.0 - decline) / 0.7 * growth
            + turn_on * decline * (-1.7 * growth / opz)
        )
    )
    dsource_du = -opz * dsource_dz

    clipped = min(max(y[0], _LOG_TK_MIN), _LOG_TK_MAX)
    T_k = math.exp(clipped)
    x_e = 1.0 / (1.0 + math.exp(-min(max(y[1], -40.0), 40.0)))
    Q = 1.0 / (1.0 + math.exp(-min(max(y[2], -40.0), 40.0)))

    compton_rate = 8.0e-20 * opz**4 * x_e / (1.0 + X_HE + x_e)
    dTdt = (
        -2.0 * H * T_k
        + compton_rate * (T_gamma - T_k)
        + 3.5e-16 * f_X * source
        + 1.2e-17 * (f_star / 0.01) * source
    )
    n_h = N_H0_CM3 * opz**3
    alpha_b = 2.6e-13 * (max(T_k, 10.0) / 1.0e4) ** -0.7
    dxedt = -alpha_b * n_h * x_e * x_e + 1.8e-17 * f_X * source * (1.0 - x_e)
    recomb_factor = (opz / 8.0) ** 3
    dQdt = (
        1.1e-16 * source * (f_star / 0.01) * (1.0 - Q) - 2.0e-17 * recomb_factor * Q * Q
    )

    den_T = H * T_k
    den_x = H * max(x_e * (1.0 - x_e), LOGIT_EPS)
    den_q = H * max(Q * (1.0 - Q), LOGIT_EPS)
    dlogH_du = dH_du / H

    # compton_rate ~ (1+z)^4 -> d/du = -4 * compton_rate.
    dTdt_du = (
        -2.0 * dH_du * T_k
        + (-4.0 * compton_rate) * (T_gamma - T_k)
        + compton_rate * dT_gamma_du
        + 3.5e-16 * f_X * dsource_du
        + 1.2e-17 * (f_star / 0.01) * dsource_du
    )
    # n_h ~ (1+z)^3 -> d/du = -3 * n_h.
    dxedt_du = 3.0 * alpha_b * n_h * x_e * x_e + 1.8e-17 * f_X * dsource_du * (
        1.0 - x_e
    )
    # recomb_factor ~ (1+z)^3 -> d/du = -3 * recomb_factor.
    dQdt_du = (
        1.1e-16 * dsource_du * (f_star / 0.01) * (1.0 - Q)
        + 3.0 * 2.0e-17 * recomb_factor * Q * Q
    )

    # f_i = rate_i / (H * scale_i) with the scale frozen, so the denominator
    # contributes -f_i * dlogH/du.
    return (
        dTdt_du / den_T - (dTdt / den_T) * dlogH_du,
        dxedt_du / den_x - (dxedt / den_x) * dlogH_du,
        dQdt_du / den_q - (dQdt / den_q) * dlogH_du,
    )


def sample_parameters(key, n_samples=N_SAMPLES):
    """Draw prior samples [log10_f_star, log10_f_X, log10_Tvir]."""
    unit = jax.random.uniform(key, (n_samples, 3), dtype=jnp.float64)
    return PARAM_LO + unit * (PARAM_HI - PARAM_LO)


def initial_state():
    """Initial Dark Ages state at z=100."""
    z0 = Z_INITIAL
    # Gas is thermally coupled until z~150 and then cools adiabatically.
    T_k0 = T_CMB0 * 151.0 * ((1.0 + z0) / 151.0) ** 2
    x_e0 = 2.0e-4
    Q0 = 1.0e-6
    return jnp.array([jnp.log(T_k0), logit(x_e0), logit(Q0)], dtype=jnp.float64)


def make_solver(backend):
    """Return a uniform ``solve(ode_fn, y0, u_span, params)`` for a backend.

    The science uses the GPU-batched modax Rodas5P solver.  Two reference
    backends integrate the identical stiff 3-component IGM history for a
    like-for-like timing comparison:
      * "scipy"   -- serial CPU integration with scipy.solve_ivp (LSODA), the
                     no-GPU baseline used by codes such as ECHO21.
      * "diffrax" -- GPU integration with plain Diffrax Kvaerno5 (jax.vmap).
    """
    if backend == "modax":
        # The kernel solver takes its own CUDA-device callbacks rather than the
        # traced ``igm_ode``; ``f`` is ignored so every backend shares one
        # ``solve(f, y0, ts, p)`` signature.
        return lambda f, y0, ts, p: rodas5P_solve(
            igm_ode_device,
            igm_jac_device,
            y0,
            ts,
            p,
            time_jac_fn=igm_time_jac_device,
            lu_precision="fp32",
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            first_step=SOLVER_FIRST_STEP,
            max_steps=SOLVER_MAX_STEPS,
            error_weights=jnp.array([1.0, 0.2, 0.2], dtype=jnp.float64),
            pcoeff=0.3,
            icoeff=0.4,
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
            max_steps=SOLVER_MAX_STEPS,
        )
    if backend == "scipy":
        from reference.solvers.python.scipy_solve_ivp import solve as scipy_solve

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


def solve_histories(params, n_save=N_SAVE, backend="modax"):
    """Integrate the batched IGM histories."""
    u_span = jnp.linspace(0.0, u_from_redshift(Z_FINAL), n_save)
    solve_fn = make_solver(backend)
    if backend == "modax":
        return rodas5P_solve(
            igm_ode_device,
            igm_jac_device,
            initial_state(),
            u_span,
            params,
            time_jac_fn=igm_time_jac_device,
            lu_precision="fp32",
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            first_step=SOLVER_FIRST_STEP,
            max_steps=SOLVER_MAX_STEPS,
            error_weights=jnp.array([1.0, 0.2, 0.2], dtype=jnp.float64),
            pcoeff=0.3,
            icoeff=0.4,
        )
    return solve_fn(igm_ode, initial_state(), u_span, params)


def compute_observables(solution, params, n_save=N_SAVE):
    """Convert solver state histories into observable arrays and quantiles."""
    u_span = jnp.linspace(0.0, u_from_redshift(Z_FINAL), n_save)
    redshift = redshift_from_u(u_span)
    frequency_mhz = NU_21_MHZ / (1.0 + redshift)

    T_k = jnp.exp(solution[:, :, 0])
    x_e = sigmoid_from_logit(solution[:, :, 1])
    Q = sigmoid_from_logit(solution[:, :, 2])
    delta_Tb = jax.vmap(
        lambda sol_T, sol_Q, p: brightness_temperature_mk(redshift, sol_T, sol_Q, p)
    )(T_k, Q, params)

    quantiles = jnp.quantile(delta_Tb, jnp.array([0.025, 0.5, 0.975]), axis=0)
    return {
        "redshift": redshift,
        "frequency_mhz": frequency_mhz,
        "T_k": T_k,
        "x_e": x_e,
        "Q": Q,
        "delta_Tb_mK": delta_Tb,
        "quantiles_mK": quantiles,
    }


def run_monte_carlo(
    n_samples=N_SAMPLES,
    n_save=N_SAVE,
    seed=RANDOM_SEED,
):
    """Run the Monte Carlo ensemble and return arrays plus timing."""
    key = jax.random.key(seed)
    params = sample_parameters(key, n_samples)
    # Warm up before timing: the first call also pays the one-off XLA and
    # numba-CUDA kernel compilation, which dwarfs the solve itself and would
    # otherwise be most of the reported number.  Both compilations are
    # specialised on the ensemble shape, so the warm-up has to use the same
    # ``params`` and ``n_save`` as the timed call.
    solve_histories(params, n_save=n_save).block_until_ready()

    start = time.perf_counter()
    solution = solve_histories(params, n_save=n_save)
    # Block before timing and NumPy conversion.
    solution.block_until_ready()
    elapsed_s = time.perf_counter() - start
    observables = compute_observables(solution, params, n_save=n_save)
    observables = jax.tree_util.tree_map(np.asarray, observables)
    observables["params"] = np.asarray(params)
    observables["elapsed_s"] = elapsed_s
    return observables


def summarize(results):
    """Return a compact summary of the 21-cm absorption envelope."""
    q025, q50, q975 = results["quantiles_mK"]
    trough_index = int(np.argmin(q50))
    trough_depth = float(q50[trough_index])
    trough_low = float(q025[trough_index])
    trough_high = float(q975[trough_index])
    trough_frequency = float(results["frequency_mhz"][trough_index])
    trough_redshift = float(results["redshift"][trough_index])
    return {
        "trough_index": trough_index,
        "median_depth_mK": trough_depth,
        "low95_depth_mK": trough_low,
        "high95_depth_mK": trough_high,
        "frequency_mhz": trough_frequency,
        "redshift": trough_redshift,
    }


def plot_envelope(results, out_path=None):
    """Plot the Monte Carlo 95% envelope and median global signal."""
    if out_path is None:
        out_path = Path(__file__).parent / "global_21cm_envelope.png"
    q025, q50, q975 = results["quantiles_mK"]
    frequency = results["frequency_mhz"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.fill_between(
        frequency,
        q025,
        q975,
        color="#7aa6c2",
        alpha=0.35,
        label="95% prior envelope",
    )
    ax.plot(frequency, q50, color="#143d59", lw=2.2, label="median")
    ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.6)
    ax.set_xlabel("observed frequency [MHz]")
    ax.set_ylabel(r"$\delta T_b$ [mK]")
    ax.set_title("Global 21-cm Monte Carlo envelope")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return Path(out_path)


def time_solve(fn, repeats):
    """Return (mean seconds excluding compile, result) over ``repeats`` runs."""
    result = fn()
    jax.block_until_ready(result)
    t0 = time.perf_counter()
    for _ in range(repeats):
        result = fn()
        jax.block_until_ready(result)
    return (time.perf_counter() - t0) / repeats, result


def run_benchmark(n, backends, repeats, seed=RANDOM_SEED):
    key = jax.random.key(seed)
    params = sample_parameters(key, n)
    u_span = jnp.linspace(0.0, u_from_redshift(Z_FINAL), N_SAVE)
    y0 = initial_state()
    print(f"21-cm IGM benchmark: N = {n:,} stiff 3-component histories\n")
    print(f"{'backend':>10}  {'wall (s)':>10}  {'per solve':>12}  median-trough(mK)")
    print("-" * 60)
    for backend in backends:
        solve_fn = make_solver(backend)
        run = lambda sf=solve_fn: sf(igm_ode, y0, u_span, params)
        try:
            secs, sol = time_solve(run, repeats)
        except Exception as exc:  # noqa: BLE001
            print(f"{backend:>10}  FAILED: {exc}")
            continue
        sol = np.asarray(sol)
        obs = compute_observables(jnp.asarray(sol), params, n_save=N_SAVE)
        trough = float(np.nanmin(np.asarray(obs["quantiles_mK"])[1]))
        print(f"{backend:>10}  {secs:10.3f}  {secs / n * 1e3:9.4f} ms  {trough:.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="time the batched ensemble solve across solver backends",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["modax", "diffrax", "scipy"],
        choices=["modax", "diffrax", "scipy"],
    )
    parser.add_argument("--n", type=int, default=2000, help="ensemble size")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.n, args.backends, args.repeats)
        return

    print(
        f"Running {N_SAMPLES:,} uncoupled 21-cm IGM histories "
        f"with {N_SAVE} save points...",
        flush=True,
    )
    results = run_monte_carlo()
    summary = summarize(results)
    out = plot_envelope(results)

    print(f"Solve time: {results['elapsed_s']:.3f} s")
    print(
        "Median absorption trough: "
        f"{summary['median_depth_mK']:.1f} mK at "
        f"{summary['frequency_mhz']:.1f} MHz (z={summary['redshift']:.2f})"
    )
    print(
        "95% envelope at median-trough frequency: "
        f"[{summary['low95_depth_mK']:.1f}, {summary['high95_depth_mK']:.1f}] mK"
    )
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
