"""Forward-sensitivity JVP tests for the numba-cuda solvers.

The solvers reach JAX as an opaque FFI custom call, so their derivatives come
from a ``jax.custom_jvp`` rule that integrates ``S' = J_y S + J_p`` alongside
the state.  These tests check that rule three ways: against closed-form
sensitivities where the ODE has them, against central differences where it does
not, and against the structural promises the rule makes -- that an
undifferentiated call never integrates sensitivities, that only the blocks JAX
asks for are carried, and that a joint solve reproduces the plain solve's value.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from menodax._sensitivity import SensitivitySpec, augmented_y0, split_augmented
from menodax.rodas5P import solve as rodas5P_solve
from menodax.tsit5 import solve as tsit5_solve

jax.config.update("jax_enable_x64", True)

requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")
both_solvers = pytest.mark.parametrize(
    "solve_fn, kwargs",
    [
        pytest.param(tsit5_solve, {}, id="tsit5"),
        pytest.param(rodas5P_solve, {"lu_precision": "fp64"}, id="rodas5P"),
    ],
)


# --- systems ---------------------------------------------------------------
# Linear cascade: y0' = -a y0, y1' = a y0 - b y1, with a closed-form solution
# and closed-form sensitivities in both parameters and both initial states.
A, B = 0.7, 1.3
Y0 = np.array([2.0, 0.5])
T_SPAN = np.linspace(0.0, 2.0, 5)


def cascade(y, t, p):
    return (-p[0] * y[0], p[0] * y[0] - p[1] * y[1])


def cascade_exact(t, y0, p):
    """``y(t)`` for the cascade, as a ``(n_save, 2)`` array."""
    a, b = p
    c0, c1 = y0
    k = a / (b - a)
    return np.stack(
        [c0 * np.exp(-a * t), (c1 - k * c0) * np.exp(-b * t) + k * c0 * np.exp(-a * t)],
        axis=-1,
    )


def cascade_dparams(t, y0, p):
    """``dy/d(a, b)`` for the cascade, as a ``(n_save, 2, 2)`` array."""
    a, b = p
    c0, c1 = y0
    k = a / (b - a)
    dk_da = b / (b - a) ** 2
    dk_db = -a / (b - a) ** 2
    ea, eb = np.exp(-a * t), np.exp(-b * t)
    d_da = np.stack(
        [-t * c0 * ea, c0 * (dk_da * (ea - eb) - k * t * ea)],
        axis=-1,
    )
    d_db = np.stack(
        [np.zeros_like(t), c0 * dk_db * (ea - eb) - t * (c1 - k * c0) * eb],
        axis=-1,
    )
    return np.stack([d_da, d_db], axis=-1)


def cascade_dy0(t, y0, p):
    """``dy/d(c0, c1)`` for the cascade, as a ``(n_save, 2, 2)`` array."""
    a, b = p
    k = a / (b - a)
    ea, eb = np.exp(-a * t), np.exp(-b * t)
    d_dc0 = np.stack([ea, k * (ea - eb)], axis=-1)
    d_dc1 = np.stack([np.zeros_like(t), eb], axis=-1)
    return np.stack([d_dc0, d_dc1], axis=-1)


# Non-autonomous linear problem: y' = -lam y + F t, y(0) = 0.  Its sensitivity
# system inherits the explicit time dependence, so it exercises the df/dt term
# Rodas5P needs -- and the W approximation that drops it from the sensitivity
# block of the joint system's time derivative.
LAM, FORCING = 10.0, 5.0


def forced(y, t, p):
    return (-p[0] * y[0] + p[1] * t,)


def forced_exact(t, lam, forcing):
    return forcing * (lam * t - 1.0 + np.exp(-lam * t)) / lam**2


def forced_dlam(t, lam, forcing):
    return (
        forcing
        * ((t - t * np.exp(-lam * t)) * lam - 2.0 * (lam * t - 1.0 + np.exp(-lam * t)))
        / lam**3
    )


def forced_dforcing(t, lam, _forcing):
    return (lam * t - 1.0 + np.exp(-lam * t)) / lam**2


def central_diff(fn, x, rel=1e-6):
    """``d fn / d x`` by central differences, stacked on a trailing axis."""
    columns = []
    for k in range(x.shape[0]):
        step = rel * abs(float(x[k])) or rel
        plus = np.asarray(fn(x.at[k].add(step)))
        minus = np.asarray(fn(x.at[k].add(-step)))
        columns.append((plus - minus) / (2.0 * step))
    return np.stack(columns, axis=-1)


# --- the spec itself (no device needed) ------------------------------------
def test_spec_sizes_only_count_requested_blocks():
    """Only the blocks JAX asks for are carried, so grad wrt p stays cheap."""
    params_only = SensitivitySpec(n_vars=4, n_params=2, wrt_y0=False, wrt_params=True)
    assert params_only.n_sens == 2
    assert params_only.n_aug == 4 * (1 + 2)

    y0_only = SensitivitySpec(n_vars=4, n_params=2, wrt_y0=True, wrt_params=False)
    assert y0_only.n_sens == 4
    assert y0_only.n_aug == 4 * (1 + 4)

    both = SensitivitySpec(n_vars=4, n_params=2, wrt_y0=True, wrt_params=True)
    assert both.n_sens == 6
    assert both.n_aug == 4 * (1 + 6)

    with pytest.raises(ValueError, match="at least one direction block"):
        SensitivitySpec(n_vars=4, n_params=2, wrt_y0=False, wrt_params=False)


def test_spec_error_norm_denominator():
    """Excluded sensitivities must not dilute the state's error norm."""
    spec = SensitivitySpec(3, 2, False, True, False)
    assert spec.n_error == 3
    assert SensitivitySpec(3, 2, False, True, True).n_error == spec.n_aug


def test_augmented_layout_round_trips():
    """``S(t0)`` is the identity for y0 directions and zero for parameters."""
    spec = SensitivitySpec(n_vars=2, n_params=3, wrt_y0=True, wrt_params=True)
    y0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    z0 = augmented_y0(y0, spec)
    assert z0.shape == (2, spec.n_aug)

    state, sens = split_augmented(z0[:, None, :], spec)
    np.testing.assert_allclose(np.asarray(state)[:, 0], np.asarray(y0))
    for trajectory in range(2):
        np.testing.assert_allclose(np.asarray(sens)[trajectory, 0, :, :2], np.eye(2))
        np.testing.assert_allclose(np.asarray(sens)[trajectory, 0, :, 2:], 0.0)


# --- closed-form sensitivities ---------------------------------------------
@requires_cuda
@both_solvers
def test_jacfwd_matches_closed_form(solve_fn, kwargs):
    """dy/dp and dy/dy0 for a linear cascade, against the analytic solution."""
    y0 = jnp.asarray(Y0)
    params = jnp.asarray([A, B])
    t_span = jnp.asarray(T_SPAN)
    settings = dict(rtol=1e-8, atol=1e-10, **kwargs)

    def run(a, p):
        return solve_fn(cascade, a, t_span, p, **settings)[0]

    np.testing.assert_allclose(
        np.asarray(run(y0, params)), cascade_exact(T_SPAN, Y0, [A, B]), rtol=1e-7
    )

    jac_p = np.asarray(jax.jacfwd(lambda p: run(y0, p))(params))
    np.testing.assert_allclose(
        jac_p, cascade_dparams(T_SPAN, Y0, [A, B]), rtol=1e-6, atol=1e-9
    )

    jac_y0 = np.asarray(jax.jacfwd(lambda a: run(a, params))(y0))
    # dy(t0)/dy0 is the identity: S starts from I and the save grid starts at t0.
    np.testing.assert_allclose(jac_y0[0], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(
        jac_y0, cascade_dy0(T_SPAN, Y0, [A, B]), rtol=1e-6, atol=1e-9
    )


@requires_cuda
@both_solvers
def test_nonautonomous_sensitivities_match_closed_form(solve_fn, kwargs):
    """The joint system stays accurate when the right-hand side depends on t.

    Rodas5P assembles the joint iteration matrix as ``diag(J, ..., J)`` and
    zeroes the sensitivity block of ``df/dt``; both are W-method approximations
    of the joint Jacobian.  If either cost the method its order, the
    sensitivities of this forced problem -- whose ``d2f/dt dp`` is a nonzero
    constant -- would be the first thing to show it.
    """
    y0 = jnp.zeros(1)
    params = jnp.asarray([LAM, FORCING])
    t_span = jnp.asarray(np.linspace(0.0, 1.0, 11))

    def run(p):
        return solve_fn(forced, y0, t_span, p, rtol=1e-8, atol=1e-10, **kwargs)[0, :, 0]

    np.testing.assert_allclose(
        np.asarray(run(params)),
        forced_exact(np.asarray(t_span), LAM, FORCING),
        rtol=1e-7,
        atol=1e-12,
    )
    jac = np.asarray(jax.jacfwd(run)(params))
    exact = np.stack(
        [
            forced_dlam(np.asarray(t_span), LAM, FORCING),
            forced_dforcing(np.asarray(t_span), LAM, FORCING),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(jac, exact, rtol=1e-5, atol=1e-10)


# --- reverse mode, which transposes the same rule ---------------------------
@requires_cuda
@both_solvers
def test_grad_matches_central_differences(solve_fn, kwargs):
    """jax.grad works by transposing the tangent contraction -- no adjoint solve."""
    y0 = jnp.asarray(Y0)
    params = jnp.asarray([A, B])
    t_span = jnp.asarray(T_SPAN)
    settings = dict(rtol=1e-10, atol=1e-12, **kwargs)

    def loss(p):
        return jnp.sum(solve_fn(cascade, y0, t_span, p, **settings) ** 2)

    # Central differences are the noisy side of this comparison, not the solver.
    grad = np.asarray(jax.grad(loss)(params))
    np.testing.assert_allclose(grad, central_diff(loss, params), rtol=1e-4)

    def loss_y0(a):
        return jnp.sum(solve_fn(cascade, a, t_span, params, **settings) ** 2)

    np.testing.assert_allclose(
        np.asarray(jax.grad(loss_y0)(y0)), central_diff(loss_y0, y0), rtol=1e-4
    )


@requires_cuda
@both_solvers
def test_value_and_grad_reuses_one_joint_solve(solve_fn, kwargs):
    """The value from the joint solve is the value the plain solve returns."""
    y0 = jnp.asarray(Y0)
    params = jnp.asarray([A, B])
    t_span = jnp.asarray(T_SPAN)
    settings = dict(rtol=1e-10, atol=1e-12, **kwargs)

    def loss(p):
        return jnp.sum(solve_fn(cascade, y0, t_span, p, **settings) ** 2)

    # The joint solve error-controls the sensitivities by default, so it picks
    # its own steps; the value it returns still agrees to solver tolerance.
    value, grad = jax.value_and_grad(loss)(params)
    np.testing.assert_allclose(float(value), float(loss(params)), rtol=1e-8)
    np.testing.assert_allclose(
        np.asarray(grad), np.asarray(jax.grad(loss)(params)), rtol=1e-12
    )


@requires_cuda
@both_solvers
def test_grad_wrt_params_ignores_the_y0_block(solve_fn, kwargs):
    """Asking only for dL/dp must not change the answer, only the cost."""
    y0 = jnp.asarray(Y0)
    params = jnp.asarray([A, B])
    t_span = jnp.asarray(T_SPAN)
    settings = dict(rtol=1e-10, atol=1e-12, **kwargs)

    def loss(a, p):
        return jnp.sum(solve_fn(cascade, a, t_span, p, **settings) ** 2)

    params_only = jax.grad(loss, 1)(y0, params)
    both = jax.grad(loss, (0, 1))(y0, params)
    # Not bit-identical: carrying the initial-state block as well puts more
    # components in the error norm, which moves the step sequence slightly.
    np.testing.assert_allclose(np.asarray(params_only), np.asarray(both[1]), rtol=1e-8)


# --- integration with the rest of the JAX surface ---------------------------
@requires_cuda
@both_solvers
def test_ensemble_gradients_are_per_trajectory(solve_fn, kwargs):
    """Each trajectory's gradient depends only on its own row of params."""
    t_span = jnp.asarray(T_SPAN)
    y0 = jnp.asarray(np.broadcast_to(Y0, (3, 2)))
    params = jnp.asarray([[A, B], [0.4, 0.9], [1.1, 2.0]])
    settings = dict(rtol=1e-10, atol=1e-12, **kwargs)

    def loss(p):
        return jnp.sum(solve_fn(cascade, y0, t_span, p, **settings) ** 2)

    grad = np.asarray(jax.jit(jax.grad(loss))(params))
    assert grad.shape == (3, 2)
    for row in range(3):
        single = jax.grad(
            lambda p: jnp.sum(solve_fn(cascade, y0[row], t_span, p, **settings) ** 2)
        )(params[row])
        np.testing.assert_allclose(grad[row], np.asarray(single), rtol=1e-8)


@requires_cuda
@both_solvers
def test_grad_through_vmap(solve_fn, kwargs):
    """An outer vmap still lowers to one ensemble launch under differentiation."""
    t_span = jnp.asarray(T_SPAN)
    y0 = jnp.asarray(np.broadcast_to(Y0, (3, 2)))
    params = jnp.asarray([[A, B], [0.4, 0.9], [1.1, 2.0]])
    settings = dict(rtol=1e-10, atol=1e-12, **kwargs)

    def loss(p):
        per_solve = jax.vmap(lambda a, q: solve_fn(cascade, a, t_span, q, **settings))(
            y0, p
        )
        return jnp.sum(per_solve**2)

    def flat_loss(p):
        return jnp.sum(solve_fn(cascade, y0, t_span, p, **settings) ** 2)

    np.testing.assert_allclose(
        np.asarray(jax.grad(loss)(params)),
        np.asarray(jax.grad(flat_loss)(params)),
        rtol=1e-8,
    )


@requires_cuda
@both_solvers
def test_stats_stay_available_under_grad(solve_fn, kwargs):
    """Step counters have no derivative but must not block differentiation."""
    y0 = jnp.asarray(Y0)
    params = jnp.asarray([A, B])
    t_span = jnp.asarray(T_SPAN)
    settings = dict(rtol=1e-10, atol=1e-12, return_stats=True, **kwargs)

    def loss(p):
        solution, stats = solve_fn(cascade, y0, t_span, p, **settings)
        assert set(stats) == {"accepted_steps", "rejected_steps", "loop_steps"}
        return jnp.sum(solution**2)

    grad = jax.grad(loss)(params)
    assert np.all(np.isfinite(np.asarray(grad)))


@requires_cuda
def test_t_span_is_not_differentiable():
    """Save times carry a real derivative the kernel does not produce; say so."""
    y0 = jnp.asarray(Y0)
    params = jnp.asarray([A, B])
    with pytest.raises(NotImplementedError, match="t_span"):
        jax.grad(
            lambda t: jnp.sum(
                tsit5_solve(cascade, y0, t, params, rtol=1e-8, atol=1e-10)
            )
        )(jnp.asarray(T_SPAN))


@requires_cuda
@both_solvers
def test_joint_solve_keeps_the_plain_step_sequence(solve_fn, kwargs):
    """Sensitivities ride along on steps chosen for the state alone.

    With ``sens_error_control=False`` the sensitivity components carry zero
    error weight and the norm still divides by ``n_vars``, so the joint solve
    should accept exactly the steps the plain solve accepts.
    """
    y0 = jnp.asarray(Y0)
    params = jnp.asarray([A, B])
    t_span = jnp.asarray(T_SPAN)
    settings = dict(
        rtol=1e-10, atol=1e-12, return_stats=True, sens_error_control=False, **kwargs
    )

    solution, stats = solve_fn(cascade, y0, t_span, params, **settings)
    assert solution.shape == (1, len(T_SPAN), 2)
    np.testing.assert_allclose(
        np.asarray(solution)[0], cascade_exact(T_SPAN, Y0, [A, B]), rtol=1e-7
    )

    # jax.jvp exposes the joint solve's own primal output, step counters and all.
    (joint_solution, joint_stats), _ = jax.jvp(
        lambda p: solve_fn(cascade, y0, t_span, p, **settings),
        (params,),
        (jnp.ones_like(params),),
    )
    np.testing.assert_array_equal(
        np.asarray(stats["accepted_steps"]),
        np.asarray(joint_stats["accepted_steps"]),
    )
    np.testing.assert_allclose(
        np.asarray(joint_solution), np.asarray(solution), rtol=1e-12
    )


@requires_cuda
@both_solvers
def test_sens_error_control_is_what_makes_the_gradient_converge(solve_fn, kwargs):
    """Error control is what *ties* the gradient's accuracy to ``rtol``.

    With the joint system's exact Jacobian both settings are accurate, because
    the method is no longer fighting an approximation; controlling the
    sensitivities is what turns that from a happy accident into a guarantee,
    and it is what the default buys for its ~20% extra steps.
    """
    y0 = jnp.zeros(1)
    params = jnp.asarray([LAM, FORCING])
    t_span = jnp.asarray(np.linspace(0.0, 1.0, 11))
    exact = np.stack(
        [
            forced_dlam(np.asarray(t_span), LAM, FORCING),
            forced_dforcing(np.asarray(t_span), LAM, FORCING),
        ],
        axis=-1,
    )

    def error(control):
        jac = np.asarray(
            jax.jacfwd(
                lambda p: solve_fn(
                    forced,
                    y0,
                    t_span,
                    p,
                    rtol=1e-8,
                    atol=1e-10,
                    sens_error_control=control,
                    **kwargs,
                )[0, :, 0]
            )(params)
        )
        return np.abs(jac - exact).max() / np.abs(exact).max()

    controlled = error(True)
    assert controlled < 1e-6
    assert controlled <= error(False)


# --- the exact joint Jacobian, which is what makes any of this affordable ---
@requires_cuda
def test_joint_solve_costs_about_what_the_plain_solve_costs():
    """A bilinear right-hand side must not blow up the joint step count.

    Rodas5P is linearly implicit: the Jacobian sits inside the formula, so an
    approximate one lands in the answer.  Dropping the joint Jacobian's
    lower-left block ``d(J_y S + J_p)/dy`` keeps order 5 (it is a W method) but
    wrecks the error constant, and the controller pays for that in steps -- 200x
    was measured on exactly this problem.  Forming that block from a
    second-order directional sweep is what brings it back to parity, so this
    test is the regression guard on it.
    """
    y0 = jnp.asarray(Y0)
    params = jnp.asarray([A, B])
    t_span = jnp.asarray(T_SPAN)
    settings = dict(rtol=1e-8, atol=1e-10, return_stats=True, lu_precision="fp64")

    _, plain = rodas5P_solve(cascade, y0, t_span, params, **settings)
    (_, joint), _ = jax.jvp(
        lambda p: rodas5P_solve(cascade, y0, t_span, p, **settings),
        (params,),
        (jnp.ones_like(params),),
    )
    plain_steps = int(np.asarray(plain["accepted_steps"])[0])
    joint_steps = int(np.asarray(joint["accepted_steps"])[0])
    assert joint_steps <= 2 * plain_steps, (
        f"joint solve took {joint_steps} steps against the plain solve's "
        f"{plain_steps}; the joint Jacobian's coupling block is probably wrong"
    )


@requires_cuda
@both_solvers
def test_stiff_bilinear_gradients(solve_fn, kwargs):
    """Robertson: stiff, three species, three rates, bilinear in state and rate.

    The bilinear terms are what make ``d2f/dy dp`` nonzero, so this is the
    shape of problem the joint Jacobian's coupling block exists for -- and the
    shape most reaction networks have.
    """

    def robertson(y, t, p):
        return (
            -p[0] * y[0] + p[2] * y[1] * y[2],
            p[0] * y[0] - p[1] * y[1] * y[1] - p[2] * y[1] * y[2],
            p[1] * y[1] * y[1],
        )

    y0 = jnp.asarray([1.0, 0.0, 0.0])
    params = jnp.asarray([0.04, 3.0e7, 1.0e4])
    t_span = jnp.asarray([0.0, 1e-3, 1e-1, 1.0, 10.0])
    settings = dict(rtol=1e-10, atol=1e-12, max_steps=1000000, **kwargs)

    def run(p):
        return solve_fn(robertson, y0, t_span, p, **settings)[0]

    # Rates span nine orders of magnitude, so the differences have to be
    # relative and the comparison loose enough for their truncation error.
    jac = np.asarray(jax.jacfwd(run)(params))
    reference = central_diff(run, params, rel=1e-6)
    scale = np.abs(reference).max()
    assert np.abs(jac - reference).max() / scale < 1e-4

    def loss(p):
        return jnp.sum(run(p)[-1] ** 2)

    grad = np.asarray(jax.grad(loss)(params))
    grad_fd = central_diff(loss, params, rel=1e-6)
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-4)
