# The coupled van der Pol ring of vdp.jl, augmented with its forward
# sensitivities with respect to the damping scale p[1].
#
# DiffEqGPU differentiates neither ensemble backend, so a Julia user wanting the
# gradient of a GPU ensemble solve with respect to each trajectory's parameter
# has to integrate the sensitivity system alongside the state, which is what
# modax's custom_jvp rule does inside its kernel. The state is u = [y; S] with
# S = ∂y/∂p[1], both of length n = 2 n_osc, and the right-hand side is
#
#     dy/dt = f(y, p),    dS/dt = J_y(y, p) S + ∂f/∂p[1](y, p),
#
# with J_y S formed directly rather than through a matrix. The Jacobian of the
# augmented system, which the EnsembleGPUArray path wants written out, is block
# lower triangular: J_y on both diagonal blocks and L = ∂(J_y S + ∂f/∂p)/∂y in
# the lower-left, which for this right-hand side has two entries per oscillator.
function make_vdp_sens_spec(config)
    n_osc = require_config_int(config, "n_osc")
    MU = Float64(get(config, "mu", 100.0))
    D = Float64(get(config, "d", 10.0))
    OMEGA = Float64(get(config, "omega", 1.0))
    OMEGA_SQ = OMEGA * OMEGA
    n = 2 * n_osc

    function ode!(du, u, p, t)
        scale = p[1]
        for i in 1:n_osc
            xi = u[2i-1]
            vi = u[2i]
            il = mod(i - 2, n_osc) + 1
            ir = mod(i, n_osc) + 1
            laplacian_i = u[2ir-1] - 2.0 * xi + u[2il-1]
            g = MU * (1.0 - xi * xi) * vi  # ∂(dv_i/dt)/∂scale
            du[2i-1] = vi
            du[2i] = scale * g - OMEGA_SQ * xi + D * laplacian_i

            sxi = u[n+2i-1]
            svi = u[n+2i]
            slaplacian_i = u[n+2ir-1] - 2.0 * sxi + u[n+2il-1]
            du[n+2i-1] = svi
            du[n+2i] = (-2.0 * scale * MU * xi * vi - OMEGA_SQ) * sxi +
                       scale * MU * (1.0 - xi * xi) * svi +
                       D * slaplacian_i + g
        end
        return nothing
    end

    function jac!(J, u, p, t)
        scale = p[1]
        fill!(J, 0.0)
        for i in 1:n_osc
            xi = u[2i-1]
            vi = u[2i]
            sxi = u[n+2i-1]
            svi = u[n+2i]
            il = mod(i - 2, n_osc) + 1
            ir = mod(i, n_osc) + 1
            dv_dx = -2.0 * scale * MU * xi * vi - OMEGA_SQ - 2.0 * D
            dv_dv = scale * MU * (1.0 - xi * xi)
            # State block, J_y.
            J[2i-1, 2i] = 1.0
            J[2i, 2i-1] = dv_dx
            J[2i, 2i] = dv_dv
            J[2i, 2*ir-1] += D
            J[2i, 2*il-1] += D
            # Sensitivity block: the same J_y.
            J[n+2i-1, n+2i] = 1.0
            J[n+2i, n+2i-1] = dv_dx
            J[n+2i, n+2i] = dv_dv
            J[n+2i, n+2*ir-1] += D
            J[n+2i, n+2*il-1] += D
            # Coupling block L = ∂(J_y S + ∂f/∂scale)/∂y.
            J[n+2i, 2i-1] = -2.0 * scale * MU * vi * sxi - 2.0 * scale * MU * xi * svi -
                            2.0 * MU * xi * vi
            J[n+2i, 2i] = -2.0 * scale * MU * xi * sxi + MU * (1.0 - xi * xi)
        end
        return nothing
    end

    function kernel_ode(u::SVector{N, T}, p, t) where {N, T}
        scale = p[1]
        n_osc_local = N ÷ 4
        n_local = N ÷ 2
        du = MVector{N, T}(undef)
        for i in 1:n_osc_local
            xi = u[2i-1]
            vi = u[2i]
            il = mod(i - 2, n_osc_local) + 1
            ir = mod(i, n_osc_local) + 1
            laplacian_i = u[2ir-1] - 2.0 * xi + u[2il-1]
            g = MU * (1.0 - xi * xi) * vi
            du[2i-1] = vi
            du[2i] = scale * g - OMEGA_SQ * xi + D * laplacian_i

            sxi = u[n_local+2i-1]
            svi = u[n_local+2i]
            slaplacian_i = u[n_local+2ir-1] - 2.0 * sxi + u[n_local+2il-1]
            du[n_local+2i-1] = svi
            du[n_local+2i] = (-2.0 * scale * MU * xi * vi - OMEGA_SQ) * sxi +
                             scale * MU * (1.0 - xi * xi) * svi +
                             D * slaplacian_i + g
        end
        return SVector(du)
    end

    return ReferenceSystemSpec(
        build_array_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(ode!; jac=jac!, tgrad=zero_tgrad!),
            copy(y0),
            tspan,
            copy(p0),
        ),
        build_kernel_full_problem=(y0, tspan, p0) -> SciMLBase.ODEProblem{false}(
            kernel_ode,
            vector_to_svector(y0),
            tspan,
            vector_to_svector(p0),
        ),
    )
end
