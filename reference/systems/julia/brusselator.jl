function make_brusselator_spec(config)
    n_grid = require_config_int(config, "n_grid")
    a = Float64(get(config, "a", 1.0))
    b = Float64(get(config, "b", 3.0))
    alpha = Float64(get(config, "alpha", 0.02))
    length_l = Float64(get(config, "length", 1.0))
    dx = length_l / n_grid
    diff_coeff = alpha / (dx * dx)
    n_vars = 2 * n_grid

    function ode!(du, u, p, t)
        scale = p[1]
        a_eff = scale * a
        b_eff = scale * b
        for i in 1:n_grid
            il = mod(i - 2, n_grid) + 1
            ir = mod(i, n_grid) + 1
            ui = u[2i - 1]
            vi = u[2i]
            u2v = ui * ui * vi
            lap_u = u[2ir - 1] - 2.0 * ui + u[2il - 1]
            lap_v = u[2ir] - 2.0 * vi + u[2il]
            du[2i - 1] = a_eff + u2v - (b_eff + 1.0) * ui + diff_coeff * lap_u
            du[2i] = b_eff * ui - u2v + diff_coeff * lap_v
        end
        return nothing
    end

    function jac!(J, u, p, t)
        fill!(J, 0.0)
        scale = p[1]
        b_eff = scale * b
        for i in 1:n_grid
            il = mod(i - 2, n_grid) + 1
            ir = mod(i, n_grid) + 1
            ui = u[2i - 1]
            vi = u[2i]
            # d(du_i)/d(u_i) = 2*ui*vi - (b_eff + 1) - 2 * diff_coeff
            J[2i - 1, 2i - 1] = 2.0 * ui * vi - (b_eff + 1.0) - 2.0 * diff_coeff
            # d(du_i)/d(v_i) = ui^2
            J[2i - 1, 2i] = ui * ui
            # d(du_i)/d(u_il), d(u_ir) = diff_coeff
            J[2i - 1, 2 * il - 1] += diff_coeff
            J[2i - 1, 2 * ir - 1] += diff_coeff
            # d(dv_i)/d(u_i) = b_eff - 2*ui*vi
            J[2i, 2i - 1] = b_eff - 2.0 * ui * vi
            # d(dv_i)/d(v_i) = -ui^2 - 2 * diff_coeff
            J[2i, 2i] = -ui * ui - 2.0 * diff_coeff
            J[2i, 2 * il] += diff_coeff
            J[2i, 2 * ir] += diff_coeff
        end
        return nothing
    end

    function kernel_ode(u::SVector{N, T}, p, t) where {N, T}
        scale = p[1]
        a_eff = scale * a
        b_eff = scale * b
        n_grid_local = N ÷ 2
        du = MVector{N, T}(undef)
        for i in 1:n_grid_local
            il = mod(i - 2, n_grid_local) + 1
            ir = mod(i, n_grid_local) + 1
            ui = u[2i - 1]
            vi = u[2i]
            u2v = ui * ui * vi
            lap_u = u[2ir - 1] - 2.0 * ui + u[2il - 1]
            lap_v = u[2ir] - 2.0 * vi + u[2il]
            du[2i - 1] = a_eff + u2v - (b_eff + 1.0) * ui + diff_coeff * lap_u
            du[2i] = b_eff * ui - u2v + diff_coeff * lap_v
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
