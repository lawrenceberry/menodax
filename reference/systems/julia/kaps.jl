function make_kaps_spec(config)
    n_pairs = require_config_int(config, "n_pairs")
    epsilon_min = require_config_float(config, "epsilon_min")
    n_vars = 2 * n_pairs
    epsilon = Tuple(logspace(0.0, log10(epsilon_min), n_pairs))

    function ode!(du, u, p, t)
        s = p[1]
        for pair_idx in 1:n_pairs
            y1_idx = 2 * pair_idx - 1
            y2_idx = y1_idx + 1
            y1 = u[y1_idx]
            y2 = u[y2_idx]
            inv_eps = 1.0 / epsilon[pair_idx]
            du[y1_idx] = s * (-(inv_eps + 2.0) * y1 + inv_eps * y2^2)
            du[y2_idx] = s * (y1 - y2 - y2^2)
        end
        return nothing
    end

    function jac!(J, u, p, t)
        fill!(J, 0.0)
        s = p[1]
        for pair_idx in 1:n_pairs
            y1_idx = 2 * pair_idx - 1
            y2_idx = y1_idx + 1
            y2 = u[y2_idx]
            inv_eps = 1.0 / epsilon[pair_idx]
            J[y1_idx, y1_idx] = -s * (inv_eps + 2.0)
            J[y1_idx, y2_idx] = s * (2.0 * inv_eps * y2)
            J[y2_idx, y1_idx] = s
            J[y2_idx, y2_idx] = s * (-1.0 - 2.0 * y2)
        end
        return nothing
    end

    function kernel_ode(u, p, t)
        s = p[1]
        return SVector{n_vars, Float64}(ntuple(n_vars) do idx
            pair_idx = fld(idx + 1, 2)
            y1_idx = 2 * pair_idx - 1
            y2_idx = y1_idx + 1
            y1 = u[y1_idx]
            y2 = u[y2_idx]
            inv_eps = 1.0 / epsilon[pair_idx]
            isodd(idx) ? s * (-(inv_eps + 2.0) * y1 + inv_eps * y2^2) : s * (y1 - y2 - y2^2)
        end)
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
