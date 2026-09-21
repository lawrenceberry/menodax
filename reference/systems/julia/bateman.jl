function make_bateman_spec(config)
    n_vars = require_config_int(config, "n_vars")
    stiffness = require_config_float(config, "stiffness")
    n_radioactive = n_vars - 1
    lambdas = Tuple(logspace_from_one_to(stiffness, n_radioactive))

    function apply_bateman!(du, u, scale)
        du[1] = -scale * lambdas[1] * u[1]
        for i in 2:n_radioactive
            du[i] = scale * (lambdas[i - 1] * u[i - 1] - lambdas[i] * u[i])
        end
        du[n_vars] = scale * lambdas[n_radioactive] * u[n_radioactive]
        return nothing
    end

    function ode!(du, u, p, t)
        return apply_bateman!(du, u, p[1])
    end

    function jac!(J, u, p, t)
        fill!(J, 0.0)
        scale = p[1]
        for i in 1:n_radioactive
            J[i, i] = -scale * lambdas[i]
            J[i + 1, i] = scale * lambdas[i]
        end
        return nothing
    end

    function kernel_ode(u, p, t)
        scale = p[1]
        return SVector{n_vars, Float64}(ntuple(n_vars) do i
            if i == 1
                -scale * lambdas[1] * u[1]
            elseif i == n_vars
                scale * lambdas[n_radioactive] * u[n_radioactive]
            else
                scale * (lambdas[i - 1] * u[i - 1] - lambdas[i] * u[i])
            end
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
