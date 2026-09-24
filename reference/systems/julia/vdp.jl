# The coupled van der Pol ring of reference/systems/python/vdp.py. Each
# oscillator is coupled diffusively to every oscillator within `coupling_range`
# places along the ring, one term per place, so an oscillator the ring reaches
# from both sides counts twice (as the nearest-neighbour ring already does at
# n_osc == 2), and the oscillator itself is skipped. The coupling D is divided
# by the range so its total strength stays comparable; a range of 1 is the
# nearest-neighbour ring and a range of n_osc ÷ 2 couples every oscillator to
# every other. This mirrors `neighbours` in the Python module exactly.
function make_vdp_spec(config)
    n_osc = require_config_int(config, "n_osc")
    MU = Float64(get(config, "mu", 100.0))
    D = Float64(get(config, "d", 10.0))
    OMEGA = Float64(get(config, "omega", 1.0))
    K = Int(get(config, "coupling_range", 1))
    OMEGA_SQ = OMEGA * OMEGA
    STRENGTH = D / K

    # Sum of (x_j - x_i) over the coupled oscillators j of oscillator i, for a
    # state whose positions are at the odd indices.
    @inline function laplacian(u, i, xi, n)
        acc = zero(xi)
        for o in 1:K
            jl = mod(i - 1 - o, n) + 1
            jr = mod(i - 1 + o, n) + 1
            if jl != i
                acc += u[2jl-1] - xi
            end
            if jr != i
                acc += u[2jr-1] - xi
            end
        end
        return acc
    end

    function ode!(du, u, p, t)
        scale = p[1]
        for i in 1:n_osc
            xi = u[2i-1]
            vi = u[2i]
            du[2i-1] = vi
            du[2i] = scale * MU * (1.0 - xi * xi) * vi - OMEGA_SQ * xi +
                     STRENGTH * laplacian(u, i, xi, n_osc)
        end
        return nothing
    end

    function jac!(J, u, p, t)
        scale = p[1]
        fill!(J, 0.0)
        for i in 1:n_osc
            xi = u[2i-1]
            vi = u[2i]
            # d(dx_i/dt)/dv_i = 1
            J[2i-1, 2i] = 1.0
            # d(dv_i/dt)/dx_i = -2*scale*MU*xi*vi - OMEGA_SQ - STRENGTH per coupling
            J[2i, 2i-1] = -2.0 * scale * MU * xi * vi - OMEGA_SQ
            # d(dv_i/dt)/dv_i = scale*MU*(1 - xi^2)
            J[2i, 2i] = scale * MU * (1.0 - xi * xi)
            for o in 1:K
                for j in (mod(i - 1 - o, n_osc) + 1, mod(i - 1 + o, n_osc) + 1)
                    if j != i
                        J[2i, 2j-1] += STRENGTH
                        J[2i, 2i-1] -= STRENGTH
                    end
                end
            end
        end
        return nothing
    end

    function kernel_ode(u::SVector{N, T}, p, t) where {N, T}
        scale = p[1]
        n_osc_local = N ÷ 2
        du = MVector{N, T}(undef)
        for i in 1:n_osc_local
            xi = u[2i-1]
            vi = u[2i]
            du[2i-1] = vi
            du[2i] = scale * MU * (1.0 - xi * xi) * vi - OMEGA_SQ * xi +
                     STRENGTH * laplacian(u, i, xi, n_osc_local)
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
