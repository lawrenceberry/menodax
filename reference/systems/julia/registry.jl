include("vdp.jl")
include("vdp_sens.jl")
include("heat.jl")
include("bateman.jl")
include("lorenz.jl")
include("robertson.jl")
include("kaps.jl")
include("brusselator.jl")

function make_system_spec(system_name::String, config)
    if system_name == "vdp"
        return make_vdp_spec(config)
    elseif system_name == "vdp_sens"
        return make_vdp_sens_spec(config)
    elseif system_name == "heat"
        return make_heat_spec(config)
    elseif system_name == "bateman"
        return make_bateman_spec(config)
    elseif system_name == "lorenz"
        return make_lorenz_spec(config)
    elseif system_name == "robertson"
        return make_robertson_spec(config)
    elseif system_name == "kaps"
        return make_kaps_spec(config)
    elseif system_name == "brusselator"
        return make_brusselator_spec(config)
    end
    error("Unknown reference ODE system '$system_name'")
end
