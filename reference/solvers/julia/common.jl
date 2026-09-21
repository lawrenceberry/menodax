using JSON
using SciMLBase
using StaticArrays

Base.@kwdef struct ReferenceSystemSpec
    build_array_full_problem::Function
    build_kernel_full_problem::Function
end

function parse_config(path::String)
    return JSON.parsefile(path)
end

function require_config_int(config, key::String)
    return Int(config[key])
end

function require_config_float(config, key::String)
    return Float64(config[key])
end

function maybe_reverse_perm(nd::Int)
    return Tuple(nd:-1:1)
end

function read_c_order_array(bin_path::String, meta_path::String)
    meta = JSON.parsefile(meta_path)
    shape = Tuple(Int.(meta["shape"]))
    data = open(bin_path, "r") do io
        read!(io, Vector{Float64}(undef, prod(shape)))
    end
    if length(shape) == 1
        return reshape(data, shape...)
    end
    reshaped = reshape(data, reverse(shape)...)
    return permutedims(reshaped, maybe_reverse_perm(length(shape)))
end

function write_c_order_array(bin_path::String, meta_path::String, arr)
    arr64 = Float64.(Array(arr))
    raw = if ndims(arr64) == 1
        vec(arr64)
    else
        vec(permutedims(arr64, maybe_reverse_perm(ndims(arr64))))
    end
    open(bin_path, "w") do io
        write(io, raw)
    end
    open(meta_path, "w") do io
        JSON.print(
            io,
            Dict(
                "dtype" => "float64",
                "order" => "C",
                "shape" => collect(size(arr64)),
            ),
        )
    end
    return nothing
end

function zero_tgrad!(dT, u, p, t)
    fill!(dT, 0.0)
    return nothing
end

function vector_to_svector(v::AbstractVector{<:Real})
    return SVector{length(v), Float64}(Tuple(Float64.(v)))
end

function matrix_row_to_svector(mat::AbstractMatrix{<:Real}, row::Int)
    return SVector{size(mat, 2), Float64}(Tuple(Float64.(mat[row, :])))
end

function logspace_from_one_to(max_value::Float64, count::Int)
    if count == 1
        return [1.0]
    end
    return [
        10.0^(log10(max_value) * (i - 1) / (count - 1))
        for i in 1:count
    ]
end

function logspace(start_exp::Float64, stop_exp::Float64, count::Int)
    if count == 1
        return [10.0^start_exp]
    end
    return [10.0^(start_exp + (stop_exp - start_exp) * (i - 1) / (count - 1)) for i in 1:count]
end
