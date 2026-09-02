# Julia GPU Reference Solvers

This folder contains Julia-based reference solvers that the Python tests call through
`run_solver.jl`. The Julia code uses `DiffEqGPU.jl` so the reference trajectories are
computed on the GPU rather than by a CPU fallback. The docs for Julia's DiffEqGPU.jl 
package are here: https://docs.sciml.ai/DiffEqGPU/stable/ and the code can be found 
here: https://github.com/SciML/DiffEqGPU.jl.

## Backends

`EnsembleGPUArray`
uses the ordinary SciML solver implementations on GPU-backed arrays. It supports a
wider range of algorithms, but stiff methods generally need the problem to provide
analytical derivative helpers such as Jacobians and time derivatives.

`EnsembleGPUKernel`
uses specialized GPU kernels for a smaller set of compatible solvers. It can be
faster when the ODE function fits the kernel restrictions, but it requires out-of-place
`StaticArrays`-style system definitions.

In this test harness:

- `Tsit5` runs as `Tsit5()` on `EnsembleGPUArray` and `GPUTsit5()` on `EnsembleGPUKernel`.
- `Kvaerno5` runs as `Kvaerno5()` on `EnsembleGPUArray` and `GPUKvaerno5()` on `EnsembleGPUKernel`.
- `Rodas5` runs as `Rodas5()` on `EnsembleGPUArray` and can run as `GPURodas5P()` on
  `EnsembleGPUKernel` for a limited validated subset in this harness:
  `stiff_scalar` and `nn_reactions` with `n_vars=5`.

## Environment

The Python wrappers launch Julia with:

```bash
julia --project=tests/reference_solvers/julia tests/reference_solvers/julia/run_solver.jl ...
```

If the local Julia environment has not been instantiated yet, activate this project and
run:

```julia
using Pkg
Pkg.instantiate()
```
