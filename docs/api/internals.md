# Internals

The modules behind `solve`. They are private — the package's surface is the two
`solve` functions — but the design argument in the guide refers to them, so
they are documented here.

## Sparsity and colouring

::: solvers._sparsity

## The compiled sparse direct solver

::: solvers._sparse_direct
    options:
      members:
        - ORDERINGS
        - MAX_UNROLLED_SUBSTITUTIONS
        - MAX_UNROLLED_UPDATES
        - SparseLULayout
        - SparseDirectSolver
        - fill_reducing_order
        - fill_pattern
        - analyse
        - compressed_jacobian
        - sparse_direct_solver
        - sparse_direct_solver_for

## Forward sensitivities

::: solvers._sensitivity

## Host-side kernel support

::: solvers._numba_common

## The JAX glue

::: solvers._jax_common

## The XLA FFI shim

::: solvers._jax_numba_custom_call
    options:
      members:
        - CudaLaunch
        - register_target
        - compile_kernel
        - make_launch
        - ffi_abi_call

## Generated device source

::: solvers._codegen
