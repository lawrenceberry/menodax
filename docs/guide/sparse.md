--8<-- "README.md:sparse"

## Where the pieces live

| Module | What it does |
|--------|--------------|
| [`solvers._sparsity`][] | colours the pattern's column intersection graph and defines `CompressedJacobian`, the layout the Enzyme sweeps write into |
| [`solvers._sparse_direct`][] | orders the pattern, factorises it symbolically, and compiles the sparse LU and triangular solves for that structure |

Both are documented under [API reference / Internals](../api/internals.md).
