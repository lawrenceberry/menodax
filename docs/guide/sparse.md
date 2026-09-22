--8<-- "README.md:sparse"

## Where the pieces live

| Module | What it does |
|--------|--------------|
| [`modax._sparsity`][] | colours the pattern's column intersection graph and defines `CompressedJacobian`, the layout the Enzyme sweeps write into |
| [`modax._sparse_direct`][] | orders the pattern, factorises it symbolically, and compiles the sparse LU and triangular solves for that structure |

Both are documented under [API reference / Internals](../api/internals.md).
