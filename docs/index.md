# menodax

--8<-- "README.md:overview"

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting started](getting-started.md)**

    Install it, and run the first solve.

-   :material-function-variant: **[Calling a solver](guide/api.md)**

    `solve(...)`, its arguments, and what the callbacks have to look like.

-   :material-grid: **[Sparse systems](guide/sparse.md)**

    One `sparsity` pattern buys both a coloured Jacobian and a compiled sparse
    direct solve.

-   :material-chart-line: **[Gradients](guide/gradients.md)**

    `jax.grad` and friends, through a continuous forward-sensitivity solve.

</div>

--8<-- "README.md:solvers"
