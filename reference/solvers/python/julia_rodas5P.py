"""Julia Rodas5P reference solver via DiffEqGPU."""

from reference.solvers.python.julia_common import make_julia_solver

solve, solve_with_timing = make_julia_solver("rodas5P")
