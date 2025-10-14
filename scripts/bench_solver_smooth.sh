#!/usr/bin/env bash

# Experiment:
#   Compare solvers for smooth unconstrained optimization.
#
# Expectation:
#   Gradient descent (GD) takes many more iterations than the rest to converge (if at all).
#   Nonlinear conjugate gradient descent (CGD), L-BFGS and quasi-Newton methods should converge relatively fast for all problems.
#   Truncated Newton method should be the fastest (in number of iterations), but it cannot handle very large problems.
#   All methods expect GD should solve these problems with moderate accuracy and in a reasonable number of iterations.

BENCH=./build/libnano/gcc-release/app/bench_solver

${BENCH} \
    --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "gd|cgd-pr|lbfgs|bfgs|hoshino|fletcher|newton" \
    --trials 20 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 11
