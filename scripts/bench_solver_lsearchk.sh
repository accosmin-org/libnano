#!/usr/bin/env bash

# Experiment:
#   Compare line-search algorithms for various solvers on smooth unconstrained optimization.
#
# Expectation:
#   The truncated Newton method is very little impacted by the line-search method as it converging quadratically.
#   The first-order methods should not fail and be the fastest with CG_DESCENT and More&Thuente line-search methods.
#   The backtracking line-search method can be very slow, but it should never fail.
#   The Fletcher and the LeMarechal line-search metholds can fail as they don't have any convergence proof.

BENCH=./build/libnano/gcc-release/app/bench_solver

${BENCH} \
    --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "cgd-pr|lbfgs|bfgs|hoshino|fletcher|newton" \
    --lsearchk ".+" \
    --trials 20 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 34
