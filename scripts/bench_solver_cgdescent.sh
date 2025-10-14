#!/usr/bin/env bash

# Experiment:
#   CG_DESCENT line-search algorithm allows high accuracy optimization (1e-14) for smooth unconstrained optimization.
#
# Expectation:
#   All solvers fail much less often (to achieve this high precision) with CG_DESCENT compared
#   to other line-search methods (e.g. More & Thuente which is close to state-of-the-art).

BENCH=./build/libnano/gcc-release/app/bench_solver

${BENCH} \
    --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --trials 20 --max-table-name 32 \
    --solver "cgd-pr|lbfgs|bfgs|hoshino|fletcher|newton" \
    --lsearchk "morethuente|cgdescent" \
    --solver::epsilon 1e-14 --solver::max_evals 10000 | tail -n 16
