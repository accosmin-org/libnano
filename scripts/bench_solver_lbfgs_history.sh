#!/usr/bin/env bash

# Experiment:
#   Compare limited-memory BFGS (L-BFGS) solver's performance relative to the history size on smooth unconstrained optimization.
#
# Expectation:
#   The impact of the history size should saturate relatively fast (10-20).

BENCH=./build/libnano/gcc-release/app/bench_solver

${BENCH} \
    --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "lbfgs|hoshino" \
    --solver::lbfgs::history 5 \
    --trials 20 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 6
for hsize in 10 20 50 100 150 200
do
    ${BENCH} \
        --min-dims 100 --max-dims 100 --function-type convex-smooth \
        --solver "lbfgs|hoshino" \
        --solver::lbfgs::history ${hsize} \
        --trials 20 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 3
done
