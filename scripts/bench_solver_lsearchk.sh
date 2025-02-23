#!/usr/bin/env bash

./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "lbfgs|bfgs" \
    --lsearchk ".+" \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 14
