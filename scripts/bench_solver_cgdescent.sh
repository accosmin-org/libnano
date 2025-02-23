#!/usr/bin/env bash

./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "cgd-pr|lbfgs|bfgs" \
    --lsearchk "morethuente|cgdescent" \
    --trials 128 --solver::epsilon 1e-14 --solver::max_evals 10000 | tail -n 10
