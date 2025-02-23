#!/usr/bin/env bash

./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "lbfgs|bfgs" \
    --solver::lbfgs::history 5 \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 6
for hsize in 10 20 50 100 150 200
do
    ./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex-smooth \
        --solver "lbfgs|bfgs" \
        --solver::lbfgs::history ${hsize} \
        --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 3
done
