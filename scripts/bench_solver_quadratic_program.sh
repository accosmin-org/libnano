#!/usr/bin/env bash

BENCH=./build/libnano/gcc-release/app/bench_solver

${BENCH} \
    --min-dims 100 --max-dims 100 --function-type quadratic-program \
    --solver "ipm|augmented-lagrangian|quadratic-penalty" --trials 100 | tail -n 7
