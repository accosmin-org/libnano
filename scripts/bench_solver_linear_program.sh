#!/usr/bin/env bash

./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type linear-program \
    --solver "ipm|augmented-lagrangian|quadratic-penalty" --trials 128 | tail -n 7
