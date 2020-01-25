#!/usr/bin/env bash

export PATH="/usr/local/opt/llvm/bin/:/usr/local/opt/llvm/share/clang:$PATH"
export CXX=/usr/local/opt/llvm/bin/clang++
bash scripts/build.sh --build-type RelWithDebInfo --suffix clang-tidy --config --clang-tidy-clang-analyzer
