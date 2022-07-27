#!/usr/bin/env bash

set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

###############################################################################################################
# code formatting using a fixed version of clang-format
###############################################################################################################

CXX=clang++-13 bash scripts/build.sh --clang-suffix -13 --clang-format

###############################################################################################################
# static analysis:
#   - cppcheck
#   - clang-tidy
###############################################################################################################

CXX=g++ bash scripts/build.sh --suffix cppcheck \
    -GNinja --config --cppcheck

CXX=clang++ bash scripts/build.sh --suffix clang-tidy \
    -GNinja --config --build --clang-tidy-all

###############################################################################################################
# code coverage:
#   - lcov + genhtml
#   - llvm-cov
###############################################################################################################

CXX=g++ GCOV=gcov bash scripts/build.sh --suffix lcov -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -GNinja --coverage -DBUILD_SHARED_LIBS=OFF --config --build --test --lcov

CXX=clang++ bash scripts/build.sh --suffix llvm-lcov -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -GNinja --llvm-coverage --libcpp -DNANO_ENABLE_LLVM_COV=ON --config --build --test --llvm-cov

###############################################################################################################
# standard debug/release/release+lto GCC builds
###############################################################################################################

CXX=g++ bash scripts/build.sh --suffix gcc-debug -DCMAKE_BUILD_TYPE=Debug \
    -GNinja --config --build --test --install --build-example

CXX=g++ bash scripts/build.sh --suffix gcc-release -DCMAKE_BUILD_TYPE=Release --native \
    -GNinja --config --build --test --install --build-example

#CXX=g++ bash scripts/build.sh --suffix gcc-release-lto -DCMAKE_BUILD_TYPE=Release --native --lto \
#    -GNinja --config --build --test --install --build-example

###############################################################################################################
# standard debug/release/release+lto CLANG builds
###############################################################################################################

CXX=clang++ bash scripts/build.sh --suffix clang-debug -DCMAKE_BUILD_TYPE=Debug \
    -GNinja --config --build --test --install --build-example

CXX=clang++ bash scripts/build.sh --suffix clang-release -DCMAKE_BUILD_TYPE=Release --native \
    -GNinja --config --build --test --install --build-example

#CXX=clang++ bash scripts/build.sh --suffix clang-release-thinlto -DCMAKE_BUILD_TYPE=Release --native --thinlto \
#    -GNinja --config --build --test --install --build-example

###############################################################################################################
# dynamic analysis:
#   - memcheck
#   - sanitizers
###############################################################################################################

CXX=g++ bash scripts/build.sh --suffix memcheck -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -GNinja --config --build --memcheck

CXX=clang++ bash scripts/build.sh --suffix clang-asan -DCMAKE_BUILD_TYPE=Debug --asan \
    -GNinja --config --build --test

CXX=clang++ bash scripts/build.sh --suffix clang-lsan -DCMAKE_BUILD_TYPE=Debug --lsan \
    -GNinja --config --build --test

CXX=clang++ bash scripts/build.sh --suffix clang-usan -DCMAKE_BUILD_TYPE=Debug --usan \
    -GNinja --config --build --test

CXX=clang++ bash scripts/build.sh --suffix clang-tsan -DCMAKE_BUILD_TYPE=Debug --tsan \
    -GNinja --config --build --test
