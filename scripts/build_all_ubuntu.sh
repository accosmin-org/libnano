#!/usr/bin/env bash

set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

CXX=g++ bash scripts/build.sh --suffix cppcheck --config --cppcheck

CXX=g++ GCOV=gcov bash scripts/build.sh --suffix coverage --build-type RelWithDebInfo \
    --generator Ninja --coverage --config --build --test --codecov

CXX=g++ bash scripts/build.sh --suffix gcc-debug --build-type Debug \
    --generator Ninja --config --build --test --install --build-example

CXX=g++ bash scripts/build.sh --suffix gcc-release --build-type Release --native \
    --generator Ninja --config --build --test --install --build-example

CXX=g++ bash scripts/build.sh --suffix gcc-release-lto --build-type Release --native --lto \
    --generator Ninja --config --build --test --install --build-example

CXX=g++ bash scripts/build.sh --suffix memcheck --build-type RelWithDebInfo \
    --generator Ninja --config --build --memcheck

CXX=g++ bash scripts/build.sh --suffix gcc-asan --build-type Debug --asan \
    --generator Ninja --config --build --test

CXX=g++ bash scripts/build.sh --suffix gcc-lsan --build-type Debug --lsan \
    --generator Ninja --config --build --test

CXX=g++ bash scripts/build.sh --suffix gcc-usan --build-type Debug --usan \
    --generator Ninja --config --build --test

CXX=g++ bash scripts/build.sh --suffix gcc-tsan --build-type Debug --tsan \
    --generator Ninja --config --build --test

CXX=clang++ bash scripts/build.sh --suffix clang-tidy \
    --generator Ninja --config --build --clang-tidy-all

CXX=clang++ bash scripts/build.sh --suffix clang-debug --build-type Debug \
    --generator Ninja --config --build --test --install --build-example

CXX=clang++ bash scripts/build.sh --suffix clang-release --build-type Release --native \
    --generator Ninja --config --build --test --install --build-example

CXX=clang++ bash scripts/build.sh --suffix clang-release-thinlto --build-type Release --native --thinlto \
    --generator Ninja --config --build --test --install --build-example