#!/usr/bin/env bash

set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

cmake_options="-GNinja" #-DCMAKE_CXX_INCLUDE_WHAT_YOU_USE=iwyu"

export CXXFLAGS="${CXXFLAGS} -march=x86-64-v3 -Og"

###############################################################################################################
# code formatting using a fixed version of clang-format
###############################################################################################################

CXX=clang++ bash scripts/build.sh --clang-format

###############################################################################################################
# check documentation
###############################################################################################################

bash scripts/build.sh --check-markdown-docs

###############################################################################################################
# standard GCC builds
###############################################################################################################

CXX=g++ bash scripts/build.sh --suffix gcc-debug -DCMAKE_BUILD_TYPE=Debug \
    ${cmake_options} --config --build --test --install --build-example

CXX=g++ bash scripts/build.sh --suffix gcc-release -DCMAKE_BUILD_TYPE=Release \
    ${cmake_options} --config --build --test --install --build-example

CXX=g++ bash scripts/build.sh --suffix gcc-relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    ${cmake_options} --config --build --test --install --build-example

#CXX=g++ bash scripts/build.sh --suffix gcc-release-lto -DCMAKE_BUILD_TYPE=Release --lto \
#    ${cmake_options} --config --build --test --install --build-example

###############################################################################################################
# standard CLANG builds
###############################################################################################################

CXX=clang++ bash scripts/build.sh --suffix clang-debug -DCMAKE_BUILD_TYPE=Debug \
    ${cmake_options} --config --build --test --install --build-example

CXX=clang++ bash scripts/build.sh --suffix clang-release -DCMAKE_BUILD_TYPE=Release \
    ${cmake_options} --config --build --test --install --build-example

CXX=clang++ bash scripts/build.sh --suffix clang-release-thinlto -DCMAKE_BUILD_TYPE=Release --thinlto \
    ${cmake_options} --config --build --test --install --build-example

###############################################################################################################
# static analysis:
#   - cppcheck
#   - clang-tidy
###############################################################################################################

CXX=g++ bash scripts/build.sh --suffix cppcheck \
    ${cmake_options} --config --cppcheck

CXX=clang++ bash scripts/build.sh --suffix clang-tidy \
    ${cmake_options} --config --build --clang-tidy-all

###############################################################################################################
# dynamic analysis:
#   - memcheck
#   - sanitizers
###############################################################################################################

CXX=g++ bash scripts/build.sh --suffix memcheck -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    ${cmake_options} --config --build --memcheck

CXX=clang++ bash scripts/build.sh --suffix clang-asan -DCMAKE_BUILD_TYPE=Debug --asan \
    ${cmake_options} --config --build --test

CXX=clang++ bash scripts/build.sh --suffix clang-lsan -DCMAKE_BUILD_TYPE=Debug --lsan \
    ${cmake_options} --config --build --test

CXX=clang++ bash scripts/build.sh --suffix clang-usan -DCMAKE_BUILD_TYPE=Debug --usan \
    ${cmake_options} --config --build --test

CXX=clang++ bash scripts/build.sh --suffix clang-tsan -DCMAKE_BUILD_TYPE=Debug --tsan \
    ${cmake_options} --config --build --test

###############################################################################################################
# code coverage:
#   - lcov + genhtml
#   - llvm-cov + sonar scanner
###############################################################################################################

rm -rf lcov* build/libnano/lcov

CXX=g++ GCOV=gcov bash scripts/build.sh --suffix lcov -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    --coverage -DBUILD_SHARED_LIBS=OFF ${cmake_options} --config --build --lcov-init --test --lcov
    --codecov

CXX=clang++ bash scripts/build.sh --suffix llvm-lcov -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    --llvm-coverage --libcpp -DNANO_ENABLE_LLVM_COV=ON ${cmake_options} --config --build --test --llvm-cov \
    --sonar
