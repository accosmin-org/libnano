#!/usr/bin/env bash

set -ex

cmake_options="-GNinja" #-DCMAKE_CXX_INCLUDE_WHAT_YOU_USE=iwyu"

export CXXFLAGS="${CXXFLAGS} -march=x86-64-v3 -Og"

###############################################################################################################
# generic checks:
#   - check bash scripts
#   - code formatting using a fixed version of clang-format
#   - code structure
#   - documentation
###############################################################################################################

bash docker/run.sh bash scripts/build.sh \
    --shellcheck \
    --clang-format \
    --check-source-files \
    --check-markdown-docs

###############################################################################################################
# standard GCC builds
###############################################################################################################

bash scripts/build.sh --gcc --suffix gcc-debug -DCMAKE_BUILD_TYPE=Debug \
    ${cmake_options} --config --build --test --install --build-example

bash scripts/build.sh --gcc --suffix gcc-release -DCMAKE_BUILD_TYPE=Release \
    ${cmake_options} --config --build --test --install --build-example

#bash scripts/build.sh --gcc --suffix gcc-relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#    ${cmake_options} --config --build --test --install --build-example

#bash scripts/build.sh -gcc --suffix gcc-release-lto -DCMAKE_BUILD_TYPE=Release --lto \
#    ${cmake_options} --config --build --test --install --build-example

###############################################################################################################
# standard CLANG builds
###############################################################################################################

bash scripts/build.sh --clang --suffix clang-debug -DCMAKE_BUILD_TYPE=Debug \
    ${cmake_options} --config --build --test --install --build-example

bash scripts/build.sh --clang --suffix clang-release -DCMAKE_BUILD_TYPE=Release \
    ${cmake_options} --config --build --test --install --build-example

#bash scripts/build.sh --clang --suffix clang-release-thinlto -DCMAKE_BUILD_TYPE=Release --thinlto \
#    ${cmake_options} --config --build --test --install --build-example

###############################################################################################################
# static analysis:
#   - cppcheck
#   - clang-tidy
###############################################################################################################

# bash docker/run.sh bash scripts/build.sh --gcc --suffix cppcheck \
#    ${cmake_options} --config --cppcheck

bash docker/run.sh bash scripts/build.sh --clang --suffix clang-tidy \
    ${cmake_options} --config --build --clang-tidy-all

###############################################################################################################
# dynamic analysis:
#   - memcheck
#   - sanitizers
###############################################################################################################

bash scripts/build.sh --clang --suffix memcheck -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    ${cmake_options} --config --build --memcheck

bash scripts/build.sh --clang --suffix clang-asan -DCMAKE_BUILD_TYPE=Debug --asan \
    ${cmake_options} --config --build --test

bash scripts/build.sh --clang --suffix clang-lsan -DCMAKE_BUILD_TYPE=Debug --lsan \
    ${cmake_options} --config --build --test

bash scripts/build.sh --clang --suffix clang-usan -DCMAKE_BUILD_TYPE=Debug --usan \
    ${cmake_options} --config --build --test

bash scripts/build.sh --clang --suffix clang-tsan -DCMAKE_BUILD_TYPE=Debug --tsan \
    ${cmake_options} --config --build --test

###############################################################################################################
# code coverage:
#   - lcov + genhtml
#   - llvm-cov + sonar scanner
###############################################################################################################

rm -rf lcov* build/libnano/lcov

GCOV=gcov bash scripts/build.sh --gcc --suffix lcov -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    --coverage -DBUILD_SHARED_LIBS=OFF ${cmake_options} --config --build --lcov-init --test --lcov \
    --codecov

bash scripts/build.sh --clang --suffix llvm-lcov -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    --llvm-coverage --libcpp -DNANO_ENABLE_LLVM_COV=ON ${cmake_options} --config --build --test --llvm-cov \
    --sonar
