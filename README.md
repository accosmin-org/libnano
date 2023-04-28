# libnano

[![ubuntu-builds](https://github.com/accosmin-org/libnano/actions/workflows/deploy_ubuntu.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/deploy_ubuntu.yml)
[![macos-builds](https://github.com/accosmin-org/libnano/actions/workflows/deploy_macos.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/deploy_macos.yml)
[![windows-builds](https://github.com/accosmin-org/libnano/actions/workflows/deploy_windows.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/deploy_windows.yml)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![clang-format](https://github.com/accosmin-org/libnano/actions/workflows/clang_format.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/clang_format.yml)
[![clang-tidy](https://github.com/accosmin-org/libnano/actions/workflows/clang_tidy.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/clang_tidy.yml)
[![cppcheck](https://github.com/accosmin-org/libnano/actions/workflows/cppcheck.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/cppcheck.yml)

[![memcheck](https://github.com/accosmin-org/libnano/actions/workflows/memcheck.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/memcheck.yml)
[![sanitizers](https://github.com/accosmin-org/libnano/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/sanitizers.yml)
[![coverage](https://github.com/accosmin-org/libnano/actions/workflows/coverage.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/coverage.yml)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=libnano&metric=alert_status)](https://sonarcloud.io/summary/overall?id=libnano)


This library provides numerical optimization routines and machine learning utilities such as:

* state-of-the-art efficient optimization methods for smooth and non-smooth convex problems (e.g. L-BFGS, quasi Newton methods, non-linear conjugate gradient descent - CGD, OSGA).

* tensors of arbitrary rank designed for machine learning applications. The implementation is using [Eigen3](https://eigen.tuxfamily.org) and as such fast and easy-to-use linear algebra operations are readily available.

* efficient in-memory storage of machine learning datasets of mixed features (categorical or continuous, structured or not). The feature values can be optional.

* linear models with standard regularization ethods (e.g. like in lasso, ridge, elastic net) and arbitrary loss functions. Any feature type (e.g. categorical, images, scalars) and missing feature values are supported as well.

* gradient boosting models with arbitray loss functions and arbitrary weak learners (e.g. decision stumps, linear models). Any feature type (e.g. categorical, images, scalars) and missing feature values are supported as well.


The implementation is cross-platform (Linux, macOS, Windows) with minimal dependencies (standard library and [Eigen3](https://eigen.tuxfamily.org) and it follows recent C++ standards and core guidelines. The library uses modern [CMake](https://cmake.org/) and as such it is easy to install and to package.


The following sections provide more details regarding each relevant topic:

1. [Build instructions](docs/build.md)

2. [Tensor module](docs/tensor.md)

3. [Numerical optimization module](docs/solver.md)

4. [Machine learning module](docs/mlearn.md)

5. [Linear models - TODO](docs/linear.md)

6. [Gradient boosting models - TODO](docs/gboost.md)
