## libnano

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![ubuntu-builds](https://github.com/accosmin-org/libnano/actions/workflows/deploy_ubuntu.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/deploy_ubuntu.yml)
[![macos-builds](https://github.com/accosmin-org/libnano/actions/workflows/deploy_macos.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/deploy_macos.yml)
[![windows-builds](https://github.com/accosmin-org/libnano/actions/workflows/deploy_windows.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/deploy_windows.yml)

[![clang-format](https://github.com/accosmin-org/libnano/actions/workflows/clang_format.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/clang_format.yml)
[![clang-tidy](https://github.com/accosmin-org/libnano/actions/workflows/clang_tidy.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/clang_tidy.yml)
[![cppcheck](https://github.com/accosmin-org/libnano/actions/workflows/cppcheck.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/cppcheck.yml)
[![memcheck](https://github.com/accosmin-org/libnano/actions/workflows/memcheck.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/memcheck.yml)
[![sanitizers](https://github.com/accosmin-org/libnano/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/sanitizers.yml)
[![coverage](https://github.com/accosmin-org/libnano/actions/workflows/coverage.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/coverage.yml)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=libnano&metric=alert_status)](https://sonarcloud.io/summary/overall?id=libnano)

## Description

Libnano implements parameter-free and flexible machine learning algorithms complemented by an extensive collection of numerical optimization algorithms. The implementation is cross-platform (Linux, macOS, Windows) with minimal dependencies (standard library and [Eigen3](https://eigen.tuxfamily.org)) and it follows recent C++ standards and core guidelines. The library uses modern [CMake](https://cmake.org/) and as such it is easy to install and to package.


In particular:

* **state-of-the-art first-order optimization algorithms** for smooth and non-smooth convex problems (e.g. L-BFGS, quasi Newton methods, non-linear conjugate gradient descent - CGD, optimal sub-gradient algorithm - OSGA). Additionally the library provides many builtin test functions of varying number of dimensions (some specific to ML applications) useful for benchmarking these algorithms.

* **tensors of arbitrary rank and scalar type** designed for machine learning applications. The implementation is using [Eigen3](https://eigen.tuxfamily.org) and as such fast and easy-to-use linear algebra operations are readily available.

* **efficient in-memory machine learning dataset containing potentially optional arbitrary features** - categorical, continuous, structured (images, time series). This data can be easily used to engineer arbitrary features on the fly as well. The generated features are compatible with all provided machine learning algorithms.

* **parameter-free machine learning algorithms** - all hyper-parameters are tuned automatically following principled approaches and the numerical optimization is performed precisely using parameter-free state-of-the-art methods. Additionally the following aspects are modeled with interfaces and can be customized by the user: the loss function, the numerical optimization method, the splitting strategy (e.g. cross-validation) and the tuning strategy (e.g. local search, surrogate minimization).

* **linear models** with standard regularization methods (e.g. lasso, ridge, elastic net). 

* **gradient boosting models** with arbitrary weak learners (e.g. decision trees, look-up-tables, linear learners).

* **statistical analysis tools** to evaluate and compare models - TODO.


## Documentation

1. [Build instructions](docs/build.md)

2. [Tensor module](docs/tensor.md)

3. [Numerical optimization module](docs/solver.md)

4. [Machine learning module](docs/mlearn.md)

5. [Linear models - TODO](docs/linear.md)

6. [Gradient boosting models - TODO](docs/gboost.md)
