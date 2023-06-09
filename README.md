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
[![sonar](https://github.com/accosmin-org/libnano/actions/workflows/sonar.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/sonar.yml)
[![codecov](https://github.com/accosmin-org/libnano/actions/workflows/codecov.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/codecov.yml)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=libnano&metric=alert_status)](https://sonarcloud.io/summary/overall?id=libnano)
[![codecov](https://codecov.io/gh/accosmin-org/libnano/branch/master/graph/badge.svg?token=X2IkpkoQEB)](https://codecov.io/gh/accosmin-org/libnano)

## Description

Libnano implements parameter-free and flexible machine learning algorithms complemented by an extensive collection of numerical optimization algorithms. The implementation is cross-platform (Linux, macOS, Windows) with minimal dependencies (standard library and [Eigen3](https://eigen.tuxfamily.org)) and it follows recent C++ standards and core guidelines. The library uses modern [CMake](https://cmake.org/) and as such it is easy to install and to package.


In particular:

* **state-of-the-art first-order optimization algorithms** for smooth and non-smooth convex problems (e.g. L-BFGS, quasi Newton methods, non-linear conjugate gradient descent - CGD, optimal sub-gradient algorithm - OSGA). Additionally the library provides many builtin test functions of varying number of dimensions (some specific to ML applications) useful for benchmarking these algorithms.

* **tensors of arbitrary rank and scalar type** designed for machine learning applications. The implementation is using [Eigen3](https://eigen.tuxfamily.org) and as such fast and easy-to-use linear algebra operations are readily available.

* **efficient in-memory storage of machine learning datasets of mixed features** (e.g. categorical. continuous, structured like images or time series). The feature values can be optional and of different storage (e.g. signed or unsigned integers of various sizes, single or double precision floating point numbers). Additional features can be constructed on the fly to be used for training and evaluating machine learning models.

* **linear models with arbitrary loss functions**. Standard regularization methods (e.g. like in lasso, ridge, elastic net) are builtin.

* **gradient boosting models with arbitray loss functions and arbitrary weak learners**. Standard weak learners (e.g. decision trees, decision stumps, lool-up-tables, linear models) are builtin.

* all machine learning models work with any feature type (e.g. categorical, continuous or structures) and potentially missing feature values.

* the regularization hyper-parameters are automatically tuned using standard model evaluation (e.g. cross-validation).

* the machine learning concepts (e.g. loss function, hyper-parameter tuning strategy, numerical optimization solver, splitting strategy, feature generation) are modelled using appropriate interfaces. As such the library is highly customizable to particular machine learning applications.


The implementation is **cross-platform** (Linux, macOS, Windows) with **minimal dependencies** (standard library and [Eigen3](https://eigen.tuxfamily.org) and it follows recent C++ standards and core guidelines. The library uses modern [CMake](https://cmake.org/) and as such it is easy to install and to package.


## Documentation

1. [Build instructions](docs/build.md)

2. [Tensor module](docs/tensor.md)

3. [Numerical optimization module](docs/solver.md)

4. [Machine learning module](docs/mlearn.md)

5. [Linear models - TODO](docs/linear.md)

6. [Gradient boosting models - TODO](docs/gboost.md)
