## libnano

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![ubuntu-builds](https://github.com/accosmin-org/libnano/actions/workflows/deploy_ubuntu.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/deploy_ubuntu.yml)
[![macos-builds](https://github.com/accosmin-org/libnano/actions/workflows/deploy_macos.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/deploy_macos.yml)
[![windows-builds](https://github.com/accosmin-org/libnano/actions/workflows/deploy_windows.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/deploy_windows.yml)

[![clang-format](https://github.com/accosmin-org/libnano/actions/workflows/clang_format.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/clang_format.yml)
[![clang-tidy](https://github.com/accosmin-org/libnano/actions/workflows/clang_tidy.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/clang_tidy.yml)
[![cppcheck](https://github.com/accosmin-org/libnano/actions/workflows/cppcheck.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/cppcheck.yml)
[![memcheck](https://github.com/accosmin-org/libnano/actions/workflows/memcheck.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/memcheck.yml)
[![docs](https://github.com/accosmin-org/libnano/actions/workflows/docs.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/docs.yml)

[![sanitizers](https://github.com/accosmin-org/libnano/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/sanitizers.yml)
[![coverage](https://github.com/accosmin-org/libnano/actions/workflows/coverage.yml/badge.svg)](https://github.com/accosmin-org/libnano/actions/workflows/coverage.yml)
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=libnano&metric=alert_status)](https://sonarcloud.io/summary/overall?id=libnano)
[![codecov](https://codecov.io/gh/accosmin-org/libnano/graph/badge.svg?token=X2IkpkoQEB)](https://codecov.io/gh/accosmin-org/libnano)


### Description

Libnano implements parameter-free and flexible machine learning algorithms complemented by an extensive collection of numerical optimization algorithms. The implementation is `cross-platform` (Linux, macOS, Windows) with `minimal dependencies` (standard library and [Eigen3](https://eigen.tuxfamily.org)) and it follows recent C++ standards and core guidelines. The library uses modern [CMake](https://cmake.org/) and as such it is easy to install and to package.


### Numerical optimization module

The library implements `state-of-the-art algorithms for both unconstrained and constrained numerical optimization problems`. Additionally builtin test functions of varying number of dimensions are provided for benchmarking these algorithms. Some of these test functions are specific to ML applications like logistic regression or multivariate linear regression with various loss functions and synthetic data.

Examples:

| Algorithm | Application |
| --------- | ----------- |
| `L-BFGS` | unconstrained smooth nonlinear optimization |
| quasi-Newton methods (e.g. `BFGS`) | unconstrained smooth nonlinear optimization |
| non-linear conjugate gradient descent (CGD) methods | unconstrained smooth nonlinear optimization |
| optimal sub-gradient algorithm (`OSGA`) | unconstrained smooth/non-smooth nonlinear optimization |
| primal-dual interior-point method | `linear and quadratic programs` |
| penalty methods | constrained nonlinear optimization |
| `augmented lagrangian` method | constrained nonlinear optimization |


### Machine learning module

The machine learning (ML) module is designed to as generic and as customizable as possible. As such various important ML concepts (e.g. loss function, hyper-parameter tuning strategy, numerical optimization solver, dataset splitting strategy, feature generation, weak learner) are modelled using appropriate orthogonal interfaces which can be extended by the user to particular machine learning applications. Additionally the implementation follows strictly the scientific principles of statistical learning to properly tune and evaluate the ML models.

In particular the following requirements were considered when designing the API:

* all ML models should work with `any feature type` (e.g. categorical, continuous or structures) and even with `missing feature values`.

* all features should be labeled with a meaningful name, a type (e.g. categorical, continuous), shape (if applicable) and labels (if applicable). This is important for model analysis (e.g. feature importance) and debugging, for designing appropriate feature selection methods (e.g. weak learners) and feature generation.

* a ML dataset must be able to handle efficiently `arbitrary sets of features` (e.g. categorical. continuous, structured like images or time series). The feature values can be optional (missing) and of different storage (e.g. signed or unsigned integers of various byte sizes, single or double precision floating point numbers). Additional features can be constructed on the fly or cached by implementing the appropriate interface.

* all ML models should work with `any loss function`. This is modeled using an appropriate interface which can be extended by the user. For example two of the most used ML models, like linear models and gradient boosting, are easy to extend to any loss function.

* the regularization `hyper-parameters are automatically tuned` using standard model evaluation protocols (e.g. cross-validation). The values of the hyper-parameters are fixed a-priori to reduce the risk of overfitting the validation dataset or of adjusting the parameter grid based on the results on the test dataset. The user can override the tuning strategy (e.g. local search) and the evaluation protocol (e.g. cross-validation, bootstrapping) using appropriate interfaces. For example the builtin parameter grid is adapted to standard regularization methods of linear models (e.g. like in lasso, ridge, elastic net). Note that models with few hyper-parameters are prefered as they are simpler to understand and tune.

* `gradient boosting models should work with arbitrary loss functions and weak learners`. Standard weak learners (e.g. decision trees, decision stumps, lool-up-tables, linear models) are builtin. The user can implement new weak learners using the appropriate interface.

Note that the library makes heavy use of its own implementation of `tensors of arbitrary rank and scalar type` designed for machine learning applications. This uses Eigen3 as backend and as such fast and easy-to-use linear algebra operations are readily available.

Examples:

| Interface | Some of builtin implementations |
| --------- | --------------------------------- |
| loss function | hinge, logistic, mse |
| dataset splitting | bootstrapping, k-fold cross-validation |
| hyper-parameter tuning | local search, quadratic model |
| weak learner | look-up table, decision stump, decision tree |
| feature generation | 2D gradient, product |



### Documentation

1. [Introduction](docs/intro.md)

2. [Tensor module](docs/tensor.md)

3. [Nonlinear unconstrained optimization](docs/nonlinear.md)

4. [Linear and quadratic programming](docs/program.md)

5. [Constrained optimization module - TODO](docs/constrained.md)

6. [Machine learning module - TODO](docs/mlearn.md)

7. [Linear models - TODO](docs/linear.md)

8. [Gradient boosting models - TODO](docs/gboost.md)
