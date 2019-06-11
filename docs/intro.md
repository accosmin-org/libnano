# Introduction


The purpose of this tiny (nano) library is to learn while doing about various topics like:

* C++ programming - new standards are published every 3 years and guidelines are updated even more frequent.

* API design - clear implementation without sacrificing extensibility and efficiency.

* [CMake](https://cmake.org/) - modern CMake is more explicit in managing and propagating dependencies.

* continuous integration tools - good software engineers use routinely tools like valgrind, cppcheck, clang-tidy, ctest, gcov or sanitizers to improve their code.

* numerical optimization - because most machine learning algorithms need to find the optimal parameters. A good understanding of the convergence properties of various batch and stochastic optimization methods is crucial to choosing and to tuning the right method for a given task.

* machine learning - implementing linear models, gradient boosting or convolution networks from scratch is a better learning experience than using directly a high level deep learning library.


Libnano is designed around several key concepts that are mapped to specific interfaces detailed in the next sections. There exists a factory for each interface that allows the retrieval of builtin implementations and the registration of new ones. Each instance is fully configurable using [JSON](https://json.org/).


Another important design choice is to use [Eigen3](https://eigen.tuxfamily.org) for most numerical operations. The implementation of various numerical optimization and machine learning algorithms is thus efficient, compact and clear.
