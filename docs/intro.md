# Introduction


#### Structure

The project is organized as follows:
* [include/nano](../include/nano) - C++ headers files representing the library's interface.
* [src](../src) - C++ source files with implementation details.
* [app](../app) - command line C++ utilities mostly used for benchmarking various components.
* [test](../test) - C++ unit tests run with [ctest](https://cmake.org/cmake/help/v3.16/manual/ctest.1.html#ctest-1).
* [docs](../docs) - documentation written in Markdown.
* [cmake](../cmake) - various CMake utilities (e.g. to simplify building unit tests).
* [docker](../docker) - Docker images to mirror the CI/CD enviroment for reproducible builds (e.g. clang-format, clang-tidy).
* [scripts](../scripts) - various Bash utilities (e.g. to simplify configuring and building of the library, to download datasets or to run experiments).
* [example](../example) - example programs used for testing the setup of the library and for showcasing some of its functionality.


#### Dependencies

System:
* compiler supporting [C++20](https://isocpp.org/get-started)
* [CMake](https://cmake.org)
* [Eigen3](https://eigen.tuxfamily.org) - high-performance linear-algebra C++ library

Libnano is tested on Linux (Ubuntu, Fedora) using the system GCC and Clang compilers, on MacOS using AppleClang and on Windows using Visual Studio. It may work with minor changes on other platforms as well.


#### How to build

The easiest way to build, test and install the library on Linux, MacOS or Windows is to call either [scripts/build.sh](../scripts/build.sh) or [scripts/build.bat](../scripts/build.bat) with the appropriate command line arguments. These scripts are invoked to run various tests on continuous integrations plaforms like [github workflows](https://github.com/accosmin-org/libnano/actions). See [.github/workflows](../.github/workflows) for examples.


Otherwise, users can also invoke the main CMake script directly for other platforms or for custom builds.


The following command displays the command line arguments supported by [scripts/build.sh](../scripts/build.sh):
```
usage: scripts/build.sh [OPTIONS]

options:
    -h,--help
        print usage
    --gcc
        setup g++ compiler
    --clang
        setup clang++ compiler
    --lld
        setup compiler and linker flags to enable the llvm linker
    --lto
        setup compiler and linker flags to enable link-time optimization
    --thinlto
        setup compiler and linker flags to enable link-time optimization with parallelization
    --asan
        setup compiler and linker flags to enable the address sanitizer
    --lsan
        setup compiler and linker flags to enable the leak sanitizer
    --usan
        setup compiler and linker flags to enable the undefined behaviour sanitizer
    --tsan
        setup compiler and linker flags to enable the thread sanitizer
    --msan
        setup compiler and linker flags to enable the memory sanitizer
    --gold
        setup compiler and linker flags to enable the gold linker
    --no-werror
        disable treating compilation warnings as errors
    --native
        setup compiler flags to optimize for the native platform
    --libcpp
        setup compiler and linker flags to use libc++
    --coverage
        setup compiler and linker flags to setup code coverage using gcov (gcc and clang)
    --llvm-coverage
        setup compiler and linker flags to setup source-based code coverage (clang)
    --suffix <string>
        suffix for the build and installation directories
    --config
        generate build using CMake
    --build
        compile the library, the unit tests and the command line applications
    --test
        run the unit tests
    --install
        install the library and the command line applications
    --build-example
        build example project
    --cppcheck
        run cppcheck (static code analyzer)
    --lcov
        generate the code coverage report using lcov and genhtml
    --lcov-init
        generate the initial code coverage report using lcov and genhtml
    --llvm-cov
        generate the code coverage report usign llvm's source-based code coverage
    --memcheck
        run the unit tests through memcheck (e.g. detects unitialized variales, memory leaks, invalid memory accesses)
    --helgrind
        run the unit tests through helgrind (e.g. detects data races)
    --clang-suffix <string>
        suffix for the llvm tools like clang-tidy or clang-format (e.g. -6.0)
    --clang-tidy-check <check name>
        run a particular clang-tidy check (e.g. misc, cert)
    --clang-tidy-all
    --clang-tidy-misc
    --clang-tidy-cert
    --clang-tidy-hicpp
    --clang-tidy-bugprone
    --clang-tidy-modernize
    --clang-tidy-concurrency
    --clang-tidy-performance
    --clang-tidy-portability
    --clang-tidy-readability
    --clang-tidy-clang-analyzer
    --clang-tidy-cppcoreguidelines
    --clang-format
        check formatting with clang-format (the code will be modified in-place)
    --check-markdown-docs
        check the markdown documentation (e.g. invalid C++ includes, invalid local links)
    --check-source-files
        check the source files are used properly (e.g. unreferenced files in CMake scripts)
    --shellcheck
        check bash scripts with shellcheck
    -D[option]
        options to pass directly to cmake build (e.g. -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON)
    -G[option]
        options to pass directly to cmake build (e.g. -GNinja)
```

The order of the command line arguments matter: the configuration parameters should be first and the actions should follow in the right order (e.g. configuration, then compilation, then testing and then installation). The build script and the CMake scripts do not override environmental variables like ```CXXFLAGS``` or ```LDFLAGS``` and as such the library can be easily wrapped by a package manager.


Examples:
* build the Debug build and run the unit tests in the folder ```build/libnano/debug```:
```
bash scripts/build.sh --suffix debug -DCMAKE_BUILD_TYPE=Debug \
    --config --build --test
```

* build the natively-optimized release build in the folder ```build/libnano/release```, run the unit tests, install the library in ```install/release``` and build the examples in the folder ```build/example/release```:
```
bash scripts/build.sh --suffix release -DCMAKE_BUILD_TYPE=Release -GNinja \
    --native --config --build --test --install --build-example
```

* build the Debug build and run the unit tests in the folder ```build/libnano/debug``` with memcheck:
```
bash scripts/build.sh --suffix debug -DCMAKE_BUILD_TYPE=Debug -GNinja \
    --config --build --memcheck
```


NB: Use the ```--native``` flag when compiling for ```Release``` builds to maximize performance on a given machine. This is because Eigen3 uses vectorization internally for the linear algebra operations. Please note that the resulting binaries may not be usable on another platform.


The commands can be prefixed with `bash docker/run.sh` and they will be run inside the default docker image used by GitHub workflows.


#### Library design

The library is designed by mapping the relevant concepts and algorithms in numerical optimization and machine learning to proper interfaces. The available implementations are fully parametrizable and are all registered to extendable factories. These can be easily discovered using the builtin command line utility [app/info](../app/info.cpp) assumed in the following examples to be built in the folder ```build/libnano/gcc-release``` using one of the example above.

Examples:
* list the available numerical optimization algorithms (solvers):
```
./build/libnano/gcc-release/app/info --list-solver
|----------------------|---------------------------------------------------------------------------|
| solver               | description                                                               |
|----------------------|---------------------------------------------------------------------------|
| gd                   | gradient descent                                                          |
| newton               | truncated newton method                                                   |
| gs                   | gradient sampling (P-nNGS)                                                |
| ags                  | adaptive gradient sampling (P-nNGS + AGS)                                 |
| gs-lbfgs             | gradient sampling with LBFGS-like updates (P-nNGS + LBFGS)                |
| ags-lbfgs            | adaptive gradient sampling with LBFGS-like updates (P-nNGS + AGS + LBFGS) |
| sgm                  | sub-gradient method                                                       |
| cgd-n                | conjugate gradient descent (N+)                                           |
| cgd-hs               | conjugate gradient descent (HS+)                                          |
| cgd-fr               | conjugate gradient descent (FR)                                           |
| cgd-pr               | conjugate gradient descent (PR+)                                          |
| cgd-cd               | conjugate gradient descent (CD)                                           |
| cgd-ls               | conjugate gradient descent (LS+)                                          |
| cgd-dy               | conjugate gradient descent (DY)                                           |
| cgd-dycd             | conjugate gradient descent (DYCD)                                         |
| cgd-dyhs             | conjugate gradient descent (DYHS)                                         |
| cgd-frpr             | conjugate gradient descent (FRPR)                                         |
| osga                 | optimal sub-gradient algorithm (OSGA)                                     |
| lbfgs                | limited-memory BFGS                                                       |
| dfp                  | quasi-newton method (DFP)                                                 |
| sr1                  | quasi-newton method (SR1)                                                 |
| bfgs                 | quasi-newton method (BFGS)                                                |
| hoshino              | quasi-newton method (Hoshino formula)                                     |
| fletcher             | quasi-newton method (Fletcher's switch)                                   |
| ellipsoid            | ellipsoid method                                                          |
| asga2                | accelerated sub-gradient algorithm (ASGA-2)                               |
| asga4                | accelerated sub-gradient algorithm (ASGA-4)                               |
| cocob                | continuous coin betting (COCOB)                                           |
| sda                  | simple dual averages (variant of primal-dual subgradient methods)         |
| wda                  | weighted dual averages (variant of primal-dual subgradient methods)       |
| pgm                  | universal primal gradient method (PGM)                                    |
| dgm                  | universal dual gradient method (DGM)                                      |
| fgm                  | universal fast gradient method (FGM)                                      |
| rqb                  | reversal quasi-newton bundle algorithm (RQB)                              |
| fpba1                | fast proximal bundle algorithm (FPBA1)                                    |
| fpba2                | fast proximal bundle algorithm (FPBA2)                                    |
| ipm                  | primal-dual interior point method for linear and quadratic programs (IPM) |
| linear-penalty       | linear penalty method for constrained problems                            |
| quadratic-penalty    | quadratic penalty method for constrained problems                         |
| augmented-lagrangian | augmented lagrangian method for constrained problems                      |
|----------------------|---------------------------------------------------------------------------|
```

* list the name, the default value and the domain of all parameters of some solvers of interest:
```
./build/libnano/gcc-release/app/info --list-solver-params --solver "lbfgs|bfgs|cgd-pr"
|--------|-------------------------------|--------------|--------------------------|
| solver | parameter                     | value        | domain                   |
|--------|-------------------------------|--------------|--------------------------|
| cgd-pr | conjugate gradient descent (PR+)                                        |
|--------|-------------------------------|--------------|--------------------------|
| |...   | solver::epsilon               | 1e-08        | 0 < 1e-08 <= 0.1         |
| |...   | solver::patience              | 100          | 1 <= 100 <= 1000000      |
| |...   | solver::max_evals             | 1000         | 10 <= 1000 <= 1000000000 |
| |...   | solver::tolerance             | (0.0001,0.1) | 0 < 0.0001 < 0.1 < 1     |
| |...   | solver::cgd::orthotest        | 0.1          | 0 < 0.1 < 1              |
|--------|-------------------------------|--------------|--------------------------|
| lbfgs  | limited-memory BFGS                                                     |
|--------|-------------------------------|--------------|--------------------------|
| |...   | solver::epsilon               | 1e-08        | 0 < 1e-08 <= 0.1         |
| |...   | solver::patience              | 100          | 1 <= 100 <= 1000000      |
| |...   | solver::max_evals             | 1000         | 10 <= 1000 <= 1000000000 |
| |...   | solver::tolerance             | (0.0001,0.9) | 0 < 0.0001 < 0.9 < 1     |
| |...   | solver::lbfgs::history        | 50           | 1 <= 50 <= 1000          |
|--------|-------------------------------|--------------|--------------------------|
| bfgs   | quasi-newton method (BFGS)                                              |
|--------|-------------------------------|--------------|--------------------------|
| |...   | solver::epsilon               | 1e-08        | 0 < 1e-08 <= 0.1         |
| |...   | solver::patience              | 100          | 1 <= 100 <= 1000000      |
| |...   | solver::max_evals             | 1000         | 10 <= 1000 <= 1000000000 |
| |...   | solver::tolerance             | (0.0001,0.9) | 0 < 0.0001 < 0.9 < 1     |
| |...   | solver::quasi::initialization | identity     | identity,scaled          |
|--------|-------------------------------|--------------|--------------------------|
```

* list the available benchmark test functions useful for comparing numerical optimization methods (solvers):
```
./build/libnano/gcc-release/app/info --list-function
|------------------------|----------------------------------------------------------------------------------------------------------|
| function               | description                                                                                              |
|------------------------|----------------------------------------------------------------------------------------------------------|
| maxq                   | MAXQ function: f(x) = max(i, x_i^2)                                                                      |
| maxquad                | MAXQUAD function: f(x) = max(k, x.dot(A_k*x) - b_k.dot(x))                                               |
| maxhilb                | MAXHILB function: f(x) = max(i, sum(j, xj / (i + j = 1))                                                 |
| chained_lq             | chained LQ function (see documentation)                                                                  |
| chained_cb3I           | chained CB3 I function (see documentation)                                                               |
| chained_cb3II          | chained CB3 II function (see documentation)                                                              |
| trid                   | Trid function: https://www.sfu.ca/~ssurjano/trid.html                                                    |
| qing                   | Qing function: http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html                                       |
| kinks                  | random kinks: f(x) = sum(|x - K_i|, i)                                                                   |
| cauchy                 | Cauchy function: f(x) = log(1 + x.dot(x))                                                                |
| sargan                 | Sargan function: http://infinity77.net/global_optimization/test_functions_nd_S.html                      |
| powell                 | Powell function: https://www.sfu.ca/~ssurjano/powell.html                                                |
| sphere                 | sphere function: f(x) = x.dot(x)                                                                         |
| zakharov               | Zakharov function: https://www.sfu.ca/~ssurjano/zakharov.html                                            |
| quadratic              | random quadratic function: f(x) = x.dot(a) + x * A * x, where A is PD                                    |
| rosenbrock             | Rosenbrock function: https://en.wikipedia.org/wiki/Test_functions_for_optimization                       |
| exponential            | exponential function: f(x) = exp(1 + x.dot(x) / D)                                                       |
| dixon-price            | Dixon-Price function: https://www.sfu.ca/~ssurjano/dixonpr.html                                          |
| chung-reynolds         | Chung-Reynolds function: f(x) = (x.dot(x))^2                                                             |
| axis-ellipsoid         | axis-parallel hyper-ellipsoid function: f(x) = sum(i*x+i^2, i=1,D)                                       |
| styblinski-tang        | Styblinski-Tang function: https://www.sfu.ca/~ssurjano/stybtang.html                                     |
| schumer-steiglitz      | Schumer-Steiglitz No. 02 function: f(x) = sum(x_i^4, i=1,D)                                              |
| rotated-ellipsoid      | rotated hyper-ellipsoid function: https://www.sfu.ca/~ssurjano/rothyp.html                               |
| geometric-optimization | generic geometric optimization function: f(x) = sum(i, exp(alpha_i + a_i.dot(x)))                        |
| mse+lasso              | mean squared error (MSE) with lasso regularization                                                       |
| mae+lasso              | mean absolute error (MAE) with lasso regularization                                                      |
| hinge+lasso            | hinge loss (linear SVM) with lasso regularization                                                        |
| cauchy+lasso           | cauchy loss (robust regression) with lasso regularization                                                |
| logistic+lasso         | logistic loss (logistic regression) with lasso regularization                                            |
| mse+ridge              | mean squared error (MSE) with ridge regularization                                                       |
| mae+ridge              | mean absolute error (MAE) with ridge regularization                                                      |
| hinge+ridge            | hinge loss (linear SVM) with ridge regularization                                                        |
| cauchy+ridge           | cauchy loss (robust regression) with ridge regularization                                                |
| logistic+ridge         | logistic loss (logistic regression) with ridge regularization                                            |
| mse+elasticnet         | mean squared error (MSE) with elastic net regularization                                                 |
| mae+elasticnet         | mean absolute error (MAE) with elastic net regularization                                                |
| hinge+elasticnet       | hinge loss (linear SVM) with elastic net regularization                                                  |
| cauchy+elasticnet      | cauchy loss (robust regression) with elastic net regularization                                          |
| logistic+elasticnet    | logistic loss (logistic regression) with elastic net regularization                                      |
| cvx48b                 | linear program: ex. 4.8(b), 'Convex Optimization', 2nd edition                                           |
| cvx48c                 | linear program: ex. 4.8(c), 'Convex Optimization', 2nd edition                                           |
| cvx48d-eq              | linear program: ex. 4.8(d) - equality case, 'Convex Optimization', 2nd edition                           |
| cvx48d-ineq            | linear program: ex. 4.8(d) - inequality case, 'Convex Optimization', 2nd edition                         |
| cvx48e-eq              | linear program: ex. 4.8(e) - equality case, 'Convex Optimization', 2nd edition                           |
| cvx48e-ineq            | linear program: ex. 4.8(e) - inequality case, 'Convex Optimization', 2nd edition                         |
| cvx48f                 | linear program: ex. 4.8(f), 'Convex Optimization', 2nd edition                                           |
| cvx49                  | linear program: ex. 4.9, 'Convex Optimization', 2nd edition                                              |
| cvx410                 | linear program: ex. 4.10, 'Convex Optimization', 2nd edition                                             |
| numopt162              | quadratic program: ex. 16.2, 'Numerical optimization', 2nd edition                                       |
| numopt1625             | quadratic program: ex. 16.25, 'Numerical optimization', 2nd edition                                      |
| osqp1                  | random quadratic program: A.1, 'OSQP: an operator splitting solver for quadratic programs'               |
| osqp2                  | equality constrained quadratic program: A.2, 'OSQP: an operator splitting solver for quadratic programs' |
| osqp4                  | portfolio optimization: A.4, 'OSQP: an operator splitting solver for quadratic programs'                 |
|------------------------|----------------------------------------------------------------------------------------------------------|
```
