# Build instructions


#### Structure

The project is organized as follows:
* [include/nano](../include/nano) - C++ headers files representing the library's interface.
* [src](../src) - C++ source files with implementation details.
* [app](../app) - command line C++ utilities mostly used for benchmarking various components.
* [test](../test) - C++ unit tests run with [ctest](https://cmake.org/cmake/help/v3.16/manual/ctest.1.html#ctest-1).
* [docs](../docs) - documentation written in Markdown.
* [cmake](../cmake) - various CMake utilities (e.g. to simplify building unit tests).
* [scripts](../scripts) - various Bash utilities (e.g. to simplify configuring and building of the library, to download datasets or to run experiments).
* [example](../example) - example programs used for testing the setup of the library and for showcasing some of its functionality.


#### Dependencies

System:
* compiler supporting [C++17](https://isocpp.org/wiki/faq/cpp17)
* [CMake](https://cmake.org)
* [Eigen3](https://eigen.tuxfamily.org) - high-performance linear-algebra C++ library

Libnano is tested on Linux (Ubuntu, Fedora) using both gcc and clang, on MacOS using AppleClang and on Windows using Visual Studio. It may work with minor changes on other platforms as well or using other compiler versions.


#### How to build

The easiest way to build, test and install the library on Linux, MacOS or Windows is to call either [scripts/build.sh](../scripts/build.sh) or [scripts/build.bat](../scripts/build.bat) with the appropriate command line arguments. These scripts are invoked to run various tests on continuous integrations plaforms like [github workflows](https://github.com/accosmin-org/libnano/actions). See [.github/workflows](../.github/workflows) for examples.


Otherwise, users can also invoke the main CMake script directly for other platforms or for custom builds.


The following command displays the command line arguments supported by [scripts/build.sh](../scripts/build.sh):
```
usage: scripts/build.sh [OPTIONS]

options:
    -h,--help
        print usage
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


### Library design

The library is designed by mapping all relevant concepts and algorithms in numerical optimization and machine learning to proper interfaces. The available implementations are fully parametrizable and are all registered to extendable factories. These can be easily discovered using the builtin command line utility [app/info](../app/info.cpp) assumed in the following examples to be built in the folder ```build/libnano/debug``` using one of the example above.

Examples:
* list the available numerical optimization algorithms (solvers):
```
./build/libnano/debug/app/info --list-solver
|-----------|---------------------------------------------------------------------|
| solver    | description                                                         |
|-----------|---------------------------------------------------------------------|
| asga2     | accelerated sub-gradient algorithm (ASGA-2)                         |
| asga4     | accelerated sub-gradient algorithm (ASGA-4)                         |
| bfgs      | quasi-newton method (BFGS)                                          |
| cgd-cd    | conjugate gradient descent (CD)                                     |
| cgd-dy    | conjugate gradient descent (DY)                                     |
| cgd-dycd  | conjugate gradient descent (DYCD)                                   |
| cgd-dyhs  | conjugate gradient descent (DYHS)                                   |
| cgd-fr    | conjugate gradient descent (FR)                                     |
| cgd-frpr  | conjugate gradient descent (FRPR)                                   |
| cgd-hs    | conjugate gradient descent (HS+)                                    |
| cgd-ls    | conjugate gradient descent (LS+)                                    |
| cgd-n     | conjugate gradient descent (N+)                                     |
| cgd-pr    | conjugate gradient descent (default)                                |
| cocob     | continuous coin betting (COCOB)                                     |
| dfp       | quasi-newton method (DFP)                                           |
| dgm       | universal dual gradient method (DGM)                                |
| ellipsoid | ellipsoid method                                                    |
| fgm       | universal fast gradient method (FGM)                                |
| fletcher  | quasi-newton method (Fletcher's switch)                             |
| gd        | gradient descent                                                    |
| hoshino   | quasi-newton method (Hoshino formula)                               |
| lbfgs     | limited-memory BFGS                                                 |
| osga      | optimal sub-gradient algorithm (OSGA)                               |
| pgm       | universal primal gradient method (PGM)                              |
| sda       | simple dual averages (variant of primal-dual subgradient methods)   |
| sgm       | sub-gradient method                                                 |
| sr1       | quasi-newton method (SR1)                                           |
| wda       | weighted dual averages (variant of primal-dual subgradient methods) |
|-----------|---------------------------------------------------------------------|
```

* list the name, the default value and the domain of all parameters of some solvers of interest:
```
./build/libnano/debug/app/info --list-solver-params --solver "lbfgs|bfgs|cgd-pr"
|--------|-------------------------------|--------------|--------------------------|
| solver | parameter                     | value        | domain                   |
|--------|-------------------------------|--------------|--------------------------|
| bfgs   | quasi-newton method (BFGS)                                              |
|--------|-------------------------------|--------------|--------------------------|
| bfgs   | solver::epsilon               | 1e-08        | 0 < 1e-08 <= 0.1         |
| bfgs   | solver::max_evals             | 1000         | 10 <= 1000 <= 1000000000 |
| bfgs   | solver::tolerance             | (0.0001,0.9) | 0 < 0.0001 < 0.9 < 1     |
| bfgs   | solver::quasi::initialization | identity     | identity,scaled          |
|--------|-------------------------------|--------------|--------------------------|
| cgd-pr | conjugate gradient descent (default)                                    |
|--------|-------------------------------|--------------|--------------------------|
| cgd-pr | solver::epsilon               | 1e-08        | 0 < 1e-08 <= 0.1         |
| cgd-pr | solver::max_evals             | 1000         | 10 <= 1000 <= 1000000000 |
| cgd-pr | solver::tolerance             | (0.0001,0.1) | 0 < 0.0001 < 0.1 < 1     |
| cgd-pr | solver::cgd::orthotest        | 0.1          | 0 < 0.1 < 1              |
|--------|-------------------------------|--------------|--------------------------|
| lbfgs  | limited-memory BFGS                                                     |
|--------|-------------------------------|--------------|--------------------------|
| lbfgs  | solver::epsilon               | 1e-08        | 0 < 1e-08 <= 0.1         |
| lbfgs  | solver::max_evals             | 1000         | 10 <= 1000 <= 1000000000 |
| lbfgs  | solver::tolerance             | (0.0001,0.9) | 0 < 0.0001 < 0.9 < 1     |
| lbfgs  | solver::lbfgs::history        | 20           | 1 <= 20 <= 1000          |
|--------|-------------------------------|--------------|--------------------------|
```
