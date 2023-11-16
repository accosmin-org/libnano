# Nonlinear optimization module


#### Introduction

Libnano provides various methods to solve unconstrained non-linear numerical optimization problems. The goal is to find the minimum `x` of a (sub-)differential function `f(x)` of dimension `n` starting from an initial point `x0`. The returned point `x` is guaranteed to be the global minimum when the function is convex, and a stationary point (not necessarily a local minimum) otherwise. More details can be found in the following sources:


* "Practical Methods of Optimization 2e", R. Fletcher, 2000
* "Introductory Lectures on Convex Optimization: A Basic Course", Y. Nesterov, 2003
* "Convex Optimization", S. Boyd, L. Vandenberghe, 2004
* "Numerical Optimization", J. Nocedal, S. Wright, 2006


The builtin solvers are `non-parametric` in the sense that their convergence properties don't dependent on particular values of hyper-parameters. Also the functions to minimize are not constrained to have a particular structure (e.g. like in machine learning applications). Thus the solvers can be used efficiently as black-box solvers without needing tuning. However it is important to use an appropriate type of solver adapted to the problem at hand:

* `monotonic` solvers which decrease the current function value at each iteration using a `descent direction` and a `line-search` along this direction. These solvers are appropriate only for smooth functions, not necessarily convex, which can be solved with very high precision. Examples: conjugate gradient descent (CGD), LBFGS, quasi-Newton methods.

* `non-monotonic` solvers which are not guaranteed to decrease the current function value at each iteration. These solvers should be used only for non-smooth convex problems. They can work well for smooth convex problems as well, but they are not as efficient (iteration-wise) as the monotonic solvers. Examples: optimal sub-gradient algorithm (OSGA), sub-gradient method (SGM) or the ellipsoid method.


Each concept involved in the optimization procedure is mapped to a particular interface. Most relevant are the [function_t](../include/nano/function.h) and the [solver_t](../include/nano/solver.h) interfaces. The builtin implementations can be accessed programatically in C++ using their associated factory.


#### Function

The function to minimize must be an instance of `function_t`. The user needs to implement the evaluation of the function value and gradient. The following piece of code extracted from the [example](../example/src/nonlinear.cpp) defines a quadratic function of arbitrary dimensions:

```
#include <nano/function.h>

using namespace nano;

class objective_t final : public function_t
{
public:

    objective_t(int size) :
        function_t("objective's name", size),
        m_b(vector_t::Random(size))
    {
        convex(convexity::yes);
        smooth(smoothness::yes);
    }

    rfunction_t clone() const override
    {
        return std::make_unique<objective_t>(*this);
    }

    scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override
    {
        assert(size() == x.size());
        assert(size() == m_b.size());

        const auto dx = 1 + (x - m_b).dot(x - m_b) / 2;

        if (gx != nullptr)
        {
            *gx = (x - m_b) / dx;
        }

        return std::log(dx);
    }

private:

    vector_t    m_b;
};
```

Additionally the library implements various widely used functions for benchmarking numerical optimization algorithms (see [Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization) for some examples). These are used by the command line utility [app/bench_solver](../app/bench_solver.cpp) to benchmark the builtin optimization algorithms.


#### Solver

Once the objective function is defined, the user needs to create a `solver_t` object to use for minimization. The associated factory can be used to list the builtin solvers in C++ like this:

```
for (const auto& solver_id : nano::solver_t::all().ids())
{
    std::cout << "solver_id: " << solver_id << std::endl;
}
```

Another possibility is to print a tabular representation with the ID and a short description of all builtin solvers:
```
std::cout << make_table("solver", solver_t::all());
// prints something like...
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
| gs        | gradient sampling                                                   |
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

The default configurations are close to optimal for most situations. Still the user is free to experiment with the available parameters. The following piece of code extracted from the [example](../example/src/nonlinear.cpp) shows how to create a L-BFGS solver and how to change the line-search strategy, the tolerance and the maximum number of iterations:
```
#include <nano/solver/lbfgs.h>

auto solver = nano::solver_lbfgs_t{};
solver.parameter("solver::lbfgs::history") = 20;
solver.parameter("solver::epsilon") = 1e-8;
solver.parameter("solver::max_evals") = 100;
solver.parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1);
solver.lsearch0("constant");
solver.lsearchk("morethuente");
```

Then the optimal point is obtained by invoking the solver on the object like described below:
```
const auto x0 = nano::make_random_vector<scalar_t>(objective.size());
const auto state = solver.minimize(objective, x0);
const auto& x = state.x;

std::cout << std::fixed << std::setprecision(12)
    << "f0=" << objective.vgrad(x0, nullptr) << ", f=" << state.fx()
    << ", g=" << state.gradient_test()
    << ", x-x*=" << (state.x() - objective.b()).lpNorm<Eigen::Infinity>()
    << ", fcalls=" << state.fcalls()
    << ", gcalls=" << state.gcalls()
    << ", status=" << nano::scat(state.status()) << std::endl;
```


Choosing the right optimization algorithm is usually done in terms of processing time and memory usage. The non-linear conjugate gradient descent familly of algorithms (CGD) and the limited-memory BFGS (L-BFGS) are recommended for large problems because they perform *O(n)* FLOPs per iteration and use *O(n)* memory while having super-linear convergence. The quasi-Newton algorithms, and BFGS in particular, converge faster both in terms of iterations and gradient evaluations, but are only recommended for small problems because of their *O(n^2)* FLOPs and memory usage. The gradient descent is provided here as a baseline to compare with as it generally takes 1-2 orders of magnitude more iterations to reach similar accuracy as CGD or L-BFGS.


The solvers can be configured with different line-search steps initialization and search methods that implement the `lsearch0_t` and the `lsearchk_t` interfaces respectively. The builtin implementations can be accessed from the associated factories with `lsearch0_t::all()` and `lsearchk_t::all()` respectively. Note that the state-of-the-art line-search method *CG_DESCENT* is recommended to use when high precision solutions of machine precision are needed.

```
std::cout << make_table("lsearchk", lsearchk_t::all());
// prints something like...
|-------------|---------------------------------------------------------|
| lsearchk    | description                                             |
|-------------|---------------------------------------------------------|
| backtrack   | backtrack using cubic interpolation (Armijo conditions) |
| cgdescent   | CG-DESCENT (regular and approximate Wolfe conditions)   |
| fletcher    | Fletcher (strong Wolfe conditions)                      |
| lemarechal  | LeMarechal (regular Wolfe conditions)                   |
| morethuente | More&Thuente (strong Wolfe conditions)                  |
|-------------|---------------------------------------------------------|
```


#### Examples


A working example for constructing and minimizing an objective function can be found in the [example](../example/src/nonlinear.cpp). The source shows additionally how to:
* compute objective function values at various points,
* compute the accuracy of the gradient using central finite difference,
* retrieve and configure the solver,
* log the optimization path in detail and
* retrieve the optimization result.


Additionally the command line utility [app/bench_solver](../app/bench_solver.cpp) is useful for benchmarking the builtin optimization algorithms on standard numerical optimization test problems and typical machine learning problems. See bellow several such experiments.


#### Compare solvers on convex smooth problems

```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --smooth \
    --solver "gd|cgd-pr|lbfgs|bfgs" \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 8
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | gnorm        | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -6.9386   | 1.84 | N/A          | 4.31366e-08  | 0      | 0      | 46     | 46     | 12    |
| lbfgs [quadratic,morethuente]    | -6.6258   | 2.08 | N/A          | 4.17281e-08  | 0      | 0      | 68     | 68     | 0     |
| cgd-pr [quadratic,morethuente]   | -6.5633   | 2.20 | N/A          | 3.49864e-08  | 0      | 0      | 99     | 99     | 0     |
| gd [quadratic,morethuente]       | -5.0477   | 3.87 | N/A          | 0.0583651    | 0      | 384    | 821    | 821    | 1     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```
The results are typical: the BFGS algorithm is faster in terms of function value and gradient evaluations, but it requires the most in terms of processing time while the CGD and L-BFGS algorithms are fairly close. The steepest gradient descent method needs as expected many more iterations to converge. Note that BFGS scales quadratically with the problem size, while CGD and L-BFGS scale linearly and are thus recommended for very large problems.


#### Compare line-search methods on convex smooth problems

```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --smooth \
    --solver "lbfgs|bfgs" \
    --lsearchk ".+" \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 14
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | gnorm        | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,lemarechal]      | -6.8104   | 5.21 | N/A          | 4.10553e-08  | 0      | 0      | 49     | 49     | 12    |
| bfgs [quadratic,fletcher]        | -6.8088   | 4.41 | N/A          | 4.17745e-08  | 0      | 0      | 49     | 49     | 12    |
| bfgs [quadratic,morethuente]     | -6.8056   | 6.09 | N/A          | 4.30813e-08  | 0      | 0      | 46     | 46     | 12    |
| bfgs [quadratic,backtrack]       | -6.8014   | 3.95 | N/A          | 4.3236e-08   | 0      | 0      | 51     | 51     | 13    |
| bfgs [quadratic,cgdescent]       | -6.8003   | 4.51 | N/A          | 4.23293e-08  | 0      | 0      | 44     | 44     | 11    |
| lbfgs [quadratic,cgdescent]      | -6.6298   | 5.70 | N/A          | 4.00705e-08  | 0      | 0      | 67     | 67     | 0     |
| lbfgs [quadratic,backtrack]      | -6.6022   | 5.58 | N/A          | 4.21418e-08  | 0      | 0      | 75     | 75     | 0     |
| lbfgs [quadratic,lemarechal]     | -6.5767   | 6.70 | N/A          | 4.25925e-08  | 0      | 0      | 69     | 69     | 0     |
| lbfgs [quadratic,morethuente]    | -6.5758   | 6.75 | N/A          | 4.16728e-08  | 0      | 0      | 68     | 68     | 0     |
| lbfgs [quadratic,fletcher]       | -6.5538   | 6.09 | N/A          | 0.0282253    | 2      | 0      | 68     | 68     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```


Very precise solutions can be obtained efficiently using the CG-descent line-search method:
```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --smooth \
    --solver "cgd-pr|lbfgs|bfgs" \
    --lsearch0 "cgdescent" \
    --lsearchk "morethuente|cgdescent" \
    --trials 128 --solver::epsilon 1e-14 --solver::max_evals 5000 | tail -n 10
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | gnorm        | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [cgdescent,morethuente]     | -13.8490  | 2.82 | N/A          | 3.33163e-09  | 0      | 492    | 886    | 784    | 63    |
| cgd-pr [cgdescent,cgdescent]     | -13.4563  | 3.56 | N/A          | 6.08133e-14  | 0      | 35     | 369    | 234    | 1     |
| bfgs [cgdescent,cgdescent]       | -13.4529  | 3.32 | N/A          | 3.97131e-15  | 0      | 0      | 152    | 80     | 24    |
| lbfgs [cgdescent,cgdescent]      | -13.4412  | 4.16 | N/A          | 4.69193e-15  | 0      | 0      | 248    | 128    | 0     |
| lbfgs [cgdescent,morethuente]    | -13.3946  | 4.05 | N/A          | 3.67177e-09  | 0      | 482    | 903    | 784    | 30    |
| cgd-pr [cgdescent,morethuente]   | -13.2778  | 3.09 | N/A          | 3.79544e-09  | 0      | 503    | 859    | 772    | 32    |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```
Notice that the CG-descent line-search method is the only one that doesn't fail to reach such a precision and also with 2-3 times fewer function calls.


#### Compare solvers on convex non-smooth problems

The line-search monotonic solvers (like L-BFGS) are not guaranteed to converge for non-smooth problems. As such non-monotonic solvers may be more appropriate in this case.

```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --non-smooth \
    --solver "gd|gs|cgd-pr|lbfgs|bfgs|osga|cocob|sgm|sda|wda|pgm|dgm|fgm|asga2|asga4" \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 19
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank  | value        | gnorm        | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -7.0000   | 1.83  | N/A          | 0.727718     | 0      | 1024   | 1833   | 1833   | 361   |
| lbfgs [quadratic,morethuente]    | -6.6116   | 3.25  | N/A          | 0.727699     | 235    | 789    | 1574   | 1574   | 110   |
| osga                             | -5.1023   | 4.24  | N/A          | 0.7276       | 0      | 593    | 2289   | 1145   | 104   |
| cgd-pr [quadratic,morethuente]   | -3.6776   | 5.11  | N/A          | 0.72298      | 0      | 1024   | 1831   | 1831   | 123   |
| gd [quadratic,morethuente]       | -3.3640   | 6.72  | N/A          | 0.727799     | 0      | 1024   | 1823   | 1823   | 122   |
| sgm                              | -3.3548   | 6.19  | N/A          | 0.733261     | 0      | 1036   | 1935   | 1935   | 125   |
| fgm                              | -3.0119   | 6.18  | N/A          | 0.727322     | 0      | 73     | 1609   | 1609   | 93    |
| asga2                            | -2.5801   | 8.77  | N/A          | 0.714641     | 0      | 125    | 1489   | 1489   | 85    |
| wda                              | -2.3653   | 10.39 | N/A          | 0.831858     | 0      | 1039   | 2292   | 2292   | 139   |
| asga4                            | -2.1634   | 10.16 | N/A          | 0.627735     | 0      | 111    | 1672   | 1672   | 97    |
| cocob                            | -1.8201   | 7.92  | N/A          | 0.762904     | 0      | 1280   | 2421   | 2421   | 159   |
| pgm                              | -1.5533   | 11.93 | N/A          | 0.266456     | 0      | 256    | 737    | 737    | 40    |
| gs                               | -1.4107   | 9.47  | N/A          | 0.727334     | 0      | 1325   | 2574   | 2574   | 367   |
| dgm                              | -1.3563   | 12.12 | N/A          | 0.2893       | 0      | 655    | 2117   | 1059   | 98    |
| sda                              | -1.3026   | 10.18 | N/A          | 0.909667     | 0      | 1349   | 2479   | 2479   | 149   |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|-------|
```
Indeed the monotonic solvers are not converging, but surprisingly they produce the most accurate solutions by at least an order of magnitude in the worst case. Out of the non-monotonic solvers only OSGA produces reasonable accurate solutions. The rest of non-monotonic solvers don't seem capable of converging fast enough for practical applications. Note that it is very difficult to have a practical and reliable stopping criterion for general convex non-smooth problems.


#### Tune the L-BFGS history size on convex smooth problems

The limited-memory quasi-Newton (L-BFGS) method uses a small number of last known gradients to build a low-rank inverse of the Hessian matrix. This results in much more efficient iterations than quasi-Newton methods, but at a lower convergence rate and potential larger number of iterations. However it is not clear from the literature what is the optimum number of last gradients to use. The benchmark program allows to measure the impact of this parameter on the number of iterations until convergence.

```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --smooth \
    --solver "lbfgs|bfgs" \
    --solver::lbfgs::history 5 \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 6
for hsize in 10 20 50 100 150 200
do
    ./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --smooth \
        --solver "lbfgs|bfgs" \
        --solver::lbfgs::history ${hsize} \
        --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 3
done
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | gnorm        | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -6.9823   | 1.41 | N/A          | 4.30812e-08  | 0      | 0      | 46     | 46     | 12    |
| lbfgs [quadratic,morethuente]    | -6.5600   | 1.59 | N/A          | 4.06911e-08  | 0      | 0      | 112    | 112    | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -6.9730   | 1.40 | N/A          | 4.31333e-08  | 0      | 0      | 46     | 46     | 12    |
| lbfgs [quadratic,morethuente]    | -6.5787   | 1.60 | N/A          | 4.31472e-08  | 0      | 0      | 89     | 89     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -6.9507   | 1.46 | N/A          | 4.31375e-08  | 0      | 0      | 46     | 46     | 12    |
| lbfgs [quadratic,morethuente]    | -6.6301   | 1.54 | N/A          | 4.1729e-08   | 0      | 0      | 68     | 68     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -6.8646   | 1.51 | N/A          | 4.30813e-08  | 0      | 0      | 46     | 46     | 12    |
| lbfgs [quadratic,morethuente]    | -6.7874   | 1.49 | N/A          | 4.04148e-08  | 0      | 0      | 44     | 44     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -6.8626   | 1.50 | N/A          | 4.30813e-08  | 0      | 0      | 46     | 46     | 12    |
| lbfgs [quadratic,morethuente]    | -6.7882   | 1.50 | N/A          | 3.98891e-08  | 0      | 0      | 39     | 39     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -6.8626   | 1.50 | N/A          | 4.30813e-08  | 0      | 0      | 46     | 46     | 12    |
| lbfgs [quadratic,morethuente]    | -6.7882   | 1.50 | N/A          | 3.98375e-08  | 0      | 0      | 39     | 39     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -6.8626   | 1.49 | N/A          | 4.31338e-08  | 0      | 0      | 46     | 46     | 12    |
| lbfgs [quadratic,morethuente]    | -6.7882   | 1.51 | N/A          | 3.98813e-08  | 0      | 0      | 39     | 39     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```

L-BFGS is catching up to BFGS in terms of both accuracy and number of iterations by increasing the number of past gradients to use at very little additional computation cost.
