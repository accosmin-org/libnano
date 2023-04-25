# Numerical optimization module


#### Introduction

Libnano provides various methods to solve unconstrained non-linear numerical optimization problems. The goal is to find the minimum **x** of a (sub-)differential function **f(x)** of dimension *n* starting from an initial point **x0**. The returned point **x** is guaranteed to be the global minimum when the function is convex, and a stationary point (not necessarily a local minimum) otherwise. More details can be found in the following sources:


* "Practical Methods of Optimization 2e", R. Fletcher, 2000
* "Introductory Lectures on Convex Optimization: A Basic Course", Y. Nesterov, 2003
* "Convex Optimization", S. Boyd, L. Vandenberghe, 2004
* "Numerical Optimization", J. Nocedal, S. Wright, 2006


The builtin solvers are **non-parametric** in the sense that their convergence properties don't dependent on particular values of hyper-parameters. Also the functions to minimize are not constrained to have a particular structure (e.g. like in machine learning applications). Thus the solvers can be used efficiently as black-box solvers without needing tuning. However it is important to use an appropriate type of solver adapted to the problem at hand:

* **monotonic** solvers which decrease the current function value at each iteration using a *descent direction* and a *line-search* along this direction. These solvers are appropriate only for smooth functions, not necessarily convex, which can be solved with very high precision. Examples: conjugate gradient descent (CGD), LBFGS, quasi-Newton methods.

* **non-monotonic** solvers which are not guaranteed to decrease the current function value at each iteration. These solvers should be used only for non-smooth convex problems, which can be solved with high precision. They can work well for smooth convex problems as well, but they are not as efficient (iteration-wise) as the monotonic solvers. Examples: optimal sub-gradient algorithm (OSGA), sub-gradient method (SGM) or the ellipsoid method.


Each concept involved in the optimization procedure is mapped to a particular interface. Most relevant are the [function_t](../include/nano/function.h) and the [solver_t](../include/nano/solver.h) interfaces. The builtin implementations can be accessed programatically in C++ using their associated factory.


#### Function

The function to minimize must be an instance of `function_t`. The user needs to implement the evaluation of the function value and gradient. The following piece of code extracted from the [example](../example/src/minimize.cpp) defines a quadratic function of arbitrary dimensions:

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

The default configurations are close to optimal for most situations. Still the user is free to experiment with the available parameters. The following piece of code extracted from the [example](../example/src/minimize.cpp) shows how to create a L-BFGS solver and how to change the line-search strategy, the tolerance and the maximum number of iterations:
```
#include <nano/solver/lbfgs.h>

auto solver = nano::solver_lbfgs_t{};
solver.parameter("solver::lbfgs::history") = 6;
solver.parameter("solver::epsilon") = 1e-6;
solver.parameter("solver::max_evals") = 100;
solver.parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1);
solver.lsearch0("constant");
solver.lsearchk("morethuente");
```

Then the optimal point is obtained by invoking the solver on the object like described below:
```
const auto x0 = nano::vector_t::Random(objective.size());
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


The solvers can be configured with different line-search steps initialization and search methods that implement the `lsearch0_t` and the `lsearchk_t` interfaces respectively. The builtin implementations can be accessed from the associated factories with `lsearch_t::all()` and `lsearchk_t::all()` respectively. Note that the state-of-the-art line-search method *CG_DESCENT* is recommended to use when high precision solutions of machine precision are needed.

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


#### Example


A working example for constructing and minimizing an objective function can be found in the [example](../example/src/minimize.cpp). The source shows additionally how to:
* compute objective function values at various points,
* compute the accuracy of the gradient using central finite difference,
* retrieve and configure the solver,
* log the optimization path in detail and
* retrieve the optimization result.


The command line utility [app/bench_solver](../app/bench_solver.cpp) is useful for benchmarking the builtin optimization algorithms on standard test functions.


The following run compares 4 solvers on all convex smooth builtin functions with 100 dimensions:
```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --smooth \
    --solver "gd|cgd-pr|lbfgs|bfgs" \
    --trials 1000 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 8
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | gnorm        | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -6.9631   | 1.84 | N/A          | 4.28363e-08  | 0      | 0      | 46     | 46     | 11    |
| cgd-pr [quadratic,morethuente]   | -6.5655   | 2.14 | N/A          | 3.50958e-08  | 0      | 0      | 99     | 99     | 0     |
| lbfgs [quadratic,morethuente]    | -6.5625   | 2.15 | N/A          | 4.0622e-08   | 0      | 0      | 106    | 106    | 0     |
| gd [quadratic,morethuente]       | -5.0437   | 3.86 | N/A          | 0.0523154    | 0      | 3000   | 821    | 821    | 1     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```
The results are typical: the BFGS algorithm is faster in terms of function value and gradient evaluations, but it requires the most in terms of processing time while the CGD and L-BFGS algorithms are fairly close. The steepest gradient descent method needs as expected many more iterations to converge. Note that BFGS scales quadratically with the problem size, while CGD and L-BFGS scale linearly and are thus recommended for very large problems.


The builtin line-search methods can be compared for the same problem like:
```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --smooth \
    --solver "lbfgs|bfgs" \
    --lsearchk ".+" \
    --trials 1000 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 14
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | gnorm        | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,lemarechal]      | -6.9298   | 5.19 | N/A          | 4.07585e-08  | 0      | 0      | 49     | 49     | 12    |
| bfgs [quadratic,fletcher]        | -6.9291   | 4.33 | N/A          | 4.11342e-08  | 0      | 0      | 49     | 49     | 11    |
| bfgs [quadratic,morethuente]     | -6.9233   | 6.06 | N/A          | 4.28211e-08  | 0      | 0      | 46     | 46     | 11    |
| bfgs [quadratic,backtrack]       | -6.9173   | 3.96 | N/A          | 4.33536e-08  | 0      | 0      | 51     | 51     | 13    |
| bfgs [quadratic,cgdescent]       | -6.9156   | 4.49 | N/A          | 4.22531e-08  | 0      | 0      | 44     | 44     | 11    |
| lbfgs [quadratic,cgdescent]      | -6.5560   | 5.84 | N/A          | 4.22903e-08  | 0      | 0      | 107    | 107    | 0     |
| lbfgs [quadratic,backtrack]      | -6.5390   | 5.47 | N/A          | 4.26358e-08  | 0      | 0      | 112    | 112    | 0     |
| lbfgs [quadratic,morethuente]    | -6.5367   | 6.94 | N/A          | 4.06069e-08  | 0      | 0      | 106    | 106    | 0     |
| lbfgs [quadratic,lemarechal]     | -6.5036   | 6.69 | N/A          | 4.28758e-08  | 0      | 0      | 107    | 107    | 0     |
| lbfgs [quadratic,fletcher]       | -6.5002   | 6.04 | N/A          | 0.0252848    | 19     | 0      | 106    | 106    | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```


Very precise solutions can be obtained efficiently using the CG-descent line-search method:
```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --smooth \
    --solver "lbfgs|bfgs" \
    --lsearch0 "cgdescent" \
    --lsearchk "morethuente|cgdescent" \
    --trials 1000 --solver::epsilon 1e-14 --solver::max_evals 5000 | tail -n 10
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | gnorm        | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [cgdescent,morethuente]     | -13.8571  | 2.99 | N/A          | 3.45099e-09  | 0      | 3362   | 791    | 693    | 52    |
| cgd-pr [cgdescent,cgdescent]     | -13.4696  | 3.47 | N/A          | 5.62854e-14  | 0      | 234    | 370    | 234    | 1     |
| bfgs [cgdescent,cgdescent]       | -13.4648  | 3.20 | N/A          | 3.96616e-15  | 0      | 0      | 152    | 80     | 23    |
| lbfgs [cgdescent,cgdescent]      | -13.4603  | 4.07 | N/A          | 4.52791e-15  | 0      | 0      | 367    | 188    | 0     |
| lbfgs [cgdescent,morethuente]    | -13.3660  | 4.13 | N/A          | 3.89321e-09  | 0      | 3391   | 922    | 753    | 21    |
| cgd-pr [cgdescent,morethuente]   | -13.2651  | 3.13 | N/A          | 3.89886e-09  | 0      | 3129   | 701    | 623    | 18    |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```
Notice that the CG-descent line-search method is the only one that doesn't fail to reach such a precision and also with 2-3 times fewer function calls.


However if the function is not smooth, then monotonic solvers like L-BFGS are not guaranteed to converge:
```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --convex --non-smooth \
    --solver "gd|cgd-pr|lbfgs|bfgs|osga|cocob|sgm|sda|wda|pgm|dgm|fgm" \
    --trials 1000 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 16
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank  | value        | gnorm        | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -7.0000   | 1.69  | N/A          | 0.710155     | 0      | 8000   | 1834   | 1834   | 363   |
| lbfgs [quadratic,morethuente]    | -6.6044   | 3.13  | N/A          | 0.710196     | 748    | 7252   | 1721   | 1721   | 118   |
| osga                             | -5.0198   | 3.77  | N/A          | 0.710154     | 0      | 3176   | 1912   | 957    | 93    |
| cgd-pr [quadratic,morethuente]   | -3.7104   | 4.85  | N/A          | 0.708688     | 5      | 7995   | 1830   | 1830   | 122   |
| sgm                              | -3.3518   | 5.67  | N/A          | 0.717901     | 0      | 8118   | 1936   | 1936   | 126   |
| gd [quadratic,morethuente]       | -3.3208   | 6.30  | N/A          | 0.709155     | 0      | 8000   | 1823   | 1823   | 122   |
| fgm                              | -3.0087   | 5.96  | N/A          | 0.711256     | 0      | 582    | 1605   | 1605   | 93    |
| wda                              | -2.3325   | 9.45  | N/A          | 0.827981     | 0      | 8253   | 2294   | 2294   | 139   |
| cocob                            | -1.8192   | 7.38  | N/A          | 0.739145     | 0      | 9996   | 2422   | 2422   | 158   |
| pgm                              | -1.5560   | 10.19 | N/A          | 0.261627     | 0      | 1904   | 709    | 709    | 39    |
| dgm                              | -1.3553   | 10.24 | N/A          | 0.291631     | 0      | 5203   | 2043   | 1022   | 94    |
| sda                              | -1.2985   | 9.37  | N/A          | 0.896372     | 0      | 10409  | 2466   | 2466   | 148   |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|-------|
```
Indeed the monotonic solvers are not converging, but surprisingly they produce the most accurate solutions by at least an order of magnitude in the worst case. Out of the non-monotonic solvers only OSGA produces reasonable accurate solutions. The rest of non-monotonic solvers don't seem capable of converging fast enough for practical applications. Note that it is very difficult to have a practical and reliable stopping criterion for general convex non-smooth problems.
