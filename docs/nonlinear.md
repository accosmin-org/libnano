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

    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override
    {
        assert(size() == x.size());
        assert(size() == m_b.size());

        const auto dx = 1 + (x - m_b).dot(x - m_b) / 2;

        if (gx.size() == x.size())
        {
            gx = (x - m_b) / dx;
        }

        return std::log(dx);
    }

private:

    vector_t m_b;
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
|----------------------|---------------------------------------------------------------------------|
| solver               | description                                                               |
|----------------------|---------------------------------------------------------------------------|
| gd                   | gradient descent                                                          |
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

The default configurations are close to optimal for most situations. Still the user is free to experiment with the available parameters. The following piece of code extracted from the [example](../example/src/nonlinear.cpp) shows how to create a L-BFGS solver and how to change the line-search strategy, the tolerance and the maximum number of iterations:
```
#include <nano/solver.h>

auto solver                                 = solver_t::all().get("lbfgs");
solver->parameter("solver::lbfgs::history") = 20;
solver->parameter("solver::epsilon")        = 1e-8;
solver->parameter("solver::max_evals")      = 100;
solver->parameter("solver::tolerance")      = std::make_tuple(1e-4, 9e-1);
solver->lsearch0("constant");
solver->lsearchk("morethuente");
```

Then the optimal point is obtained by invoking the solver on the object like described below:
```
const auto  x0    = nano::make_random_vector<scalar_t>(objective.size());
const auto  state = solver->minimize(objective, x0, make_stdout_logger());
const auto& x     = state.x;

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


##### Compare solvers on convex smooth problems

```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "gd|cgd-pr|lbfgs|bfgs" \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 8
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | grad test    | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -6.9420   | 2.56 | N/A          | 2.58606e-08  | 0      | 0      | 49     | 49     | 0     |
| bfgs [quadratic,cgdescent]       | -6.9147   | 2.86 | N/A          | 3.18915e-08  | 0      | 0      | 38     | 38     | 12    |
| cgd-pr [quadratic,cgdescent]     | -6.7318   | 2.40 | N/A          | 2.43024e-08  | 0      | 0      | 106    | 106    | 0     |
| gd [quadratic,cgdescent]         | -6.1569   | 2.19 | N/A          | 0.0192149    | 0      | 253    | 472    | 472    | 1     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```
The results are typical: the BFGS algorithm is faster in terms of function value and gradient evaluations, but it requires the most in terms of processing time while the CGD and L-BFGS algorithms are fairly close. The steepest gradient descent method needs as expected many more iterations to converge. Note that BFGS scales quadratically with the problem size, while CGD and L-BFGS scale linearly and are thus recommended for very large problems.


##### Compare line-search methods on convex smooth problems

```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "lbfgs|bfgs" \
    --lsearchk ".+" \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 14
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | grad test    | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,lemarechal]      | -6.8718   | 6.53 | N/A          | 7.24873e-08  | 124    | 0      | 54     | 54     | 14    |
| bfgs [quadratic,fletcher]        | -6.8712   | 4.82 | N/A          | 6.27586e-08  | 159    | 0      | 47     | 47     | 14    |
| bfgs [quadratic,morethuente]     | -6.8691   | 6.96 | N/A          | 3.10749e-08  | 0      | 0      | 42     | 42     | 13    |
| bfgs [quadratic,backtrack]       | -6.8662   | 5.66 | N/A          | 7.25283e-08  | 0      | 76     | 133    | 133    | 22    |
| bfgs [quadratic,cgdescent]       | -6.8658   | 5.98 | N/A          | 3.05747e-08  | 0      | 0      | 38     | 38     | 12    |
| lbfgs [quadratic,cgdescent]      | -6.8493   | 4.74 | N/A          | 2.58819e-08  | 0      | 0      | 48     | 48     | 0     |
| lbfgs [quadratic,morethuente]    | -6.8171   | 5.62 | N/A          | 2.72689e-08  | 0      | 0      | 50     | 50     | 0     |
| lbfgs [quadratic,backtrack]      | -6.8170   | 4.75 | N/A          | 4.31076e-08  | 0      | 79     | 145    | 145    | 5     |
| lbfgs [quadratic,fletcher]       | -6.7991   | 4.17 | N/A          | 6.11294e-08  | 130    | 0      | 57     | 57     | 0     |
| lbfgs [quadratic,lemarechal]     | -6.7932   | 5.78 | N/A          | 5.24941e-08  | 99     | 0      | 62     | 62     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```
Notice that the CG-descent line-search method is the only one that doesn't fail while also has the smallest number of function evaluations.


Very precise solutions can be obtained efficiently using the CG-descent line-search method:
```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "cgd-pr|lbfgs|bfgs" \
    --lsearchk "morethuente|cgdescent" \
    --trials 128 --solver::epsilon 1e-14 --solver::max_evals 10000 | tail -n 10
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | grad test    | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,morethuente]     | -13.9813  | 3.82 | N/A          | 5.19544e-10  | 0      | 495    | 1258   | 1258   | 67    |
| bfgs [quadratic,cgdescent]       | -13.9457  | 4.50 | N/A          | 2.34693e-15  | 0      | 0      | 198    | 198    | 24    |
| cgd-pr [quadratic,cgdescent]     | -13.8782  | 2.83 | N/A          | 2.49759e-15  | 0      | 0      | 382    | 382    | 1     |
| lbfgs [quadratic,cgdescent]      | -13.8681  | 3.72 | N/A          | 3.06282e-15  | 0      | 0      | 194    | 194    | 1     |
| lbfgs [quadratic,morethuente]    | -13.7364  | 3.47 | N/A          | 1.05953e-09  | 0      | 408    | 984    | 984    | 24    |
| cgd-pr [quadratic,morethuente]   | -13.5462  | 2.67 | N/A          | 1.24311e-09  | 0      | 451    | 1123   | 1123   | 31    |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```
Notice that the CG-descent line-search method is the only one that doesn't fail to reach such a precision and also with around 3 times fewer function calls.


##### Compare solvers on convex non-smooth problems

The line-search monotonic solvers (like L-BFGS) are not guaranteed to converge for non-smooth problems. As such non-monotonic solvers may be more appropriate in this case.

```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex \
    --solver "gd|gs|cgd-pr|lbfgs|bfgs|osga|cocob|sgm|sda|wda|pgm|dgm|fgm|asga2|asga4" \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 19
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank  | value        | grad test    | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,cgdescent]       | -5.2944   | 3.68  | N/A          | 105093       | 63     | 518    | 491    | 491    | 81    |
| cgd-pr [quadratic,cgdescent]     | -4.3943   | 3.71  | N/A          | 94458.7      | 1827   | 770    | 195    | 195    | 3     |
| lbfgs [quadratic,cgdescent]      | -4.1832   | 3.91  | N/A          | 104462       | 62     | 544    | 513    | 513    | 23    |
| osga                             | -3.4477   | 5.39  | N/A          | 3.32437e+31  | 0      | 548    | 695    | 348    | 17    |
| asga2                            | -1.8241   | 7.76  | N/A          | 4.95335      | 0      | 758    | 594    | 594    | 22    |
| sgm                              | -1.7800   | 7.05  | N/A          | 27.3983      | 0      | 5326   | 1706   | 1706   | 72    |
| asga4                            | -1.3489   | 9.58  | N/A          | 1.94343      | 0      | 340    | 432    | 432    | 14    |
| fgm                              | -1.3353   | 7.35  | N/A          | 15.767       | 428    | 1019   | 648    | 648    | 19    |
| cocob                            | -0.8178   | 9.82  | N/A          | 1.07521      | 0      | 2039   | 1020   | 1020   | 38    |
| pgm                              | -0.1748   | 11.47 | N/A          | 0.748772     | 428    | 2098   | 750    | 750    | 21    |
| dgm                              | 0.0007    | 12.82 | N/A          | 0.66373      | 372    | 3806   | 1776   | 888    | 53    |
| wda                              | 0.3571    | 11.13 | N/A          | 5.35861      | 0      | 773    | 703    | 703    | 23    |
| gs                               | 1.2287    | 10.59 | N/A          | 6.51652      | 0      | 8417   | 2697   | 2550   | 863   |
| sda                              | 1.8354    | 11.62 | N/A          | 3.10457      | 0      | 3741   | 1527   | 1527   | 51    |
| gd [quadratic,cgdescent]         | inf       | 4.13  | N/A          | N/A          | 2342   | 616    | 283    | 283    | 2     |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|-------|
```
Indeed the monotonic solvers are not converging, but surprisingly they produce the most accurate solutions by at least an order of magnitude in the worst case. Out of the non-monotonic solvers only OSGA produces reasonable accurate solutions. The rest of non-monotonic solvers don't seem capable of converging fast enough for practical applications. Note that it is very difficult to have a practical and reliable stopping criterion for general convex non-smooth problems.


###### TODO: Add the bundle methods and the gradient sampling methods to the comparison.


##### Tune the L-BFGS history size on convex smooth problems

The limited-memory quasi-Newton (L-BFGS) method uses a small number of last known gradients to build a low-rank inverse of the Hessian matrix. This results in much more efficient iterations than quasi-Newton methods, but at a lower convergence rate and potential larger number of iterations. However it is not clear from the literature what is the optimum number of last gradients to use. The benchmark program allows to measure the impact of this parameter on the number of iterations until convergence.

```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "lbfgs|bfgs" \
    --solver::lbfgs::history 5 \
    --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 6
for hsize in 10 20 50 100 150 200
do
    ./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type convex-smooth \
        --solver "lbfgs|bfgs" \
        --solver::lbfgs::history ${hsize} \
        --trials 128 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 3
done
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | grad test    | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,cgdescent]       | -6.9667   | 1.65 | N/A          | 3.23335e-08  | 0      | 0      | 38     | 38     | 12    |
| lbfgs [quadratic,cgdescent]      | -6.8471   | 1.35 | N/A          | 2.91432e-08  | 0      | 6      | 134    | 134    | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,cgdescent]       | -6.9602   | 1.64 | N/A          | 3.14235e-08  | 0      | 0      | 38     | 38     | 12    |
| lbfgs [quadratic,cgdescent]      | -6.8383   | 1.36 | N/A          | 2.76048e-08  | 0      | 0      | 96     | 96     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,cgdescent]       | -6.9453   | 1.66 | N/A          | 3.19484e-08  | 0      | 0      | 38     | 38     | 12    |
| lbfgs [quadratic,cgdescent]      | -6.8650   | 1.34 | N/A          | 2.72612e-08  | 0      | 0      | 71     | 71     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,cgdescent]       | -6.9206   | 1.68 | N/A          | 3.13979e-08  | 0      | 0      | 38     | 38     | 12    |
| lbfgs [quadratic,cgdescent]      | -6.9163   | 1.32 | N/A          | 2.56038e-08  | 0      | 0      | 48     | 48     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -6.9233   | 1.31 | N/A          | 2.58559e-08  | 0      | 0      | 43     | 43     | 0     |
| bfgs [quadratic,cgdescent]       | -6.8930   | 1.69 | N/A          | 3.1198e-08   | 0      | 0      | 38     | 38     | 12    |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| bfgs [quadratic,cgdescent]       | -6.9218   | 1.68 | N/A          | 3.08453e-08  | 0      | 0      | 38     | 38     | 12    |
| lbfgs [quadratic,cgdescent]      | -6.9075   | 1.32 | N/A          | 2.57515e-08  | 0      | 0      | 43     | 43     | 0     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -6.9162   | 1.33 | N/A          | 2.60451e-08  | 0      | 0      | 43     | 43     | 0     |
| bfgs [quadratic,cgdescent]       | -6.9151   | 1.67 | N/A          | 3.25824e-08  | 0      | 0      | 38     | 38     | 12    |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```

L-BFGS is catching up to BFGS in terms of both accuracy and number of iterations by increasing the number of past gradients to use at very little additional computation cost.
