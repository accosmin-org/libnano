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

    scalar_t do_eval(eval_t eval) const override
    {
        const auto x  = eval.m_x;
        const auto dx = 1.0 + 0.5 * (x - m_b).dot(x - m_b);

        if (eval.has_grad())
        {
            eval.m_gx = (x - m_b) / dx;
        }

        if (eval.has_hess())
        {
            eval.m_hx = -((x - m_b) * (x - m_b).transpose()) / square(dx);
            eval.m_hx.diagonal().array() += 1.0 / dx;
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
    << "f0=" << objective(x0) << ", f=" << state.fx()
    << ", g=" << state.gradient_test()
    << ", x-x*=" << (state.x() - objective.b()).lpNorm<Eigen::Infinity>()
    << ", fcalls=" << state.fcalls()
    << ", gcalls=" << state.gcalls()
    << ", hcalls=" << state.hcalls()
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


A working example for constructing and minimizing an objective function can be found in the [example](../example/src/nonlinear.cpp). The source shows how to:
* compute objective function values at various points,
* compute the accuracy of the gradient using central finite difference,
* retrieve and configure the solver,
* log the optimization path in detail and
* retrieve the optimization result.


Additionally the command line utility [app/bench_solver](../app/bench_solver.cpp) is useful for benchmarking the builtin optimization algorithms on standard numerical optimization test problems and typical machine learning problems. See bellow several such experiments.


##### Compare solvers on convex smooth problems

```
./build/libnano/gcc-release/app/bench_solver \
    --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "gd|cgd-pr|lbfgs|bfgs|hoshino|fletcher|newton" \
    --trials 20 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 11
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | log10(gtest) | errors | maxits | fcalls | gcalls | hcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| newton [quadratic,cgdescent]     | -7.0000   | 2.45 | N/A          | -11.8102     | 0      | 0      | 6      | 3      | 2      | 3     |
| cgd-pr [quadratic,cgdescent]     | -7.0000   | 4.03 | N/A          | -8.52384     | 0      | 0      | 34     | 34     | 0      | 1     |
| lbfgs [quadratic,cgdescent]      | -7.0000   | 4.25 | N/A          | -9.84792     | 0      | 0      | 24     | 24     | 0      | 1     |
| bfgs [quadratic,cgdescent]       | -7.0000   | 4.13 | N/A          | -8.65299     | 0      | 0      | 35     | 35     | 0      | 6     |
| hoshino [quadratic,cgdescent]    | -7.0000   | 4.08 | N/A          | -8.91487     | 0      | 0      | 38     | 38     | 0      | 9     |
| fletcher [quadratic,cgdescent]   | -7.0000   | 4.46 | N/A          | -8.84036     | 0      | 0      | 38     | 38     | 0      | 5     |
| gd [quadratic,cgdescent]         | -6.9496   | 4.60 | N/A          | -8.09401     | 0      | 295    | 373    | 373    | 0      | 17    |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
```
The results are typical: the truncated Newton method is faster in terms of function value and gradient evaluations, but it requires the most in terms of processing time while the quasi-Newton methods and CGD and L-BFGS algorithms are fairly close. The steepest gradient descent method needs as expected many more iterations to converge. Note that BFGS scales quadratically with the problem size, while CGD and L-BFGS scale linearly and are thus recommended for very large problems.


##### Compare line-search methods on convex smooth problems

```
./build/libnano/gcc-release/app/bench_solver \
    --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "cgd-pr|lbfgs|bfgs|hoshino|fletcher|newton" \
    --lsearchk ".+" \
    --trials 20 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 34
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| solver                           | precision | rank  | value        | log10(gtest) | errors | maxits | fcalls | gcalls | hcalls | [ms]  |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| newton [quadratic,fletcher]      | -7.0000   | 7.65  | N/A          | -11.6398     | 0      | 0      | 6      | 3      | 2      | 3     |
| newton [quadratic,backtrack]     | -7.0000   | 8.60  | N/A          | -11.6492     | 0      | 0      | 6      | 3      | 2      | 3     |
| newton [quadratic,cgdescent]     | -7.0000   | 9.42  | N/A          | -11.8051     | 0      | 0      | 6      | 3      | 2      | 3     |
| newton [quadratic,lemarechal]    | -7.0000   | 10.46 | N/A          | -11.6464     | 0      | 0      | 6      | 3      | 2      | 3     |
| newton [quadratic,morethuente]   | -7.0000   | 11.34 | N/A          | -11.7017     | 0      | 0      | 6      | 3      | 2      | 3     |
| cgd-pr [quadratic,cgdescent]     | -7.0000   | 15.85 | N/A          | -8.53309     | 0      | 0      | 34     | 34     | 0      | 1     |
| cgd-pr [quadratic,morethuente]   | -7.0000   | 16.33 | N/A          | -8.39877     | 0      | 0      | 27     | 27     | 0      | 1     |
| lbfgs [quadratic,cgdescent]      | -7.0000   | 18.21 | N/A          | -9.84905     | 0      | 0      | 24     | 24     | 0      | 1     |
| lbfgs [quadratic,morethuente]    | -7.0000   | 16.91 | N/A          | -9.63575     | 0      | 0      | 23     | 23     | 0      | 1     |
| bfgs [quadratic,cgdescent]       | -7.0000   | 17.09 | N/A          | -8.66312     | 0      | 0      | 34     | 34     | 0      | 5     |
| hoshino [quadratic,cgdescent]    | -7.0000   | 17.02 | N/A          | -8.92809     | 0      | 0      | 38     | 38     | 0      | 9     |
| fletcher [quadratic,cgdescent]   | -7.0000   | 18.25 | N/A          | -8.84129     | 0      | 0      | 38     | 38     | 0      | 5     |
| lbfgs [quadratic,lemarechal]     | -6.9977   | 16.59 | N/A          | -9.09815     | 14     | 0      | 27     | 27     | 0      | 1     |
| lbfgs [quadratic,fletcher]       | -6.9969   | 16.31 | N/A          | -8.70927     | 17     | 0      | 24     | 24     | 0      | 1     |
| fletcher [quadratic,morethuente] | -6.9918   | 17.44 | N/A          | -8.55925     | 0      | 62     | 98     | 98     | 0      | 11    |
| cgd-pr [quadratic,fletcher]      | -6.9914   | 15.50 | N/A          | -8.12543     | 61     | 0      | 35     | 35     | 0      | 1     |
| cgd-pr [quadratic,lemarechal]    | -6.9907   | 16.99 | N/A          | -8.13962     | 50     | 0      | 45     | 45     | 0      | 2     |
| hoshino [quadratic,morethuente]  | -6.9891   | 16.79 | N/A          | -8.53061     | 0      | 71     | 103    | 103    | 0      | 18    |
| bfgs [quadratic,morethuente]     | -6.9887   | 16.97 | N/A          | -8.47358     | 0      | 58     | 84     | 84     | 0      | 12    |
| lbfgs [quadratic,backtrack]      | -6.9855   | 15.58 | N/A          | -9.19055     | 0      | 70     | 70     | 70     | 0      | 5     |
| bfgs [quadratic,backtrack]       | -6.9511   | 16.25 | N/A          | -8.18337     | 0      | 202    | 205    | 205    | 0      | 26    |
| cgd-pr [quadratic,backtrack]     | -6.9508   | 15.48 | N/A          | -8.13753     | 0      | 100    | 294    | 294    | 0      | 17    |
| fletcher [quadratic,backtrack]   | -6.9485   | 17.02 | N/A          | -8.51573     | 0      | 219    | 257    | 257    | 0      | 30    |
| hoshino [quadratic,backtrack]    | -6.9473   | 15.99 | N/A          | -8.56424     | 0      | 216    | 259    | 259    | 0      | 49    |
| bfgs [quadratic,lemarechal]      | -6.9246   | 17.09 | N/A          | -8.04096     | 292    | 0      | 71     | 71     | 0      | 10    |
| hoshino [quadratic,lemarechal]   | -6.9153   | 16.78 | N/A          | -7.98973     | 341    | 0      | 74     | 74     | 0      | 15    |
| fletcher [quadratic,lemarechal]  | -6.9052   | 17.72 | N/A          | -7.98371     | 378    | 0      | 72     | 72     | 0      | 9     |
| bfgs [quadratic,fletcher]        | -6.8971   | 16.52 | N/A          | -7.99241     | 370    | 0      | 60     | 60     | 0      | 9     |
| hoshino [quadratic,fletcher]     | -6.8736   | 15.95 | N/A          | -7.95861     | 499    | 0      | 62     | 62     | 0      | 14    |
| fletcher [quadratic,fletcher]    | -6.8657   | 16.88 | N/A          | -7.90974     | 536    | 0      | 60     | 60     | 0      | 8     |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|--------|-------|
```
Notice that the CG-descent and the More&Thuente line-search methods are the most efficient ones across all solvers.


Very precise solutions can be obtained efficiently only using the CG-descent line-search method:
```
./build/libnano/gcc-release/app/bench_solver \
    --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --trials 20 --max-table-name 32 \
    --solver "cgd-pr|lbfgs|bfgs|hoshino|fletcher|newton" \
    --lsearchk "morethuente|cgdescent" \
    --solver::epsilon 1e-14 --solver::max_evals 10000 | tail -n 16
cosmin@fedora:~/libnano$ time bash scripts/bench_solver_cgdescent.sh
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| solver                           | precision | rank  | value        | log10(gtest) | errors | maxits | fcalls | gcalls | hcalls | [ms]  |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| newton [quadratic,cgdescent]     | -14.0000  | 5.46  | N/A          | -14.9295     | 0      | 0      | 21     | 18     | 3      | 5     |
| lbfgs [quadratic,cgdescent]      | -14.0000  | 7.71  | N/A          | -14.6943     | 0      | 0      | 129    | 129    | 0      | 8     |
| hoshino [quadratic,cgdescent]    | -14.0000  | 9.56  | N/A          | -14.6711     | 0      | 0      | 198    | 198    | 0      | 24    |
| fletcher [quadratic,cgdescent]   | -14.0000  | 10.55 | N/A          | -14.6765     | 0      | 0      | 199    | 199    | 0      | 19    |
| cgd-pr [quadratic,cgdescent]     | -13.9883  | 6.74  | N/A          | -14.7252     | 0      | 20     | 179    | 179    | 0      | 10    |
| bfgs [quadratic,cgdescent]       | -13.9771  | 8.70  | N/A          | -14.5484     | 0      | 283    | 564    | 564    | 0      | 41    |
| newton [quadratic,morethuente]   | -13.3869  | 5.01  | N/A          | -14.0709     | 0      | 973    | 1188   | 1173   | 15     | 116   |
| lbfgs [quadratic,morethuente]    | -12.6061  | 5.01  | N/A          | -13.0693     | 0      | 1775   | 2230   | 2230   | 0      | 184   |
| bfgs [quadratic,morethuente]     | -12.0493  | 4.54  | N/A          | -12.3091     | 0      | 2388   | 3049   | 3049   | 0      | 250   |
| hoshino [quadratic,morethuente]  | -12.0287  | 5.11  | N/A          | -12.3193     | 0      | 2358   | 3007   | 3007   | 0      | 253   |
| fletcher [quadratic,morethuente] | -12.0071  | 5.59  | N/A          | -12.2838     | 0      | 2441   | 3092   | 3092   | 0      | 245   |
| cgd-pr [quadratic,morethuente]   | -11.9273  | 4.04  | N/A          | -12.2333     | 0      | 2447   | 3014   | 3014   | 0      | 231   |
|----------------------------------|-----------|-------|--------------|--------------|--------|--------|--------|--------|--------|-------|
```
Notice that the CG-descent line-search method is the only one that doesn't fail to reach such a precision and also with around 10 times fewer function calls.


##### Compare solvers on convex non-smooth problems

The line-search monotonic solvers (like L-BFGS) are not guaranteed to converge for non-smooth problems. As such non-monotonic solvers may be more appropriate in this case.

```
./build/libnano/gcc-release/app/bench_solver \
    --min-dims 100 --max-dims 100 --function-type convex \
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
./build/libnano/gcc-release/app/bench_solver \
    --min-dims 100 --max-dims 100 --function-type convex-smooth \
    --solver "lbfgs|hoshino" \
    --solver::lbfgs::history 5 \
    --trials 20 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 6
for hsize in 10 20 50 100 150 200
do
    ${BENCH} \
        --min-dims 100 --max-dims 100 --function-type convex-smooth \
        --solver "lbfgs|hoshino" \
        --solver::lbfgs::history ${hsize} \
        --trials 20 --solver::epsilon 1e-7 --solver::max_evals 5000 | tail -n 3
done
cosmin@fedora:~/libnano$ time bash scripts/bench_solver_lbfgs_history.sh
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | log10(gtest) | errors | maxits | fcalls | gcalls | hcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -7.0000   | 1.48 | N/A          | -9.81196     | 0      | 0      | 30     | 30     | 0      | 2     |
| hoshino [quadratic,cgdescent]    | -7.0000   | 1.52 | N/A          | -8.92582     | 0      | 0      | 38     | 38     | 0      | 7     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -7.0000   | 1.46 | N/A          | -9.86578     | 0      | 0      | 27     | 27     | 0      | 1     |
| hoshino [quadratic,cgdescent]    | -7.0000   | 1.54 | N/A          | -8.91147     | 0      | 0      | 38     | 38     | 0      | 7     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -7.0000   | 1.47 | N/A          | -9.84288     | 0      | 0      | 25     | 25     | 0      | 1     |
| hoshino [quadratic,cgdescent]    | -7.0000   | 1.53 | N/A          | -8.93516     | 0      | 0      | 38     | 38     | 0      | 7     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -7.0000   | 1.46 | N/A          | -9.86558     | 0      | 0      | 24     | 24     | 0      | 1     |
| hoshino [quadratic,cgdescent]    | -7.0000   | 1.54 | N/A          | -8.91544     | 0      | 0      | 38     | 38     | 0      | 7     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -7.0000   | 1.46 | N/A          | -9.85619     | 0      | 0      | 24     | 24     | 0      | 1     |
| hoshino [quadratic,cgdescent]    | -7.0000   | 1.54 | N/A          | -8.93524     | 0      | 0      | 37     | 37     | 0      | 7     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -7.0000   | 1.46 | N/A          | -9.84131     | 0      | 0      | 24     | 24     | 0      | 1     |
| hoshino [quadratic,cgdescent]    | -7.0000   | 1.54 | N/A          | -8.88926     | 0      | 0      | 38     | 38     | 0      | 7     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
| lbfgs [quadratic,cgdescent]      | -7.0000   | 1.47 | N/A          | -9.85997     | 0      | 0      | 24     | 24     | 0      | 1     |
| hoshino [quadratic,cgdescent]    | -7.0000   | 1.53 | N/A          | -8.93451     | 0      | 0      | 38     | 38     | 0      | 7     |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|--------|-------|
```
The performance of L-BFGS saturates at 20 past gradients kept in the model.
