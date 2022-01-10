# Numerical optimization module


#### Introduction

Libnano provides various methods to solve unconstrained non-linear numerical optimization problems. The goal is to find the minimum **x** of a differential function **f(x)** of dimension *n*. The returned point **x** is guaranteed to be the global minimum when the function is convex, and a critical point (not necessarily a local minimum) otherwise. **x** is found by iteratively decreasing the current function value from a given point **x0** using a descent direction and a line-search along this direction. More details can be found in the following sources:


* "Practical Methods of Optimization 2e", R. Fletcher, 2000
* "Introductory Lectures on Convex Optimization: A Basic Course", Y. Nesterov, 2003
* "Convex Optimization", S. Boyd, L. Vandenberghe, 2004
* "Numerical Optimization", J. Nocedal, S. Wright, 2006


Each concept involved in the optimization procedure is mapped to a particular interface. Most relevant are the [function_t](../include/nano/function.h) and the [solver_t](../include/nano/solver.h) interfaces. The builtin implementations can be accessed programatically in C++ using the associated factory.


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
        convex(true);
        smooth(true);
    }

    scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr, vgrad_config_t config = vgrad_config_t{}) const override
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
|----------|-----------------------------------------|
| solver   | description                             |
|----------|-----------------------------------------|
| bfgs     | quasi-newton method (BFGS)              |
| cgd      | conjugate gradient descent (default)    |
| cgd-cd   | conjugate gradient descent (CD)         |
| cgd-dy   | conjugate gradient descent (DY)         |
| cgd-dycd | conjugate gradient descent (DYCD)       |
| cgd-dyhs | conjugate gradient descent (DYHS)       |
| cgd-fr   | conjugate gradient descent (FR)         |
| cgd-hs   | conjugate gradient descent (HS+)        |
| cgd-ls   | conjugate gradient descent (LS+)        |
| cgd-n    | conjugate gradient descent (N+)         |
| cgd-pr   | conjugate gradient descent (PR+)        |
| cgd-prfr | conjugate gradient descent (FRPR)       |
| dfp      | quasi-newton method (DFP)               |
| fletcher | quasi-newton method (Fletcher's switch) |
| gd       | gradient descent                        |
| hoshino  | quasi-newton method (Hoshino formula)   |
| lbfgs    | limited-memory BFGS                     |
| sr1      | quasi-newton method (SR1)               |
|----------|-----------------------------------------|
```

The default configurations are close to optimal for most situations. Still the user is free to experiment with the available parameters. The following piece of code extracted from the [example](../example/src/minimize.cpp) shows how to create a L-BFGS solver and how to change the line-search strategy, the tolerance and the maximum number of iterations:
```
#include <nano/solver/lbfgs.h>

auto solver = nano::solver_lbfgs_t{};
solver.history(6);
solver.epsilon(1e-6);
solver.max_evals(100);
solver.tolerance(1e-4, 9e-1);
solver.lsearch0("constant");
solver.lsearchk("morethuente");
```

Then the optimal point is obtained by invoking the solver on the object like described below:
```
const auto x0 = nano::vector_t::Random(objective.size());
const auto state = solver.minimize(objective, x0);
const auto& x = state.x;

std::cout << std::fixed << std::setprecision(12)
    << "f0=" << objective.vgrad(x0, nullptr) << ", f=" << state.f
    << ", g=" << state.convergence_criterion()
    << ", x-x*=" << (state.x - objective.b()).lpNorm<Eigen::Infinity>()
    << ", iters=" << state.m_iterations
    << ", fcalls=" << state.m_fcalls
    << ", gcalls=" << state.m_gcalls
    << ", status=" << nano::to_string(state.m_status) << std::endl;
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


The command line utility [app/bench_solver](../app/bench_solver.cpp) is useful for benchmarking the builtin optimization algorithms on standard test functions. The following run compares 4 solvers on all convex builtin functions of dimensions from 16 to 32:
```
./build/libnano/release/app/bench_solver --min-dims 16 --max-dims 32 --convex --smooth --solver "gd|cgd|lbfgs|bfgs" --epsilon 1e-7 --trials 1000 --max-iterations 1000 | tail -n 8
|--------|-----------|-------------|-----------------------|--------|--------|---------|---------|---------|---------|------|------|
| Solver | lsearch0  | lsearchk    | gnorm                 | #fails | #iters | #errors | #maxits | #fcalls | #gcalls | cost | [ms] |
|--------|-----------|-------------|-----------------------|--------|--------|---------|---------|---------|---------|------|------|
| bfgs   | quadratic | morethuente | 3.900931337929577e-08 | 0      | 26     | 0       | 0       | 29      | 29      | 88   | 1561 |
| lbfgs  | quadratic | morethuente | 4.765617208344007e-08 | 0      | 36     | 0       | 0       | 42      | 42      | 128  | 230  |
| cgd    | quadratic | morethuente | 3.42032422701215e-08  | 0      | 21     | 0       | 0       | 61      | 61      | 184  | 136  |
| gd     | quadratic | morethuente | 0.01033748928933569   | 4353   | 301    | 0       | 4353    | 533     | 533     | 1599 | 905  |
|--------|-----------|-------------|-----------------------|--------|--------|---------|---------|---------|---------|------|------|
```
The results are typical: the BFGS algorithm is faster in terms of function value and gradient evaluations, but it requires the most in terms of processing time while the CGD and L-BFGS algorithms are close while being much faster. The steepest gradient descent method needs as expected many more iterations to converge.


#### Future work

* Implement sub-gradient methods
* Implement stochastic gradient (descent) methods
* Implement methods using second-order oracle (e.g. Newton method)
* Implement non-monotone line-search methods (e.g. Nesterov's accelerated gradient, Barzilai and Borwein method)
