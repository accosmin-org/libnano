# Linear and quadratic programming module


#### Introduction

Libnano provides various methods to solve linear and quadratic programs. The goal is to find the minimum `x` of either a linear or a quadratic combination of its components, contrained optionally with affine equalities and inequalities. The returned point `x` is guaranteed to be the global minimum if the program is feasible. More details can be found in the following sources:


* "Primal-dual interior-point methods", S. Wright, 1997
* "Convex Optimization", S. Boyd, L. Vandenberghe, 2004
* "Numerical Optimization", J. Nocedal, S. Wright, 2006


#### Program definition

Linear and quadratic programs can be defined using the associated classes [linear_program_t](../include/nano/function/linear.h) and [quadratic_program_t](../include/nano/function/quadratic.h). Optionally arbitrarily many linear equality and inequality constraints can be chained using builtin C++ operator overloads. Following we show how to define various programs using the C++ interface:

* the `standard linear program` consists of equality contraints and positive element-wise solutions:
```
 min  c.dot(x)                          auto program = linear_program_t{"lp", c};
 s.t. A * x = b,                        A * program.variable() == b;
      x >= 0.                           program.variable() >= 0.0;
```

* the `inequality linear program` consists of only inequality constraints:
```
 min  c.dot(x)                          auto program = linear_program_t{"lp", c};
 s.t. A * x <= b.                       A * program.variable() <= b;
```

* the `general linear program` consists of both equality and inequality constraints:
```
 min  c.dot(x)                          auto program = linear_program_t{"lp", c};
 s.t. A * x = b,                        A * program.variable() == b;
      G * x <= h.                       G * program.variable() <= h;
```

* the `rectangle linear program` consists of element-wise inequality constraints:
```
 min  c.dot(x)                          auto program = linear_program_t{"lp", c};
 s.t. A * x = b,                        A * program.variable() == b;
      l <= x <= u.                      l <= program.variable(); program.variable() <= u;
```

* the `general quadratic program` consists of both equality and inequality constraints:
```
 min  1/2 x.dot(Q * x) + c.dot(x)       auto program = quadratic_program_t{"qp", Q, c};
 s.t. A * x = b,                        A * program.variable() == b;
      G * x <= h.                       G * program.variable() <= h;
```
Note that all equalities and inequalities above are specified element-wise.


More specifically the following example code extracted from the [linear programming example](../example/src/linprog.cpp):
```
const auto n_equals = 2;
const auto c        = make_vector<scalar_t>(1, 1, 1);
const auto A        = make_matrix<scalar_t>(n_equals, 2, 1, 0, 1, 0, 1);
const auto b        = make_vector<scalar_t>(4, 1);

auto program = linear_program_t{"lp", c};
A * program.variable() == b;
program.variable() >= 0.0;
```
illustrates how to define the following standard-form linear program:
```
 min  x1 + x2 + x3
 s.t. 2 * x1 + x2 = 4,
      x1 + x3 = 1,
      (x1, x2, x3) >= 0.
```
with the obvious solution `(1, 2, 0)`.


#### Solution

The linear and the quadratic programs are solved typically with variations of either the simplex method or the `primal-dual interior point method`. Libnano implements the latter as described in the references above. This variation is relatively easy to code and understand and it can be easily tuned for specific applications.

The following example code extracted from the [linear programming example](../example/src/linprog.cpp) shows how to solve the linear program defined above:
```
auto solver = solver_t::all().get("ipm");
assert(solver != nullptr);
solver->parameter("solver::epsilon")   = 1e-12;
solver->parameter("solver::max_evals") = 100;

auto program = linear_program_t{"lp", c};
critical(A * program.variable() == b);
critical(program.variable() >= 0.0);

const auto logger = make_stdout_logger();
const auto state  = solver->minimize(program, make_random_vector<scalar_t>(program.size()), logger);

std::cout << std::fixed << std::setprecision(12) << "solution: x=" << state.x().transpose() << std::endl;

assert(state.status() == solver_status::kkt_optimality_test);
assert(close(state.x(), xbest, 1e-10));
```

Please refer to the [quadratic programming example](../example/src/quadprog.cpp) for a similar example on how to solve quadratic programs.


#### Hints

* Typically the solution is found with 8-10 decimals in less than 20 iterations with the default settings. If convergence is not achieved, then the program is most likely unfeasible or unbounded.

* Convexity can be checked for quadratic problems using the `is_convex(Q)` function.

* If the number of equality constraints is larger than the number of variables or the constraints may be linear dependent, then it is better to call `reduce(A, b)` to transform the equality constraints `A` in a full row-rank matrix. This speeds-up the solver and often improves accuracy.


#### Examples

The command line utility [app/bench_solver](../app/bench_solver.cpp) is useful for benchmarking the builtin optimization algorithms on linear and quadratic programs of various dimensions. See bellow several such experiments.


##### Compare solvers

The most efficient solvers - the primal dual interior point method and the augmented lagrangian methods, can be evaluated on the builtin randomly generated linear programs for a given dimension size.
```
./build/libnano/gcc-release/app/bench_solver --min-dims 100 --max-dims 100 --function-type linear-program \
    --solver "ipm|augmented-lagrangian" --trials 128 | tail -n 6
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| solver                           | precision | rank | value        | kkt test     | errors | maxits | fcalls | gcalls | [ms]  |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
| ipm                              | -8.0000   | 1.62 | N/A          | 1.30231e-13  | 0      | 0      | 29     | 29     | 13    |
| augmented-lagrangian             | -7.8905   | 1.38 | N/A          | 8.40402e-08  | 0      | 128    | 1682   | 1682   | 28    |
|----------------------------------|-----------|------|--------------|--------------|--------|--------|--------|--------|-------|
```

The primal-dual interior point method is significantly more accurate and requires far fewer iterations. Note that the KKT optimality test is used as the stopping criterion as both methods supports this criterion.


##### TODO: Add the benchmark for quadratic programs once

##### TODO: Also compare with penalty methods

##### TODO: Compare solvers for large programs (1K, 10K dimensions)
