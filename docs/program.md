# Linear and quadratic programming module


#### Introduction

Libnano provides various methods to solve linear and quadratic programs. The goal is to find the minimum `x` of either a linear or a quadratic combination of its components, contrained optionally with affine equalities and inequalities. The returned point `x` is guaranteed to be the global minimum if the program is feasible. More details can be found in the following sources:


* "Primal-dual interior-point methods", S. Wright, 1997
* "Convex Optimization", S. Boyd, L. Vandenberghe, 2004
* "Numerical Optimization", J. Nocedal, S. Wright, 2006


#### Definition

Linear and quadratic programs can be defined using the associated classes [linear_program_t](../include/nano/program/linear.h) and [quadratic_program_t](../include/nano/program/quadratic.h). Optionally arbitrarily many linear equality and inequality constraints can be chained using the `&` operator. See below for some examples:


* the `standard linear program` consists of equality contraints and positive element-wise solutions:
```
 min  c.dot(x)
 s.t. A * x = b,
      x >= 0.
```
and it can be defined in C++ as:
```
const auto program = linear_program_t{c} & equality_t{A, b} & inequality_t::greater(c.size(), 0.0);
```


* the `inequality linear program` consists of only inequality constraints:
```
 min  c.dot(x)
 s.t. A * x <= b.
```
and it can be defined in C++ as:
```
const auto program = linear_program_t{c} & inequality_t{A, b};
```


* the `general linear program` consists of both equality and inequality constraints:
```
 min  c.dot(x)
 s.t. A * x = b,
      G * x <= h.
```
and it can be defined in C++ as:
```
const auto program = linear_program_t{c} & equality_t{A, b} & inequality_t{G, h};
```


* the `rectangle linear program` consists of element-wise inequality constraints:
```
 min  c.dot(x)
 s.t. A * x = b,
      l <= x <= u.
```
and it can be defined in C++ as:
```
const auto program = linear_program_t{c} & equality_t{A, b} & inequality_t::from_rectangle(l, u);
```


* the `general quadratic program` consists of both equality and inequality constraints:
```
 min  1/2 x.dot(Q * x) + c.dot(x)
 s.t. A * x = b,
      G * x <= h.
```
and it can be defined in C++ as:
```
const auto program = quadratic_program_t{Q, c} & equality_t{A, b} & inequality_t{G, h};
```


Note that all equalities and inequalities above are specified element-wise.


The following example code extracted from the [example](../example/src/linprog.cpp):
```
const auto n_equals = 2;
const auto c        = make_vector<scalar_t>(1, 1, 1);
const auto A        = make_matrix<scalar_t>(n_equals, 2, 1, 0, 1, 0, 1);
const auto b        = make_vector<scalar_t>(4, 1);

const auto program  = linear_program_t{c} & equality_t{A, b} & inequality_t::greater(c.size(), 0.0);
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

The following example code extracted from the [example](../example/src/linprog.cpp) shows how to solve the linear program defined above:
```
const auto logger = [](const solver_state_t& state)
{
    std::cout << std::fixed << std::setprecision(12) << "i=" << state.m_iters << ",fx=" << state.m_fx
              << ",eta=" << state.m_eta << ",rdual=" << state.m_rdual.lpNorm<Eigen::Infinity>()
              << ",rcent=" << state.m_rcent.lpNorm<Eigen::Infinity>()
              << ",rprim=" << state.m_rprim.lpNorm<Eigen::Infinity>() << ",rcond=" << state.m_ldlt_rcond
              << (state.m_ldlt_positive ? "(+)" : "(-)") << "[" << state.m_status << "]" << std::endl;
    return true;
};

auto solver                           = solver_t{logger};
solver.parameter("solver::epsilon")   = 1e-12;
solver.parameter("solver::max_iters") = 100;

const auto program = linear_program_t{c} & equality_t{A, b} & inequality_t::greater(c.size(), 0.0);
const auto state   = solver.solve(program);

std::cout << std::fixed << std::setprecision(12) << "solution: x=" << state.m_x.transpose() << std::endl;

assert(state.m_status == solver_status::converged);
assert(close(state.m_x, xbest, 1e-10));
```

Please refer to [example](../example/src/quadprog.cpp) for a similar example on how to solve quadratic programs.


#### Hints

Typically the solution is found with 8-10 decimals in less than 20 iterations with the default settings. If convergence is not achieved, then the program is most likely unfeasible or unbounded.


Convexity can be checked for quadratic problems using `program.convex()` function.


If the number of equality constraints is larger than the number of variables or the constraints may be linear dependent, then it is better to call `program.reduce()` to transform the equality constraints `A` in a full row-rank matrix. This speeds-up the solver and often improves accuracy.
