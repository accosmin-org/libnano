# Linear programming module


#### Introduction

Libnano provides various methods to solve linear programming problems. The goal is to find the minimum **x** of a linear combination of its components, contrained optionally with affine equalities and inequalities. The returned point **x** is guaranteed to be the global minimum if the problem is feasible. More details can be found in the following sources:


* "Convex Optimization", S. Boyd, L. Vandenberghe, 2004
* "Numerical Optimization", J. Nocedal, S. Wright, 2006


#### Problem definition

The linear programming problems can be defined in various equivalent formulations. Libnano supports the following most common formulations:

* the **standard form** implemented by [linprog::problem_t](../include/nano/solver/linprog.h) with equality contraints and positive element-wise solutions:
```
 min  c.dot(x)
 s.t. A * x = b and x >= 0.
```

* the **inequality form** implemented by [linprog::inequality_problem_t](../include/nano/solver/linprog.h) with only inequality constraints:
```
 min  c.dot(x)
 s.t. A * x <= b.
```

* the **general form** implemented by [linprog::inequality_problem_t](../include/nano/solver/linprog.h) with both equality and inequality constraints:
```
 min  c.dot(x)
 s.t. A * x = b and G * x <= h.
```

Note that all inequalities above are specified element-wise. Additionally all non-standard problem formulations can be transformed to the standard form explicitly by calling the appropriate `transform` function. This is typically not necessary as the solver is performing the appropriate transformations if needed.


The following example code extracted from `[example](../example/src/linprog.cpp)`:
```
const auto n_equals = 2;
const auto c        = make_vector<scalar_t>(1, 1, 1);
const auto A        = make_matrix<scalar_t>(n_equals, 2, 1, 0, 1, 0, 1);
const auto b        = make_vector<scalar_t>(4, 1);

const auto problem = linprog::problem_t{c, A, b};
```

illustrates how to define the following standard-form linear programming problem:
```
 min  x1 + x2 + x3
 s.t. 2 * x1 + x2 = 4, x1 + x3 = 1, x1 >= 0, x2 >= 0, x3 >= 0.
```
with the obvious solution `(1, 2, 0)`.


#### Solution

The linear programming problems are solved typically with variations of either the simplex method or the **primal-dual interior point method**. Libnano implements the latter - the variant of the `Mehrotra` algorithm as described in the `Numerical Optimization` reference above. This variation is relatively easy to code and understand and it can be easily tuned for specific applications.

The following example code extracted from [example](../example/src/linprog.cpp) shows how to solve the linear programming problem defined above:
```
const auto logger = [](const linprog::problem_t& problem, const linprog::solution_t& solution)
{
    std::cout << std::fixed << std::setprecision(12) << "i=" << solution.m_iters << ",miu=" << solution.m_miu
              << ",KKT=" << solution.m_kkt << ",c.dot(x)=" << problem.m_c.dot(solution.m_x)
              << ",|Ax-b|=" << (problem.m_A * solution.m_x - problem.m_b).lpNorm<Eigen::Infinity>() << std::endl;
};

const auto problem  = linprog::problem_t{c, A, b};
const auto solution = linprog::solve(problem, logger);

std::cout << std::fixed << std::setprecision(12) << "solution: x=" << solution.m_x.transpose() << std::endl;
```

Note that typically the solution is found with 8-10 decimals in less than 10 iterations. Convergence can be checked by calling `solution.converged()` and if this is not the case then the problem is unfeasible.

Note that the solver has overloads for all supported non-standard formulations.
