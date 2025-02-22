# Constrained optimization module


#### Introduction

In addition to linear and quadratic programs, libnano supports generic constrained optimization problems. The library implements linear, quadratic and non-linear equality and inequality constraints, but also more specific ones like box constraints. More details and examples can be found in the following sources:
* "Convex Optimization", S. Boyd, L. Vandenberghe, 2004
* "Numerical Optimization", J. Nocedal, S. Wright, 2006


#### Problem definition

This is achieved in two steps:
* first the objective is defined using a specific `function_t` implementation and
* second the constraints are added using `function_t::constrain()`.


#### Solution

The library implements several methods to solve generic constrained optimization problems. The `augmented lagrangian` method is the most efficient and it also provides the KKT optimality test as theoretically motivated stopping criterion. In addition the `external penalty methods` can also solve these problems, but they take more iterations and don't come with a proper stopping criterion.


#### Examples

Please refer to the [constrained example](../example/src/constrained.cpp) for an example on how to use the augmented lagrangian method to solve a constrained optimization problem.
