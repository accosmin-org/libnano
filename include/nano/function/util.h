#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief compute the gradient accuracy (given vs. central finite difference approximation).
///
NANO_PUBLIC scalar_t grad_accuracy(const function_t&, const vector_t& x, scalar_t desired_epsilon = 0.0);

///
/// \brief check if the function is convex along the [x1, x2] line.
///
NANO_PUBLIC bool is_convex(const function_t&, const vector_t& x1, const vector_t& x2, int steps,
                           scalar_t epsilon = epsilon1<scalar_t>());

///
/// \brief return true if the equality constraint `Ax = b` is not full row rank.
///
/// in this case the constraints are transformed in-place to obtain row-independant linear constraints
///     by performing an appropriate matrix decomposition.
///
NANO_PUBLIC bool reduce(matrix_t& A, vector_t& b);

///
/// \brief return the minimum eigen value of the given squared matrix.
///
/// NB: the matrix is positive semi-definite (and thus a quadratic convex objective) if all eigen values are positive.
///
NANO_PUBLIC scalar_t min_eigval(matrix_cmap_t);
} // namespace nano
