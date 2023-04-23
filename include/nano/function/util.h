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
NANO_PUBLIC bool is_convex(const function_t&, const vector_t& x1, const vector_t& x2, int steps);

///
/// \brief returns true whether the given quadratic term is convex (aka positive semi-definite).
///
NANO_PUBLIC bool convex(const matrix_t&);

///
/// \brief returns the strong convexity factor of a quadratic term, or zero if not convex.
///
NANO_PUBLIC scalar_t strong_convexity(const matrix_t&);
} // namespace nano
