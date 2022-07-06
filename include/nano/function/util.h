#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief compute the gradient accuracy (given vs. central finite difference approximation).
    ///
    NANO_PUBLIC scalar_t grad_accuracy(const function_t&, const vector_t& x);

    ///
    /// \brief check if the function is convex along the [x1, x2] line.
    ///
    NANO_PUBLIC bool is_convex(const function_t&, const vector_t& x1, const vector_t& x2, int steps);
} // namespace nano
