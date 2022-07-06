#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief returns whether the given constraint is convex.
    ///
    NANO_PUBLIC bool convex(const constraint_t&);

    ///
    /// \brief returns whether the given constraint is smooth.
    ///
    NANO_PUBLIC bool smooth(const constraint_t&);

    ///
    /// \brief returns whether the strong convexity coefficient of the given constraint.
    ///
    NANO_PUBLIC scalar_t strong_convexity(const constraint_t&);

    ///
    /// \brief returns how much a point violates the given constraint (the larger, the worse).
    ///
    NANO_PUBLIC scalar_t valid(const vector_t&, const constraint_t&);

    ///
    /// \brief evaluate the given constraint's function value at the given point
    ///     (and its gradient or sub-gradient if not smooth).
    ///
    NANO_PUBLIC scalar_t vgrad(const constraint_t&, const vector_t& x, vector_t* gx = nullptr);

    ///
    /// \brief compute the gradient accuracy (given vs. central finite difference approximation).
    ///
    NANO_PUBLIC scalar_t grad_accuracy(const function_t&, const vector_t& x);

    ///
    /// \brief check if the function is convex along the [x1, x2] line.
    ///
    NANO_PUBLIC bool is_convex(const function_t&, const vector_t& x1, const vector_t& x2, int steps);
}
