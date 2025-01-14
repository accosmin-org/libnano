#pragma once

#include <nano/enum.h>

namespace nano
{
enum class solver_convergence : uint8_t
{
    ///< no theoretical motivated stopping criterion.
    ///< heuristically the algorithm stops when no sufficient decrease in the most recent number of iterations.
    ///< applicable to all problems: convex/non-convex, smooth/non-smooth, w/o constraints.
    value_test,

    ///< theoretical motivated stopping criterion: the relative magnitude of the gradient.
    ///< applicable only to smooth problems convex and non-convex, but without constraints.
    gradient_test,

    ///< theoretical motivated stopping criterion specific to a particular algorithm (and a class of functions).
    ///< e.g. an upper bound of the gap between the current point and the optimum.
    specific_test,

    ///< theoretical motivated stopping criterion: the KKT optimality conditions.
    ///< applicable only to constrained problems, but mostly useful for convex smooth constrained problems.
    kkt_optimality_test,
};

template <>
inline enum_map_t<solver_convergence> enum_string()
{
    return {
        {         solver_convergence::value_test,          "value-test"},
        {      solver_convergence::gradient_test,       "gradient-test"},
        {      solver_convergence::specific_test,       "specific-test"},
        {solver_convergence::kkt_optimality_test, "kkt-optimality-test"}
    };
}
}
