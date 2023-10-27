#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief optimal subgradient algorithm (OSGA).
///
/// see (1) "OSGA: A fast subgradient algorithm with optimal complexity", by A. Neumaier, 2014
/// see (2) "Optimal subgradient algorithms with application to large-scale linear inverse problems", by Ahookhosh, 2014
/// see (3) "An optimal subgradient algorithm for large-scale bound-constrained convex optimization", by A, N, 2015
/// see (4) "An optimal subgradient algorithm for large-scale convex optimization in simple domains", by A, N, 2015
///
/// NB: the implementation follows the notation from (1).
/// NB: the functional constraints (if any) are all ignored.
/// NB: the iterations are stopped when either eta is smaller than epsilon or
///     no significant decrease in the function value in the recent iterations.
///
class NANO_PUBLIC solver_osga_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_osga_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;
};
} // namespace nano
