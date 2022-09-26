#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief optimal subgradient algorithm (OSGA).
    ///     see (1) "OSGA: A fast subgradient algorithm with optimal complexity", by A. Neumaier, 2014
    ///     see (2) "Optimal subgradient algorithms with application to large-scale linear inverse problems",
    ///         by M. Ahookhosh, 2014
    ///     see (3) "An optimal subgradient algorithm for large-scale bound-constrained convex optimization",
    ///         by M. Ahookhosh, A. Neumaier, 2015
    ///     see (4) "An optimal subgradient algorithm for large-scale convex optimization in simple domains",
    ///         by M. Ahookhosh, A. Neumaier, 2015
    ///
    /// NB: the implementation follows the notation from (1).
    /// NB: the functional constraints (if any) are all ignored.
    /// NB: the convergence criterion is either that eta is smaller than epsilon0 or that the
    ///     the difference of two consecutive best updates is smaller than epsilon.
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
        rsolver_t clone() const final;

        ///
        /// \brief @see solver_t
        ///
        solver_state_t do_minimize(const function_t&, const vector_t& x0) const final;
    };
} // namespace nano
