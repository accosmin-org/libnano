#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief optimal subgradient algorithm (OSGA).
    ///     see "OSGA: A fast subgradient algorithm with optimal complexity", by A. Neumaier, 2014
    ///     see "Optimal subgradient algorithms with application to large-scale linear inverse problems", by M.
    ///     Ahookhosh, 2014 see "An optimal subgradient algorithm for large-scale bound-constrained convex
    ///     optimization", by M. Ahookhosh, A. Neumaier, 2015 see "An optimal subgradient algorithm for large-scale
    ///     convex optimization in simple domains", by M. Ahookhosh, A. Neumaier, 2015
    ///
    /// NB: the functional constraints (if any) are all ignored.
    ///
    class NANO_PUBLIC solver_osga_t final : public solver_t
    {
    public:
        ///
        /// \brief constructor
        ///
        solver_osga_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;
    };
} // namespace nano
