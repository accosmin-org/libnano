#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief the ellipsoid method.
    ///     see "Introductory lectures on convex optimization: A basic course", by Y. Nesterov, 2004
    ///
    /// NB: the functional constraints (if any) are all ignored.
    /// NB: the algorithm is sensitive to the initial radius of the hyper-ellipsoid.
    ///
    class NANO_PUBLIC solver_ellipsoid_t final : public solver_t
    {
    public:
        ///
        /// \brief constructor
        ///
        solver_ellipsoid_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t do_minimize(const function_t&, const vector_t& x0) const final;
    };
} // namespace nano
