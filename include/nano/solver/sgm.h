#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief the sub-gradient method.
    ///     see "Introductory lectures on convex optimization: A basic course", by Y. Nesterov, 2004
    ///
    /// NB: the functional constraints (if any) are all ignored.
    /// NB: the algorithm is very slow and it is provided as a baseline.
    /// NB: the iterations are stopped when no significant decrease in the function value in the recent iterations.
    ///
    class NANO_PUBLIC solver_sgm_t final : public solver_t
    {
    public:
        ///
        /// \brief constructor
        ///
        solver_sgm_t();

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
