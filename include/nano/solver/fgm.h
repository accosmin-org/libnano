#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief universal fast gradient method (FGM).
    ///     see "Universal Gradient Methods for Convex Optimization Problems", by Yu. Nesterov, 2013
    ///
    /// NB: the algorithm was designed to minimize a structured problem,
    ///     but here it is applied to a sub-differentiable function directly.
    ///
    class NANO_PUBLIC solver_fgm_t final : public solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_fgm_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;

    private:

        // attributes
        iparam1_t   m_lsearch_max_iterations{"fgm::lsearch_max_iterations", 10, LE, 20, LE, 30};    ///<
    };
}
