#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief sub-gradient method with an adaptive step length strategy:
    ///     - proportional to the current estimate of the magnitude of the sub-gradients (~Lipschitz constant) and
    ///     - decreasing geometric if no significant decrease in the past iterations.
    ///
    class NANO_PUBLIC solver_asgm_t final : public solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_asgm_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;

    private:

        // attributes
        iparam1_t   m_patience{"asgm::patience", 3, LE, 3, LE, 100};    ///<
        sparam1_t   m_gamma{"asgm::gamma", 1.0, LT, 5.0, LE, 10.0};     ///<
    };
}
