#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief optimal subgradient algorithm (OSGA).
    ///     see "OSGA: A fast subgradient algorithm with optimal complexity", by A. Neumaier, 2014
    ///     see "Optimal subgradient algorithms with application to large-scale linear inverse problems", by M. Ahookhosh, 2014
    ///     see "An optimal subgradient algorithm for large-scale bound-constrained convex optimization", by M. Ahookhosh, A. Neumaier, 2015
    ///     see "An optimal subgradient algorithm for large-scale convex optimization in simple domains", by M. Ahookhosh, A. Neumaier, 2015
    ///
    class NANO_PUBLIC solver_osga_t final : public solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_osga_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;

    private:

        // attributes
        sparam1_t   m_lambda{"osga::lambda", 0, LT, 0.9, LT, 1};            ///<
        sparam1_t   m_alpha_max{"osga::alpha_max", 0, LT, 0.7, LT, 1};      ///<
        sparam2_t   m_kappas{"osga::kappas", 0, LT, 0.5, LE, 0.5, LE, 1};   ///<
    };
}
