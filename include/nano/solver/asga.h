#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief accelerated subgradient algorithm (ASGA-2).
    ///     see "Accelerated first-order methods for large-scale convex minimization", by M. Ahookhosh, 2016
    ///
    /// NB: the algorithm was designed to minimize a structured problem,
    ///     but here it is applied to a sub-differentiable function directly.
    ///
    class NANO_PUBLIC solver_asga2_t final : public solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_asga2_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;

    private:

        // attributes
        sparam1_t   m_gamma1{"asga2::gamma1", 1.0, LT, 1.1, LT, 10.0};          ///<
        sparam1_t   m_gamma2{"asga2::gamma2", 0.0, LT, 0.9, LT, 1.00};          ///<
        iparam1_t   m_lsearch_max_iterations{"asga2::lsearch_max_iterations", 10, LE, 100, LE, 200};    ///<
    };

    ///
    /// \brief accelerated subgradient algorithm (ASGA-4).
    ///     see "Accelerated first-order methods for large-scale convex minimization", by M. Ahookhosh, 2016
    ///
    /// NB: the algorithm was designed to minimize a structured problem,
    ///     but here it is applied to a sub-differentiable function directly.
    ///
    class NANO_PUBLIC solver_asga4_t final : public solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_asga4_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;

    private:

        // attributes
        sparam1_t   m_gamma1{"asga4::gamma1", 1.0, LT, 1.1, LT, 10.0};          ///<
        sparam1_t   m_gamma2{"asga4::gamma2", 0.0, LT, 0.6, LT, 1.0};           ///<
        iparam1_t   m_lsearch_max_iterations{"asga4::lsearch_max_iterations", 10, LE, 100, LE, 200};    ///<
    };
}
