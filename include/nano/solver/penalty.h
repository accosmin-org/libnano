#pragma once

#include <nano/function/penalty.h>
#include <nano/solver.h>

namespace nano
{
    // TODO: solver_state_t like class to store results related to constraints - violation per constraint, Lagrangian
    // etc.
    // TODO: the penalty solver cannot be a true solver

    ///
    /// \brief penalty method to solve constrained optimization problem using
    ///     a given solver gradient descent with line-search.
    ///
    template <typename tpenalty>
    class NANO_PUBLIC solver_penalty_t final : public estimator_t
    {
    public:
        static_assert(std::is_base_of_v<penalty_function_t, tpenalty>);

        ///
        /// \brief constructor
        ///
        solver_penalty_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const solver_t&, const function_t&, const vector_t& x0) const;
    };

    using solver_linear_penalty_t    = solver_penalty_t<linear_penalty_function_t>;
    using solver_quadratic_penalty_t = solver_penalty_t<quadratic_penalty_function_t>;
} // namespace nano
