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
    template <typename tpenalty, std::enable_if_t<std::is_base_of_v<penalty_function_t, tpenalty>, bool> = true>
        > class NANO_PUBLIC solver_penalty_t
    {
    public:
        ///
        /// \brief constructor
        ///
        solver_penalty_t(rsolver_t&& solver);

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;

    private:
        // attributes
        rsolver_t m_solver; ///< base solver
    };

    using solver_linear_penalty_t    = solver_penalty_t<linear_penalty_function_t>;
    using solver_quadratic_penalty_t = solver_penalty_t<quadratic_penalty_function_t>;

} // namespace nano
