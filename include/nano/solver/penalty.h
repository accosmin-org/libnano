#pragma once

#include <nano/function/penalty.h>
#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief penalty method to solve constrained optimization problem using a given solver.
    ///     see "Numerical Optimization", by J. Nocedal, S. Wright, 2006
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

    ///
    /// \brief penalty method to solve constrained optimization problem using a given solver
    ///     adapted to the epsilon-smoothed linear quadratic penalty.
    ///     see "On smoothing exact penalty functions for convex constrained optimization", by M. Pinar, S. Zenios, 1994
    ///
    class NANO_PUBLIC solver_linear_quadratic_penalty_t final : public estimator_t
    {
    public:
        ///
        /// \brief constructor
        ///
        solver_linear_quadratic_penalty_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const solver_t&, const function_t&, const vector_t& x0) const;
    };
} // namespace nano
