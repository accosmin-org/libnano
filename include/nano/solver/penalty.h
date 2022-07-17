#pragma once

#include <nano/function/penalty.h>
#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief base class for penalty methods to solve constrained optimization problem using a given solver.
    ///
    class NANO_PUBLIC solver_penalty_t : public estimator_t
    {
    public:
        ///
        /// \brief constructor
        ///
        solver_penalty_t();
    };

    ///
    /// \brief penalty method using the linear penalty function.
    ///
    /// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006
    ///
    /// NB: the penalty function is exact, but it is not smooth and thus the optimization is typically not very precise.
    ///
    class NANO_PUBLIC solver_linear_penalty_t final : public solver_penalty_t
    {
    public:
        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const solver_t&, const function_t&, const vector_t& x0) const;
    };

    ///
    /// \brief penalty method using the quadratic penalty function.
    ///
    /// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006
    ///
    /// NB: the penalty function is not exact, but it is smooth and thus the optimization is typically very precise.
    ///
    class NANO_PUBLIC solver_quadratic_penalty_t final : public solver_penalty_t
    {
    public:
        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const solver_t&, const function_t&, const vector_t& x0) const;
    };

    ///
    /// \brief penalty method using the epsilon-smoothed linear quadratic penalty.
    ///
    /// see "On smoothing exact penalty functions for convex constrained optimization", by M. Pinar, S. Zenios, 1994
    ///
    /// NB: the penalty function is exact and smooth and thus the optimization is typically very precise.
    ///
    class NANO_PUBLIC solver_linear_quadratic_penalty_t final : public solver_penalty_t
    {
    public:
        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const solver_t&, const function_t&, const vector_t& x0) const;
    };
} // namespace nano
