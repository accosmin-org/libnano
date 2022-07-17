#pragma once

#include <nano/function/penalty.h>
#include <nano/solver.h>

namespace nano
{
    // TODO: implement the penalty method from "A new exact penalty function", by W. Huyer and A. Neumaier, 2003.

    ///
    /// \brief interface for penalty methods to solve constrained optimization problem using a given solver.
    ///
    /// NB: the penalty method works by increasing the penalty term in the outer loop
    ///     and using the given solver to minimize the penalty function (the inner loop).
    ///
    class NANO_PUBLIC solver_penalty_t : public estimator_t
    {
    public:
        ///
        /// \brief constructor
        ///
        solver_penalty_t();

        ///
        /// \brief destructor
        ///
        virtual ~solver_penalty_t();

        ///
        /// \brief set the logging callback.
        ///
        void logger(const solver_t::logger_t& logger);

        ///
        /// \brief minimize the given constrained function starting from the initial point x0.
        ///
        virtual solver_state_t minimize(const solver_t&, const function_t&, const vector_t& x0) const = 0;

    protected:
        bool done(const solver_state_t& curr_state, solver_state_t& best_state, scalar_t epsilon) const;

    private:
        // attributes
        solver_t::logger_t m_logger; ///<
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
        /// \brief @see solver_penalty_t
        ///
        solver_state_t minimize(const solver_t&, const function_t&, const vector_t& x0) const override;
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
        /// \brief @see solver_penalty_t
        ///
        solver_state_t minimize(const solver_t&, const function_t&, const vector_t& x0) const override;
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
        /// \brief @see solver_penalty_t
        ///
        solver_state_t minimize(const solver_t&, const function_t&, const vector_t& x0) const override;
    };
} // namespace nano
