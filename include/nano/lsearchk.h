#pragma once

#include <nano/core/estimator.h>
#include <nano/core/factory.h>
#include <nano/solver/lstep.h>

namespace nano
{
    class lsearchk_t;
    using lsearchk_factory_t = factory_t<lsearchk_t>;
    using rlsearchk_t        = lsearchk_factory_t::trobject;

    ///
    /// \brief compute the step size along the given descent direction starting from the initial guess `t0`.
    ///
    /// NB: the returned step size is positive and guaranteed to decrease the function value (if no failure).
    ///
    class NANO_PUBLIC lsearchk_t : public estimator_t
    {
    public:
        using interpolation = lsearch_step_t::interpolation;

        ///
        /// logging operator: op(solver_state_at_0, solver_state_at_t), called for each trial of the line-search length.
        ///
        using logger_t = std::function<void(const solver_state_t&, const solver_state_t&)>;

        ///
        /// \brief constructor
        ///
        lsearchk_t();

        ///
        /// \brief returns the available implementations
        ///
        static lsearchk_factory_t& all();

        ///
        /// \brief clone the object, by keeping the parameters but with an internal "clean state"
        ///
        virtual rlsearchk_t clone() const = 0;

        ///
        /// \brief compute the step size starting from the given state and the initial estimate of the step size
        ///
        virtual bool get(solver_state_t& state, scalar_t t);

        ///
        /// \brief set the logging operator.
        ///
        void logger(const logger_t& logger) { m_logger = logger; }

        ///
        /// \brief minimum allowed line-search step.
        ///
        static scalar_t stpmin() { return scalar_t(10) * std::numeric_limits<scalar_t>::epsilon(); }

        ///
        /// \brief maximum allowed line-search step.
        ///
        static scalar_t stpmax() { return scalar_t(1) / stpmin(); }

    protected:
        ///
        /// \brief compute the step size given the previous state and the current state
        ///
        virtual bool get(const solver_state_t& state0, solver_state_t&) = 0;

        ///
        /// \brief log the current line-search trial length (if the logger is provided)
        ///
        void log(const solver_state_t& state0, const solver_state_t& state) const
        {
            if (m_logger)
            {
                m_logger(state0, state);
            }
        }

    private:
        // attributes
        logger_t m_logger; ///<
    };
} // namespace nano
