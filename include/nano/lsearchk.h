#pragma once

#include <nano/arch.h>
#include <nano/factory.h>
#include <nano/parameter.h>
#include <nano/solver/lstep.h>

namespace nano
{
    class lsearchk_t;
    using lsearchk_factory_t = factory_t<lsearchk_t>;
    using rlsearchk_t = lsearchk_factory_t::trobject;

    ///
    /// \brief compute the step size along the given descent direction starting from the initial guess `t0`.
    ///
    /// NB: the returned step size is positive and guaranteed to decrease the function value (if no failure).
    ///
    class NANO_PUBLIC lsearchk_t
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
        lsearchk_t() = default;

        ///
        /// \brief enable copying
        ///
        lsearchk_t(const lsearchk_t&) = default;
        lsearchk_t& operator=(const lsearchk_t&) = default;

        ///
        /// \brief enable moving
        ///
        lsearchk_t(lsearchk_t&&) noexcept = default;
        lsearchk_t& operator=(lsearchk_t&&) noexcept = default;

        ///
        /// \brief destructor
        ///
        virtual ~lsearchk_t() = default;

        ///
        /// \brief returns the available implementations
        ///
        static lsearchk_factory_t& all();

        ///
        /// \brief clone the object, by keeping the parameters but with an internal "clean state"
        ///
        [[nodiscard]] virtual rlsearchk_t clone() const = 0;

        ///
        /// \brief compute the step size starting from the given state and the initial estimate of the step size
        ///
        virtual bool get(solver_state_t& state, scalar_t t);

        ///
        /// \brief change parameters
        ///
        void logger(const logger_t& logger) { m_logger = logger; }
        void tolerance(const scalar_t c1, const scalar_t c2) { m_tolerance.set(c1, c2); }
        void max_iterations(const int max_iterations) { m_max_iterations.set(max_iterations); }

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto c1() const { return m_tolerance.get1(); }
        [[nodiscard]] auto c2() const { return m_tolerance.get2(); }
        [[nodiscard]] auto max_iterations() const { return m_max_iterations.get(); }

        ///
        /// \brief minimum allowed line-search step
        ///
        static scalar_t stpmin()
        {
            return scalar_t(10) * std::numeric_limits<scalar_t>::epsilon();
        }

        ///
        /// \brief maximum allowed line-search step
        ///
        static scalar_t stpmax()
        {
            return scalar_t(1) / stpmin();
        }

    protected:

        ///
        /// \brief compute the step size given the previous state and the current state
        ///
        virtual bool get(const solver_state_t& state0, solver_state_t&)
        {
            (void)state0;
            return false;
        }

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
        sparam2_t   m_tolerance{"lsearchk::tolerance", 0, LT, 1e-4, LT, 0.1, LT, 1};    ///< see Armijo-Wolfe conditions
        iparam1_t   m_max_iterations{"lsearchk::max_iterations", 1, LE, 40, LE, 100};   ///<
        logger_t    m_logger;                                                           ///<
    };
}
