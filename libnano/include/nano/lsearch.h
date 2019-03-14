#pragma once

#include <nano/json.h>
#include <nano/factory.h>
#include <nano/solver_state.h>
#include <nano/lsearch_step.h>

namespace nano
{
    class lsearch_init_t;
    class lsearch_strategy_t;

    using lsearch_init_factory_t = factory_t<lsearch_init_t>;
    using lsearch_strategy_factory_t = factory_t<lsearch_strategy_t>;

    using rlsearch_init_t = lsearch_init_factory_t::trobject;
    using rlsearch_strategy_t = lsearch_strategy_factory_t::trobject;

    ///
    /// \brief returns the registered line-search algorithms.
    ///
    NANO_PUBLIC lsearch_init_factory_t& get_lsearch_inits();
    NANO_PUBLIC lsearch_strategy_factory_t& get_lsearch_strategies();

    ///
    /// \brief compute the initial step length of the line search procedure.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
    ///
    class lsearch_init_t : public json_configurable_t
    {
    public:

        lsearch_init_t() = default;

        ///
        /// \brief returns the initial step length given the current state
        /// NB: may keep track of the previous states
        ///
        scalar_t get(const solver_state_t& state)
        {
            return get(state, m_iteration ++);
        }

    private:

        virtual scalar_t get(const solver_state_t&, const int iteration) = 0;

        // attributes
        int         m_iteration{0}; ///<
    };

    ///
    /// \brief compute the step length of the line search procedure.
    ///
    class lsearch_strategy_t : public json_configurable_t
    {
    public:

        ///
        /// logging operator: op(solver_state_t), called for each trial of the line-search length.
        ///
        using logger_t = std::function<void(const solver_state_t&)>;

        ///
        /// \brief
        ///
        virtual bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t&) = 0;

        ///
        /// \brief change parameters
        ///
        auto& c1(const scalar_t c1) { m_c1 = c1; return *this; }
        auto& c2(const scalar_t c2) { m_c2 = c2; return *this; }
        auto& logger(const logger_t& logger) { m_logger = logger; return *this; }
        auto& max_iterations(const int max_iterations) { m_max_iterations = max_iterations; return *this; }

        ///
        /// \brief access functions
        ///
        auto c1() const { return m_c1; }
        auto c2() const { return m_c2; }
        auto max_iterations() const { return m_max_iterations; }

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
        /// \brief log the current line-search trial length (if the logger is provided)
        ///
        void log(const solver_state_t& state) const
        {
            if (m_logger)
            {
                m_logger(state);
            }
        }

    private:

        // attributes
        scalar_t    m_c1{static_cast<scalar_t>(1e-4)};      ///< sufficient decrease rate
        scalar_t    m_c2{static_cast<scalar_t>(0.1)};       ///< sufficient curvature
        int         m_max_iterations{40};                   ///< #maximum iterations
        logger_t    m_logger;                               ///<
    };
}
