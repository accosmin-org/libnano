#pragma once

#include <nano/json.h>
#include <nano/factory.h>
#include <nano/solver_state.h>
#include <nano/lsearch_step.h>

namespace nano
{
    class lsearch_algo_t;
    using lsearch_algo_factory_t = factory_t<lsearch_algo_t>;
    using rlsearch_algo_t = lsearch_algo_factory_t::trobject;

    ///
    /// \brief returns the registered line-search algorithms.
    ///
    NANO_PUBLIC lsearch_algo_factory_t& get_lsearch_algos();

    ///
    /// \brief compute the step length of the line search procedure.
    ///
    class lsearch_algo_t : public json_configurable_t
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
