#pragma once

#include <nano/lsearch/init.h>
#include <nano/solver/function.h>
#include <nano/lsearch/strategy.h>

namespace nano
{
    class solver_t;
    using solver_factory_t = factory_t<solver_t>;
    using rsolver_t = solver_factory_t::trobject;

    ///
    /// \brief generic optimization algorithm typically using an adaptive line-search method.
    ///
    class NANO_PUBLIC solver_t : public json_configurable_t
    {
    public:

        ///
        /// logging operator: op(state), returns false if the optimization should stop
        ///
        using logger_t = std::function<bool(const solver_state_t&)>;

        ///
        /// \brief constructor
        ///
        solver_t(
            const scalar_t c1 = 1e-1,
            const scalar_t c2 = 9e-1,
            const string_t& lsearch_init = "quadratic",
            const string_t& lsearch_strategy = "morethuente");

        ///
        /// \brief returns the available implementations
        ///
        static solver_factory_t& all();

        ///
        /// \brief minimize the given function starting from the initial point x0 until:
        ///     - convergence is achieved (critical point, possiblly a local/global minima) or
        ///     - the maximum number of iterations is reached or
        ///     - the user canceled the optimization (using the logging function) or
        ///     - the solver failed (e.g. line-search failed)
        ///
        solver_state_t minimize(const function_t& f, const vector_t& x0) const
        {
            assert(f.size() == x0.size());
            return minimize(solver_function_t(f), x0);
        }

        ///
        /// \brief configure
        ///
        json_t config() const override;
        void config(const json_t&) override;

        ///
        /// \brief change the line-search initialization
        ///
        void lsearch_init(const json_t&);
        void lsearch_init(const string_t& id);
        void lsearch_init(const string_t& id, rlsearch_init_t&&);

        ///
        /// \brief change the line-search strategy
        ///
        void lsearch_strategy(const json_t&);
        void lsearch_strategy(const string_t& id);
        void lsearch_logger(const lsearch_strategy_t::logger_t&);
        void lsearch_strategy(const string_t& id, rlsearch_strategy_t&&);

        ///
        /// \brief change parameters
        ///
        void logger(const logger_t& logger) { m_logger = logger; }
        void epsilon(const scalar_t epsilon) { m_epsilon = epsilon; }
        void max_iterations(const int max_iterations) { m_max_iterations = max_iterations; }

        ///
        /// \brief access functions
        ///
        auto epsilon() const { return m_epsilon; }
        auto max_iterations() const { return m_max_iterations; }

    protected:

        ///
        /// \brief minimize the given function starting from the initial point x0
        ///
        virtual solver_state_t minimize(const solver_function_t&, const vector_t& x0) const = 0;

        ///
        /// \brief log the current optimization state (if the logger is provided)
        ///
        auto log(const solver_state_t& state) const
        {
            return !m_logger ? true : m_logger(state);
        }

        ///
        /// \brief update the current state using line-search
        ///
        bool lsearch(solver_state_t& state) const;

        ///
        /// \brief check if the optimization is done (convergence or error) after an iteration
        ///
        bool done(const solver_function_t& function, solver_state_t& state, const bool iter_ok) const;

    private:

        // attributes
        scalar_t            m_epsilon{1e-6};            ///< required precision (~magnitude of the gradient)
        int                 m_max_iterations{1000};     ///< maximum number of iterations
        logger_t            m_logger;                   ///<
        string_t            m_lsearch_init_id;          ///<
        string_t            m_lsearch_strategy_id;      ///<
        rlsearch_init_t     m_lsearch_init;             ///<
        rlsearch_strategy_t m_lsearch_strategy;         ///<
    };
}
