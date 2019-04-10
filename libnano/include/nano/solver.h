#pragma once

#include <nano/lsearch.h>
#include <nano/solver/function.h>

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

        using lsearch_init_logger_t = lsearch_init_t::logger_t;
        using lsearch_strategy_logger_t = lsearch_strategy_t::logger_t;

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
        solver_state_t minimize(const function_t&, const vector_t& x0) const;

        ///
        /// \brief configure
        ///
        json_t config() const override;
        void config(const json_t&) override;

        ///
        /// \brief set logging callbacks
        ///
        void logger(const logger_t& logger) { m_logger = logger; }
        void lsearch_init_logger(const lsearch_init_logger_t& logger) { m_lsearch_init_logger = logger; }
        void lsearch_strategy_logger(const lsearch_strategy_logger_t& logger) { m_lsearch_strategy_logger = logger; }

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
        void lsearch_strategy(const string_t& id, rlsearch_strategy_t&&);

        ///
        /// \brief change parameters
        ///
        void c1(const scalar_t tol) { m_c1 = tol; }
        void c2(const scalar_t tol) { m_c2 = tol; }
        void epsilon(const scalar_t epsilon) { m_epsilon = epsilon; }
        void max_iterations(const int max_iterations) { m_max_iterations = max_iterations; }

        ///
        /// \brief access functions
        ///
        auto c1() const { return m_c1; }
        auto c2() const { return m_c2; }
        auto epsilon() const { return m_epsilon; }
        auto max_iterations() const { return m_max_iterations; }
        const auto& lsearch_init_id() const { return m_lsearch_init_id; }
        const auto& lsearch_strategy_id() const { return m_lsearch_strategy_id; }

    protected:

        ///
        /// \brief minimize the given function starting from the initial point x0
        ///     and using the given line-search strategy
        ///
        virtual solver_state_t minimize(const solver_function_t&, const lsearch_t&, const vector_t& x0) const = 0;

        ///
        /// \brief log the current optimization state (if the logger is provided)
        ///
        bool log(solver_state_t& state) const;

        ///
        /// \brief check if the optimization is done (convergence or error) after an iteration
        ///
        bool done(const solver_function_t& function, solver_state_t& state, const bool iter_ok) const;

        ///
        /// \brief construct line-search
        ///
        rlsearch_init_t make_lsearch_init() const;
        rlsearch_strategy_t make_lsearch_strategy() const;

    private:

        // attributes
        scalar_t                    m_c1{0};                    ///<
        scalar_t                    m_c2{0};                    ///<
        scalar_t                    m_epsilon{1e-6};            ///< required precision (~magnitude of the gradient)
        int                         m_max_iterations{1000};     ///< maximum number of iterations
        logger_t                    m_logger;                   ///<
        string_t                    m_lsearch_init_id;          ///<
        rlsearch_init_t             m_lsearch_init;             ///<
        lsearch_init_logger_t       m_lsearch_init_logger;      ///<
        string_t                    m_lsearch_strategy_id;      ///<
        rlsearch_strategy_t         m_lsearch_strategy;         ///<
        lsearch_strategy_logger_t   m_lsearch_strategy_logger;  ///<
    };
}
