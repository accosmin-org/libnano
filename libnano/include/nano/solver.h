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
            const string_t& lsearch0 = "quadratic",
            const string_t& lsearchk = "morethuente");

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
        /// \brief set the logging callbacks
        ///
        void logger(const logger_t& logger) { m_logger = logger; }
        void lsearch0_logger(const lsearch0_t::logger_t& logger) { m_lsearch0_logger = logger; }
        void lsearchk_logger(const lsearchk_t::logger_t& logger) { m_lsearchk_logger = logger; }

        ///
        /// \brief set the line-search initialization
        ///
        void lsearch0(const json_t&);
        void lsearch0(const string_t& id);
        void lsearch0(const string_t& id, rlsearch0_t&&);

        ///
        /// \brief set the line-search strategy
        ///
        void lsearchk(const json_t&);
        void lsearchk(const string_t& id);
        void lsearchk(const string_t& id, rlsearchk_t&&);

        ///
        /// \brief set parameters
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
        const auto& lsearch0_id() const { return m_lsearch0_id; }
        const auto& lsearchk_id() const { return m_lsearchk_id; }

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

    private:

        // attributes
        scalar_t                    m_c1{0};                    ///<
        scalar_t                    m_c2{0};                    ///<
        scalar_t                    m_epsilon{1e-6};            ///< required precision (~magnitude of the gradient)
        int                         m_max_iterations{1000};     ///< maximum number of iterations
        logger_t                    m_logger;                   ///<
        string_t                    m_lsearch0_id;              ///<
        rlsearch0_t                 m_lsearch0;                 ///<
        lsearch0_t::logger_t        m_lsearch0_logger;          ///<
        string_t                    m_lsearchk_id;              ///<
        rlsearchk_t                 m_lsearchk;                 ///<
        lsearchk_t::logger_t        m_lsearchk_logger;          ///<
    };
}
