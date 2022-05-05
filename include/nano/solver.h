#pragma once

#include <nano/solver/lsearch.h>
#include <nano/solver/function.h>

namespace nano
{
    class solver_t;
    using solver_factory_t = factory_t<solver_t>;
    using rsolver_t = solver_factory_t::trobject;

    ///
    /// \brief unconstrained numerical optimization algorithm that uses line-search
    ///     along a descent direction to iteratively minimize a smooth lower-bounded function.
    ///
    /// NB: the resulting point (if enough iterations have been used) is either:
    ///     - the global minimum if the function is convex or
    ///     - a critical point (not necessarily a local minimum) otherwise.
    ///
    class NANO_PUBLIC solver_t : public estimator_t
    {
    public:

        ///
        /// \brief logging operator: op(state), returns false if the optimization should stop
        ///
        using logger_t = std::function<bool(const solver_state_t&)>;

        ///
        /// \brief constructor
        ///
        solver_t();

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
        virtual solver_state_t minimize(const function_t&, const vector_t& x0) const = 0;

        ///
        /// \brief set the logging callback
        ///
        void logger(const logger_t& logger);
        void lsearch0_logger(const lsearch0_t::logger_t& logger);
        void lsearchk_logger(const lsearchk_t::logger_t& logger);

        ///
        /// \brief set the line-search initialization
        ///
        void lsearch0(const string_t& id);
        void lsearch0(const string_t& id, rlsearch0_t&&);

        ///
        /// \brief set the line-search strategy
        ///
        void lsearchk(const string_t& id);
        void lsearchk(const string_t& id, rlsearchk_t&&);

        ///
        /// \brief return true if the solver is monotonic, which guarantees that at each iteration:
        ///     - either the function value decreases
        ///     - or the optimization stops (e.g. convergence reached, line-search failed, user requested termination).
        ///
        /// NB: the solver ignores the line-search attributes if not monotonic.
        ///
        bool monotonic() const;

        ///
        /// \brief access functions
        ///
        const auto& lsearch0_id() const { return m_lsearch0_id; }
        const auto& lsearchk_id() const { return m_lsearchk_id; }

    protected:

        ///
        /// \brief create a copy of the line-search utility.
        ///
        lsearch_t make_lsearch() const;

        ///
        /// \brief create a solver function to keep track of various statistics.
        ///
        solver_function_t make_function(const function_t&, const vector_t& x0) const;

        ///
        /// \brief log the current optimization state (if the logger is provided).
        ///
        bool log(solver_state_t& state) const;

        ///
        /// \brief check if the optimization is done (convergence or error) after an iteration.
        ///
        bool done(const solver_function_t& function, solver_state_t& state, bool iter_ok, bool converged) const;

        ///
        /// \brief sets the monotonicity.
        ///
        void monotonic(bool);

    private:

        // attributes
        logger_t        m_logger;                   ///<
        string_t        m_lsearch0_id;              ///<
        rlsearch0_t     m_lsearch0;                 ///<
        string_t        m_lsearchk_id;              ///<
        rlsearchk_t     m_lsearchk;                 ///<
        bool            m_monotonic{true};          ///<
    };
}
