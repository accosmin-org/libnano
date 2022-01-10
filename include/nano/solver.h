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
    class NANO_PUBLIC solver_t
    {
    public:

        ///
        /// logging operator: op(state), returns false if the optimization should stop
        ///
        using logger_t = std::function<bool(const solver_state_t&)>;

        ///
        /// \brief returns the available implementations
        ///
        static solver_factory_t& all();

        ///
        /// \brief constructor
        ///
        solver_t();


        ///
        /// \brief enable moving
        ///
        solver_t(solver_t&&) noexcept = default;
        solver_t& operator=(solver_t&&) noexcept = default;

        ///
        /// \brief disable copying
        ///
        solver_t(const solver_t&) = delete;
        solver_t& operator=(const solver_t&) = delete;

        ///
        /// \brief destructor
        ///
        virtual ~solver_t() = default;

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
        /// \brief change the desired accuracy:
        ///     - ~gradient magnitude for smooth functions, see state_t::converged, or
        ///     - ~function and point decrease for non-smooth functions
        ///
        void epsilon(scalar_t epsilon);

        ///
        /// \brief change the maximum number of function evaluations.
        ///
        void max_evals(int max_evals);

        ///
        /// \brief change the desired function value and gradient tolerance,
        ///     aka the c1 and c2 parameters in the (strong) Wolfe conditions
        ///
        /// NB: the recommended values depend very much on the optimization algorithm and
        ///     provide a good balance between gradient updates and accuracy of the line-search step length.
        ///
        void tolerance(scalar_t c1, scalar_t c2);

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
        auto c1() const { return m_lsearchk->c1(); }
        auto c2() const { return m_lsearchk->c2(); }
        auto epsilon() const { return m_epsilon.get(); }
        auto max_evals() const { return m_max_evals.get(); }
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

        ///
        /// \brief check convergence for non-monotonic solvers.
        ///
        bool converged(const vector_t& xk, scalar_t fxk, const vector_t& xk1, scalar_t fxk1) const;

    private:

        // attributes
        sparam1_t       m_epsilon{"solver::epsilon", 0, LT, 1e-6, LE, 1e-3};    ///< desired accuracy
        iparam1_t       m_max_evals{"solver::maxevals", 10, LE, 1000, LT, 1e+6};///< maximum number of function evaluations
        logger_t        m_logger;                   ///<
        string_t        m_lsearch0_id;              ///<
        rlsearch0_t     m_lsearch0;                 ///<
        string_t        m_lsearchk_id;              ///<
        rlsearchk_t     m_lsearchk;                 ///<
        bool            m_monotonic{true};          ///<
    };
}
