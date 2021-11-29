#pragma once

#include <nano/lsearch0.h>
#include <nano/lsearchk.h>
#include <nano/solver/function.h>

namespace nano
{
    class solver_t;
    using solver_factory_t = factory_t<solver_t>;
    using rsolver_t = solver_factory_t::trobject;

    ///
    /// \brief line-search procedure using two steps:
    ///     - estimate the initial step length and
    ///     - adjust the step length to satisfy the associated conditions (e.g. Armijo-Goldstein or Wolfe).
    ///
    class lsearch_t
    {
    public:

        ///
        /// \brief constructor
        ///
        lsearch_t(rlsearch0_t&& lsearch0, rlsearchk_t&& lsearchk) :
            m_lsearch0(std::move(lsearch0)),
            m_lsearchk(std::move(lsearchk))
        {
        }

        ///
        /// \brief compute the step length
        ///
        bool get(solver_state_t& state) const
        {
            assert(m_lsearch0);
            assert(m_lsearchk);

            const auto t0 = m_lsearch0->get(state);
            return m_lsearchk->get(state, t0);
        }

    private:

        // attributes
        rlsearch0_t         m_lsearch0;     ///< procedure to guess the initial step length
        rlsearchk_t         m_lsearchk;     ///< procedure to adjust the step length
    };

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
        explicit solver_t(
            scalar_t c1 = 1e-1,
            scalar_t c2 = 9e-1,
            const string_t& lsearch0_id = "quadratic",
            const string_t& lsearchk_id = "morethuente");


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
        virtual solver_state_t minimize(const function_t&, const vector_t& x0) const;

        ///
        /// \brief set the logging callback
        ///
        void logger(const logger_t& logger);
        void lsearch0_logger(const lsearch0_t::logger_t& logger);
        void lsearchk_logger(const lsearchk_t::logger_t& logger);

        ///
        /// \brief change the desired accuracy (~ gradient magnitude, see state_t::converged)
        ///
        void epsilon(scalar_t epsilon);

        ///
        /// \brief change the maximum number of iterations
        ///
        void max_iterations(int max_iterations);

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
        /// \brief access functions
        ///
        auto c1() const { return m_lsearchk->c1(); }
        auto c2() const { return m_lsearchk->c2(); }
        auto epsilon() const { return m_epsilon.get(); }
        auto max_iterations() const { return m_max_iterations.get(); }
        const auto& lsearch0_id() const { return m_lsearch0_id; }
        const auto& lsearchk_id() const { return m_lsearchk_id; }

    protected:

        ///
        /// \brief minimize the given function starting from the initial point x0
        ///     and using the given line-search strategy.
        ///
        virtual solver_state_t iterate(const solver_function_t&, const lsearch_t&, const vector_t& x0) const = 0;

        ///
        /// \brief log the current optimization state (if the logger is provided)
        ///
        bool log(solver_state_t& state) const;

        ///
        /// \brief check if the optimization is done (convergence or error) after an iteration
        ///
        bool done(const solver_function_t& function, solver_state_t& state, bool iter_ok) const;

    private:

        // attributes
        sparam1_t       m_epsilon{"solver::epsilon", 0, LT, 1e-6, LE, 1e-3};        ///< desired accuracy
        iparam1_t       m_max_iterations{"solver::maxiters", 1, LE, 1000, LT, 1e+6};///< maximum number of iterations
        logger_t        m_logger;                   ///<
        string_t        m_lsearch0_id;              ///<
        rlsearch0_t     m_lsearch0;                 ///<
        string_t        m_lsearchk_id;              ///<
        rlsearchk_t     m_lsearchk;                 ///<
    };
}
