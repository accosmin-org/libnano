#pragma once

#include <nano/solver.h>
#include <nano/lsearch0.h>
#include <nano/lsearchk.h>

namespace nano
{
    class lsearch_solver_t;
    using lsearch_solver_factory_t = factory_t<lsearch_solver_t>;
    using rlsearch_solver_t = lsearch_solver_factory_t::trobject;

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
    class NANO_PUBLIC lsearch_solver_t : public solver_t
    {
    public:

        ///
        /// \brief constructor
        ///
        explicit lsearch_solver_t(
            scalar_t c1 = 1e-1,
            scalar_t c2 = 9e-1,
            const string_t& lsearch0 = "quadratic",
            const string_t& lsearchk = "morethuente");

        ///
        /// \brief returns the available implementations
        ///
        static lsearch_solver_factory_t& all();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const override;

        ///
        /// \brief set the logging callbacks
        ///
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
        /// \brief change the desired function value and gradient tolerance,
        ///     aka the c1 and c2 parameters in the (strong) Wolfe conditions
        ///
        /// NB: the recommended values depend very much of the optimization algorithm and
        ///     provide a good balance between gradient updates and accuracy of the line-search step length.
        ///
        void tolerance(scalar_t c1, scalar_t c2);

        ///
        /// \brief access functions
        ///
        auto c1() const { return m_lsearchk->c1(); }
        auto c2() const { return m_lsearchk->c2(); }
        const auto& lsearch0_id() const { return m_lsearch0_id; }
        const auto& lsearchk_id() const { return m_lsearchk_id; }

    protected:

        ///
        /// \brief minimize the given function starting from the initial point x0
        ///     and using the given line-search strategy
        ///
        virtual solver_state_t iterate(const solver_function_t&, const lsearch_t&, const vector_t& x0) const = 0;

    private:

        // attributes
        string_t        m_lsearch0_id;              ///<
        rlsearch0_t     m_lsearch0;                 ///<
        string_t        m_lsearchk_id;              ///<
        rlsearchk_t     m_lsearchk;                 ///<
    };
}
