#pragma once

#include <nano/solver/lsearch.h>

namespace nano
{
    class solver_t;
    using rsolver_t = std::unique_ptr<solver_t>;

    ///
    /// \brief unconstrained numerical optimization algorithm that uses line-search
    ///     along a descent direction to iteratively minimize a smooth lower-bounded function.
    ///
    /// NB: the resulting point (if enough iterations have been used) is either:
    ///     - the global minimum if the function is convex or
    ///     - a critical point (not necessarily a local minimum) otherwise.
    ///
    class NANO_PUBLIC solver_t : public estimator_t, public clonable_t<solver_t>
    {
    public:
        ///
        /// \brief logging operator: op(state), returns false if the optimization should stop
        ///
        using logger_t = std::function<bool(const solver_state_t&)>;

        ///
        /// \brief constructor
        ///
        explicit solver_t(string_t id);

        ///
        /// \brief enable copying
        ///
        solver_t(const solver_t&);
        solver_t& operator=(const solver_t&) = delete;

        ///
        /// \brief enable moving
        ///
        solver_t(solver_t&&) noexcept   = default;
        solver_t& operator=(solver_t&&) = default;

        ///
        /// \brief destructor
        ///
        ~solver_t() override = default;

        ///
        /// \brief returns the available implementations.
        ///
        static factory_t<solver_t>& all();

        ///
        /// \brief minimize the given function starting from the initial point x0 until:
        ///     - convergence is achieved (critical point, possiblly a local/global minima) or
        ///     - the maximum number of iterations is reached or
        ///     - the user canceled the optimization (using the logging function) or
        ///     - the solver failed (e.g. line-search failed)
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const;

        ///
        /// \brief set the logging callbacks.
        ///
        void logger(const logger_t& logger);
        void lsearch0_logger(const lsearch0_t::logger_t& logger);
        void lsearchk_logger(const lsearchk_t::logger_t& logger);

        ///
        /// \brief set the line-search initialization method.
        ///
        void lsearch0(const lsearch0_t&);
        void lsearch0(const string_t& id);

        ///
        /// \brief set the line-search strategy method.
        ///
        void lsearchk(const lsearchk_t&);
        void lsearchk(const string_t& id);

        ///
        /// \brief return true if the solver is monotonic, which guarantees that at each iteration:
        ///     - either the function value decreases
        ///     - or the optimization stops (e.g. convergence reached, line-search failed, user requested termination).
        ///
        /// NB: the solver usually doesn't use line-search if not monotonic.
        ///
        bool monotonic() const;

        ///
        /// \brief return the line-search initialization method.
        ///
        const auto& lsearch0() const { return *m_lsearch0; }

        ///
        /// \brief return the the line-search strategy method.
        ///
        const auto& lsearchk() const { return *m_lsearchk; }

    protected:
        void      monotonic(bool);
        lsearch_t make_lsearch() const;
        bool      log(solver_state_t&) const;
        bool      done(const function_t&, solver_state_t&, bool iter_ok, bool converged) const;

        virtual solver_state_t do_minimize(const function_t&, const vector_t& x0) const = 0;

    private:
        // attributes
        logger_t    m_logger;          ///<
        rlsearch0_t m_lsearch0;        ///<
        rlsearchk_t m_lsearchk;        ///<
        bool        m_monotonic{true}; ///<
    };
} // namespace nano
