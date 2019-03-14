#pragma once

#include <nano/lsearch.h>

namespace nano
{
    class solver_t;
    using solver_factory_t = factory_t<solver_t>;
    using rsolver_t = solver_factory_t::trobject;

    ///
    /// \brief returns the registered solvers.
    ///
    NANO_PUBLIC solver_factory_t& get_solvers();

    ///
    /// \brief wrapper over function_t to keep track of the number of function value and gradient calls.
    ///
    class solver_function_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit solver_function_t(const function_t& function) :
            function_t(function),
            m_function(function)
        {
        }

        ///
        /// \brief compute function value (and gradient if provided)
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
        {
            m_fcalls += 1;
            m_gcalls += gx ? 1 : 0;
            return m_function.vgrad(x, gx);
        }

        ///
        /// \brief number of function evaluation calls
        ///
        size_t fcalls() const { return m_fcalls; }

        ///
        /// \brief number of function gradient calls
        ///
        size_t gcalls() const { return m_gcalls; }

    private:

        // attributes
        const function_t&   m_function;         ///<
        mutable size_t      m_fcalls{0};        ///< #function value evaluations
        mutable size_t      m_gcalls{0};        ///< #function gradient evaluations
    };

    ///
    /// \brief generic (batch) optimization algorithm typically using an adaptive line-search method.
    ///
    class NANO_PUBLIC solver_t : public json_configurable_t
    {
    public:

        ///
        /// logging operator: op(state), returns false if the optimization should stop
        ///
        using logger_t = std::function<bool(const solver_state_t&)>;

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
        /// \brief change parameters
        ///
        auto& logger(const logger_t& logger) { m_logger = logger; return *this; }
        auto& epsilon(const scalar_t epsilon) { m_epsilon = epsilon; return *this; }
        auto& max_iterations(const int max_iterations) { m_max_iterations = max_iterations; return *this; }

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
        /// \brief check if the optimization is done (convergence or error) after an iteration
        ///
        bool done(const solver_function_t& function, solver_state_t& state, const bool iter_ok) const
        {
            state.m_fcalls = function.fcalls();
            state.m_gcalls = function.gcalls();

            const auto step_ok = iter_ok && state;
            const auto converged = state.converged(epsilon());

            if (converged || !step_ok)
            {
                // either converged or failed
                state.m_status = converged ?
                    solver_state_t::status::converged :
                    solver_state_t::status::failed;
                log(state);
                return true;
            }
            else if (!log(state))
            {
                // stopping was requested
                state.m_status = solver_state_t::status::stopped;
                return true;
            }

            // OK, go on with the optimization
            return false;
        }

    private:

        // attributes
        scalar_t    m_epsilon{1e-6};        ///< required precision (uper bounds the magnitude of the gradient)
        int         m_max_iterations{1000}; ///< maximum number of iterations
        logger_t    m_logger;               ///<
    };
}
