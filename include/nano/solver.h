#pragma once

#include <nano/factory.h>
#include <nano/parameter.h>
#include <nano/solver/state.h>
#include <nano/solver/function.h>

namespace nano
{
    class solver_t;
    using solver_factory_t = factory_t<solver_t>;
    using rsolver_t = solver_factory_t::trobject;

    ///
    /// \brief unconstrained numerical optimization algorithm
    ///     to iteratively minimize a smooth lower-bounded function.
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
        /// \brief constructor
        ///
        solver_t() = default;

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

        ///
        /// \brief change the desired accuracy (~ gradient magnitude, see state_t::converged)
        ///
        void epsilon(scalar_t epsilon);

        ///
        /// \brief change the maximum number of iterations
        ///
        void max_iterations(int max_iterations);

        ///
        /// \brief access functions
        ///
        auto epsilon() const { return m_epsilon.get(); }
        auto max_iterations() const { return m_max_iterations.get(); }

    protected:

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
    };
}
