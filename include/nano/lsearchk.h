#pragma once

#include <nano/core/estimator.h>
#include <nano/core/factory.h>
#include <nano/solver/lstep.h>

namespace nano
{
    class lsearchk_t;
    using lsearchk_factory_t = factory_t<lsearchk_t>;
    using rlsearchk_t        = lsearchk_factory_t::trobject;

    ///
    /// \brief the objective type of the line-search procedure.
    ///
    enum class lsearch_type
    {
        none,               ///<
        armijo,             ///< sufficient decrease of the function value (Armijo)
        wolfe,              ///< armijo + decrease of the slope (Wolfe)
        strong_wolfe,       ///< armijo + small slow (Wolfe)
        wolfe_approx_wolfe, ///< armijo + wolfe or approximated armijo + wolfe (see CG_DESCENT)
    };

    ///
    /// \brief compute the step size along the given descent direction starting from the initial guess `t0`.
    ///
    /// NB: the returned step size is positive and guaranteed to decrease the function value (if no failure).
    ///
    class NANO_PUBLIC lsearchk_t : public estimator_t
    {
    public:
        ///
        /// logging operator: op(solver_state_at_0, solver_state_at_t), called for each trial of the line-search length.
        ///
        using logger_t = std::function<void(const solver_state_t&, const solver_state_t&)>;

        ///
        /// \brief constructor
        ///
        lsearchk_t();

        ///
        /// \brief returns the available implementations.
        ///
        static lsearchk_factory_t& all();

        ///
        /// \brief clone the object, by keeping the parameters but with an internal "clean state".
        ///
        virtual rlsearchk_t clone() const = 0;

        ///
        /// \brief compute the step size starting from the given state and the initial estimate of the step size.
        ///
        virtual bool get(solver_state_t& state, scalar_t t) const;

        ///
        /// \brief set the logging operator.
        ///
        void logger(const logger_t& logger);

        ///
        /// \brief minimum allowed line-search step.
        ///
        static scalar_t stpmin();

        ///
        /// \brief maximum allowed line-search step.
        ///
        static scalar_t stpmax();

        ///
        /// \brief returns the objective optimized by the line-search implementation.
        ///
        lsearch_type type() const;

    protected:
        void type(lsearch_type);
        void log(const solver_state_t& state0, const solver_state_t& state) const;

        virtual bool get(const solver_state_t& state0, solver_state_t&) const = 0;

    private:
        // attributes
        logger_t     m_logger;                   ///<
        lsearch_type m_type{lsearch_type::none}; ///<
    };
} // namespace nano
