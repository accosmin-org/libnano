#pragma once

#include <nano/arch.h>
#include <nano/solver/lstep.h>
#include <nano/core/factory.h>
#include <nano/core/parameter.h>

namespace nano
{
    class lsearch0_t;
    using lsearch0_factory_t = factory_t<lsearch0_t>;
    using rlsearch0_t = lsearch0_factory_t::trobject;

    ///
    /// \brief estimate the initial step length of the line-search procedure.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
    ///     see "Practical methods of optimization", Fletcher, chapter 2
    ///
    class NANO_PUBLIC lsearch0_t
    {
    public:

        ///
        /// logging operator: op(solver_state, proposed_line_search_step_length_length).
        ///
        using logger_t = std::function<void(const solver_state_t&, const scalar_t)>;

        ///
        /// \brief constructor
        ///
        lsearch0_t() = default; // LCOV_EXCL_LINE

        ///
        /// \brief enable copying
        ///
        lsearch0_t(const lsearch0_t&) = default; // LCOV_EXCL_LINE
        lsearch0_t& operator=(const lsearch0_t&) = default;

        ///
        /// \brief enable moving
        ///
        lsearch0_t(lsearch0_t&&) noexcept = default;
        lsearch0_t& operator=(lsearch0_t&&) noexcept = default;

        ///
        /// \brief destructor
        ///
        virtual ~lsearch0_t() = default;

        ///
        /// \brief returns the available implementations
        ///
        static lsearch0_factory_t& all();

        ///
        /// \brief clone the object, by keeping the parameters but with an internal "clean state"
        ///
        virtual rlsearch0_t clone() const = 0;

        ///
        /// \brief returns the initial step length given the current state
        /// NB: may keep track of the previous states
        ///
        virtual scalar_t get(const solver_state_t& state) = 0;

        ///
        /// \brief change parameters
        ///
        void logger(const logger_t& logger) { m_logger = logger; }
        void epsilon(scalar_t epsilon) { m_epsilon.set(epsilon); }

        ///
        /// \brief access functions
        ///
        auto epsilon() const { return m_epsilon.get(); }

    protected:

        ///
        /// \brief log the current line-search trial length (if the logger is provided)
        ///
        void log(const solver_state_t& state, const scalar_t t) const
        {
            if (m_logger)
            {
                m_logger(state, t);
            }
        }

    private:

        // attributes
        logger_t    m_logger;                                           ///<
        sparam1_t   m_epsilon{"lsearch0::epsilon", 0, LT, 1e-6, LT, 1}; ///< tolerance of the convergence criterion
    };
}
