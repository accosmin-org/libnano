#pragma once

#include <nano/core/estimator.h>
#include <nano/core/factory.h>
#include <nano/solver/lstep.h>

namespace nano
{
    class lsearch0_t;
    using rlsearch0_t = std::unique_ptr<lsearch0_t>;

    ///
    /// \brief estimate the initial step length of the line-search procedure.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
    ///     see "Practical methods of optimization", Fletcher, chapter 2
    ///
    class NANO_PUBLIC lsearch0_t : public estimator_t, public clonable_t<lsearch0_t>
    {
    public:
        ///
        /// logging operator: op(solver_state, proposed_line_search_step_length_length).
        ///
        using logger_t = std::function<void(const solver_state_t&, const scalar_t)>;

        ///
        /// \brief constructor
        ///
        explicit lsearch0_t(string_t id);

        ///
        /// \brief returns the available implementations
        ///
        static factory_t<lsearch0_t>& all();

        ///
        /// \brief returns the initial step length given the current state
        /// NB: may keep track of the previous states
        ///
        virtual scalar_t get(const solver_state_t& state) = 0;

        ///
        /// \brief set the logging operator.
        ///
        void logger(const logger_t& logger) { m_logger = logger; }

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
        logger_t m_logger; ///<
    };
} // namespace nano
