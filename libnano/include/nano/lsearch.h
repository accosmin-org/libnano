#pragma once

#include <nano/lsearch/init.h>
#include <nano/lsearch/strategy.h>

namespace nano
{
    ///
    /// \brief compute the step length of the line search procedure.
    ///
    class lsearch_t
    {
    public:

        ///
        /// \brief constructor
        ///
        lsearch_t(rlsearch_init_t&& init, rlsearch_strategy_t&& strategy) :
            m_init(std::move(init)),
            m_strategy(std::move(strategy))
        {
        }

        ///
        /// \brief compute the step length
        ///
        bool get(solver_state_t& state) const
        {
            assert(m_init);
            assert(m_strategy);

            const auto t0 = m_init->get(state);
            return m_strategy->get(state, t0);
        }

    private:

        // attributes
        rlsearch_init_t         m_init;                 ///<
        rlsearch_strategy_t     m_strategy;             ///<
    };
}
