#pragma once

#include <nano/lsearch0.h>
#include <nano/lsearchk.h>

namespace nano
{
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
        lsearch_t(rlsearch0_t&& lsearch0, rlsearchk_t&& lsearchk)
            : m_lsearch0(std::move(lsearch0))
            , m_lsearchk(std::move(lsearchk))
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
        rlsearch0_t m_lsearch0; ///< procedure to guess the initial step length
        rlsearchk_t m_lsearchk; ///< procedure to adjust the step length
    };
} // namespace nano
