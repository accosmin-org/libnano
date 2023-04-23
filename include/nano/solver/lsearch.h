#pragma once

#include <nano/lsearch0.h>
#include <nano/lsearchk.h>

namespace nano
{
///
/// \brief line-search procedure using two steps:
///     - estimate the initial step size and
///     - adjust the step size to satisfy the associated conditions (e.g. Armijo-Goldstein or Wolfe).
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
    /// \brief compute the step size along the given descent direction.
    ///
    bool get(solver_state_t& state, const vector_t& descent) const
    {
        assert(m_lsearch0);
        assert(m_lsearchk);

        const auto init_step_size  = m_lsearch0->get(state, descent, m_last_step_size);
        const auto [ok, step_size] = m_lsearchk->get(state, descent, init_step_size);
        m_last_step_size           = step_size;
        return ok;
    }

private:
    // attributes
    rlsearch0_t      m_lsearch0;             ///< procedure to guess the initial step size
    rlsearchk_t      m_lsearchk;             ///< procedure to adjust the step size
    mutable scalar_t m_last_step_size{-1.0}; ///< step size of the previous iteration
};
} // namespace nano
