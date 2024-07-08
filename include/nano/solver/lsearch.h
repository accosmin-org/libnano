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
    lsearch_t(rlsearch0_t&& lsearch0, rlsearchk_t&& lsearchk);

    ///
    /// \brief compute the step size along the given descent direction.
    ///
    bool get(solver_state_t&, const vector_t& descent, const logger_t&) const;

private:
    // attributes
    rlsearch0_t      m_lsearch0;             ///< procedure to guess the initial step size
    rlsearchk_t      m_lsearchk;             ///< procedure to adjust the step size
    mutable scalar_t m_last_step_size{-1.0}; ///< step size of the previous iteration
};
} // namespace nano
