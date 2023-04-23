#pragma once

#include <nano/lsearchk.h>

namespace nano
{
///
/// \brief backtracking line-search that stops when the Armijo condition is satisfied,
///     see "Numerical optimization", Nocedal & Wright, 2nd edition
///
class NANO_PUBLIC lsearchk_backtrack_t final : public lsearchk_t
{
public:
    ///
    /// \brief constructor
    ///
    lsearchk_backtrack_t();

    ///
    /// \brief @see lsearchk_t
    ///
    rlsearchk_t clone() const override;

    ///
    /// \brief @see lsearchk_t
    ///
    result_t do_get(const solver_state_t&, const vector_t&, scalar_t, solver_state_t&) const override;
};
} // namespace nano
