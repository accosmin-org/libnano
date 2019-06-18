#pragma once

#include <nano/lsearch/lsearchk.h>

namespace nano
{
    ///
    /// \brief backtracking line-search that stops when the Armijo condition is satisfied,
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
    ///
    class lsearchk_backtrack_t final : public lsearchk_t
    {
    public:

        lsearchk_backtrack_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        bool get(const solver_state_t& state0, solver_state_t& state) final;

    private:

        // attributes
        interpolation   m_interpolation{interpolation::cubic};  ///<
    };
}
