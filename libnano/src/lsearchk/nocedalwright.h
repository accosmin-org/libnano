#pragma once

#include <nano/lsearch/lsearchk.h>

namespace nano
{
    ///
    /// \brief the More&Thuente-like line-search algorithm described here:
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
    ///
    /// todo: this algorithm may have been first described in Fletcher's "Practical methods of optimization"
    ///     rename the class accordingly if this is the case
    ///
    class lsearchk_nocedalwright_t final : public lsearchk_t
    {
    public:

        lsearchk_nocedalwright_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        bool get(const solver_state_t& state0, solver_state_t& state) final;

    private:

        bool zoom(const solver_state_t&, lsearch_step_t lo, lsearch_step_t hi, solver_state_t&) const;

        // attributes
        interpolation   m_interpolation{interpolation::cubic};  ///<
        scalar_t        m_ro{3};                                ///< extrapolation factor
    };
}
