#pragma once

#include <nano/lsearch/strategy.h>

namespace nano
{
    ///
    /// \brief the More&Thuente-like line-search algorithm described here:
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
    ///
    class lsearch_nocedalwright_t final : public lsearch_strategy_t
    {
    public:

        lsearch_nocedalwright_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;

    private:

        bool zoom(const solver_state_t&, lsearch_step_t lo, lsearch_step_t hi, solver_state_t&) const;

        // attributes
        scalar_t    m_ro{3};    ///< extrapolation factor
    };
}
