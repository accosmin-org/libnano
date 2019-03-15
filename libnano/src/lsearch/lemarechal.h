#pragma once

#include <nano/lsearch/strategy.h>

namespace nano
{
    ///
    /// \brief the line-search algorithm described here:
    ///     see "A view of line-searches", C. Lemarechal
    ///
    class lsearch_lemarechal_t final : public lsearch_strategy_t
    {
    public:

        lsearch_lemarechal_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;

    private:

        // attributes
        scalar_t    m_ro{3};     ///< extrapolation factor
    };
}
