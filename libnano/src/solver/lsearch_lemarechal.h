#pragma once

#include <nano/solver.h>

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
        bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;
    };
}
