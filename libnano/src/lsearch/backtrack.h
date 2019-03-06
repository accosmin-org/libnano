#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief backtracking line-search that stops when the Armijo condition is satisfied,
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
    ///
    class lsearch_backtrack_t final : public lsearch_strategy_t
    {
    public:

        lsearch_backtrack_t() = default;
        bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;

        void to_json(json_t&) const final;
        void from_json(const json_t&) final;

    private:

        // attributes
        scalar_t    m_ro{static_cast<scalar_t>(0.5)};   ///<
    };
}
