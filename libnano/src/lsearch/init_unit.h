#pragma once

#include <nano/lsearch/init.h>

namespace nano
{
    ///
    /// \brief unit step length (useful for LBFGS, Quasi-Newton and Newton methods).
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
    ///
    class lsearch_unit_init_t final : public lsearch_init_t
    {
    public:

        lsearch_unit_init_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        scalar_t get(const solver_state_t&, const int iteration) final;
    };
}
