#pragma once

#include <nano/lsearch/lsearch0.h>

namespace nano
{
    ///
    /// \brief constant step length (useful for LBFGS, Quasi-Newton and Newton methods).
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
    ///
    class lsearch0_constant_t final : public lsearch0_t
    {
    public:

        lsearch0_constant_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        scalar_t get(const solver_state_t&) final;

    private:

        // attributes
        scalar_t    m_t0{1};    ///< the constant line-search step length to return
    };
}
