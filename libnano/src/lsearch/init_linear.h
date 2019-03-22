#pragma once

#include <nano/lsearch/init.h>

namespace nano
{
    ///
    /// \brief use linear interpolation of the previous line-search step lengths.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
    ///
    class lsearch_linear_init_t final : public lsearch_init_t
    {
    public:

        lsearch_linear_init_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        scalar_t get(const solver_state_t&) final;

    private:

        // attributes
        scalar_t    m_t0{1};        ///< line-search step length to return in the first iteration
        scalar_t    m_tro{0.25};    ///< minimum line-search step length ratio to the previous iteration
        scalar_t    m_tmax{1};      ///< maximum line-search step length
        scalar_t    m_prevdg{1};    ///< previous direction dot product
    };
}
