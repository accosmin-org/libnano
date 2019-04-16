#pragma once

#include <nano/lsearch/lsearch0.h>

namespace nano
{
    ///
    /// \brief use linear interpolation of the previous line-search step lengths.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
    ///     see "Practical methods of optimization", Fletcher, p.38-39
    ///
    class lsearch0_linear_t final : public lsearch0_t
    {
    public:

        lsearch0_linear_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        scalar_t get(const solver_state_t&) final;

    private:

        // attributes
        scalar_t    m_alpha{1.01};  ///< correction to truncate to 1
        scalar_t    m_beta{10.0};   ///< factor relative to epsilon to safeguard the interpolated step length
        scalar_t    m_prevdg{1};    ///< previous direction dot product: dg_{k-1}
    };
}
