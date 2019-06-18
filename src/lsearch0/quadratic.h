#pragma once

#include <nano/lsearch/lsearch0.h>

namespace nano
{
    ///
    /// \brief use quadratic interpolation of the previous line-search step lengths.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
    ///     see "Practical methods of optimization", Fletcher, p.38-39
    ///
    class lsearch0_quadratic_t final : public lsearch0_t
    {
    public:

        lsearch0_quadratic_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        scalar_t get(const solver_state_t&) final;

    private:

        // attributes
        scalar_t    m_alpha{1.01};  ///< correction to truncate to 1
        scalar_t    m_beta{10.0};   ///< factor relative to epsilon to safeguard the interpolated step length
        scalar_t    m_prevf{0};     ///< previous function value
        scalar_t    m_prevdg{1};    ///< previous direction dot product
    };
}
