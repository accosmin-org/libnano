#pragma once

#include <nano/lsearch/lsearch0.h>

namespace nano
{
    ///
    /// \brief use linear interpolation of the previous line-search step lengths.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
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
        scalar_t    m_tro{1.01};    ///<
        scalar_t    m_prevdg{1};    ///< previous direction dot product
    };
}
