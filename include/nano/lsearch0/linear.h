#pragma once

#include <nano/lsearch0.h>

namespace nano
{
    ///
    /// \brief use linear interpolation of the previous line-search step lengths.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
    ///     see "Practical methods of optimization", Fletcher, p.38-39
    ///
    class NANO_PUBLIC lsearch0_linear_t final : public lsearch0_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        lsearch0_linear_t();

        ///
        /// \brief @see lsearch0_t
        ///
        rlsearch0_t clone() const final;

        ///
        /// \brief @see lsearch0_t
        ///
        scalar_t get(const solver_state_t&) final;

    private:

        // attributes
        scalar_t    m_prevdg{1};            ///< previous direction dot product: dg_{k-1}
    };
}
