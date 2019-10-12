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
        lsearch0_linear_t() = default;

        ///
        /// \brief @see lsearch0_t
        ///
        rlsearch0_t clone() const final;

        ///
        /// \brief @see lsearch0_t
        ///
        scalar_t get(const solver_state_t&) final;

        ///
        /// \brief change parameters
        ///
        void alpha(const scalar_t alpha) { m_alpha = alpha; }
        void beta(const scalar_t beta) { m_beta = beta; }

        ///
        /// \brief access functions
        ///
        auto alpha() const { return m_alpha.get(); }
        auto beta() const { return m_beta.get(); }

    private:

        // attributes
        sparam1_t   m_alpha{"lsearch0::linear::alpha", 1, LT, 1.01, LT, 1e+6};  ///< see (2)
        sparam1_t   m_beta{"lsearch0::linear::beta", 1, LT, 10.0, LT, 1e+6};    ///< see (2)
        scalar_t    m_prevdg{1};                                                ///< previous direction dot product: dg_{k-1}
    };
}
