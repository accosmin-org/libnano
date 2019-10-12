#pragma once

#include <nano/lsearch0.h>

namespace nano
{
    ///
    /// \brief constant step length (useful for LBFGS, Quasi-Newton and Newton methods).
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
    ///
    class NANO_PUBLIC lsearch0_constant_t final : public lsearch0_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        lsearch0_constant_t() = default;

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
        void t0(const scalar_t t0) { m_t0 = t0; }

        ///
        /// \brief access functions
        ///
        auto t0() const { return m_t0.get(); }

    private:

        // attributes
        sparam1_t   m_t0{"lsearch0::constant::t0", 0, LT, 1, LT, 1e+6};  ///< see (1)
    };
}
