#pragma once

#include <nano/lsearch0.h>

namespace nano
{
    ///
    /// \brief CG_DESCENT initial step length strategy.
    ///
    class NANO_PUBLIC lsearch0_cgdescent_t final : public lsearch0_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        lsearch0_cgdescent_t() = default;

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
        void phi0(const scalar_t phi0) { m_phi0.set(phi0); }
        void phi1(const scalar_t phi1) { m_phi1.set(phi1); }
        void phi2(const scalar_t phi2) { m_phi2.set(phi2); }

        ///
        /// \brief access functions
        ///
        auto phi0() const { return m_phi0.get(); }
        auto phi1() const { return m_phi1.get(); }
        auto phi2() const { return m_phi2.get(); }

    private:

        // attributes
        sparam1_t   m_phi0{"lsearch0::cgdescent::phi0", 0, LT, 0.01, LT, 1};    ///<
        sparam1_t   m_phi1{"lsearch0::cgdescent::phi1", 0, LT, 0.10, LT, 1};    ///<
        sparam1_t   m_phi2{"lsearch0::cgdescent::phi2", 1, LT, 2.00, LT, 1e+6}; ///<
    };
}
