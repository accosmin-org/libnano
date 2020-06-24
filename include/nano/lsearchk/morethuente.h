#pragma once

#include <nano/lsearchk.h>

namespace nano
{
    ///
    /// \brief More & Thunte line-search.
    ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease",
    ///     by Jorge J. More and David J. Thuente
    ///
    /// NB: this implementation ports the 'dcsrch' and the 'dcstep' Fortran routines from MINPACK-2.
    ///     see http://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/
    ///
    class NANO_PUBLIC lsearchk_morethuente_t final : public lsearchk_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        lsearchk_morethuente_t() = default;

        ///
        /// \brief @see lsearchk_t
        ///
        [[nodiscard]] rlsearchk_t clone() const final;

        ///
        /// \brief @see lsearchk_t
        ///
        bool get(const solver_state_t& state0, solver_state_t& state) final;

        ///
        /// \brief change parameters
        ///
        void delta(const scalar_t delta) { m_delta = delta; }

        ///
        /// \brief access parameters
        ///
        [[nodiscard]] auto delta() const { return m_delta.get(); }

    private:

        ///
        /// \brief see dcstep routine in MINPACK-2 (see http://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/)
        ///
        void dcstep(
            scalar_t& stx, scalar_t& fx, scalar_t& dx,
            scalar_t& sty, scalar_t& fy, scalar_t& dy,
            scalar_t& stp, const scalar_t& fp, const scalar_t& dp,
            bool& brackt, scalar_t stpmin, scalar_t stpmax) const;

        // attributes
        sparam1_t   m_delta{"lsearchk::morethuente", 0, LT, 0.66, LT, 1};   ///< see (1)
    };
}
