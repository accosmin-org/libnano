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
    /// \brief constructor
    ///
    lsearchk_morethuente_t();

    ///
    /// \brief @see lsearchk_t
    ///
    rlsearchk_t clone() const override;

    ///
    /// \brief @see lsearchk_t
    ///
    result_t do_get(const solver_state_t&, const vector_t&, scalar_t, solver_state_t&) const override;
};
} // namespace nano
