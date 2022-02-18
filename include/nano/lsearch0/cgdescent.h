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
        /// \brief constructor
        ///
        lsearch0_cgdescent_t();

        ///
        /// \brief @see lsearch0_t
        ///
        rlsearch0_t clone() const final;

        ///
        /// \brief @see lsearch0_t
        ///
        scalar_t get(const solver_state_t&) final;
    };
}
