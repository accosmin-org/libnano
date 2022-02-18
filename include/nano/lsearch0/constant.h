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
        /// \brief constructor
        ///
        lsearch0_constant_t();

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
