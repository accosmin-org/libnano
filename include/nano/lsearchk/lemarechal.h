#pragma once

#include <nano/lsearchk.h>

namespace nano
{
    ///
    /// \brief the line-search algorithm described here:
    ///     see "A view of line-searches", C. Lemarechal
    ///
    class NANO_PUBLIC lsearchk_lemarechal_t final : public lsearchk_t
    {
    public:

        ///
        /// \brief constructor
        ///
        lsearchk_lemarechal_t();

        ///
        /// \brief @see lsearchk_t
        ///
        rlsearchk_t clone() const final;

        ///
        /// \brief @see lsearchk_t
        ///
        bool get(const solver_state_t& state0, solver_state_t& state) final;
    };
}
