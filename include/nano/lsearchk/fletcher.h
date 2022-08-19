#pragma once

#include <nano/lsearchk.h>

namespace nano
{
    ///
    /// \brief the More&Thuente-like line-search algorithm described here:
    ///     see (1) "Practical methods of optimization", Fletcher, 2nd edition, p.34
    ///     see (2) "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
    ///
    /// NB: the implementation follows the notation from (1).
    ///
    class NANO_PUBLIC lsearchk_fletcher_t final : public lsearchk_t
    {
    public:
        ///
        /// \brief constructor
        ///
        lsearchk_fletcher_t();

        ///
        /// \brief @see lsearchk_t
        ///
        rlsearchk_t clone() const final;

        ///
        /// \brief @see lsearchk_t
        ///
        bool get(const solver_state_t& state0, solver_state_t& state) const final;

    private:
        bool zoom(const solver_state_t&, lsearch_step_t lo, lsearch_step_t hi, solver_state_t&) const;
    };
} // namespace nano
