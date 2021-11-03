#pragma once

#include <nano/lsearchk.h>

namespace nano
{
    ///
    /// \brief backtracking line-search that stops when the Armijo condition is satisfied,
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
    ///
    class NANO_PUBLIC lsearchk_backtrack_t final : public lsearchk_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        lsearchk_backtrack_t() = default; // LCOV_EXCL_LINE

        ///
        /// \brief @see lsearchk_t
        ///
        rlsearchk_t clone() const final;

        ///
        /// \brief @see lsearchk_t
        ///
        bool get(const solver_state_t& state0, solver_state_t& state) final;

        ///
        /// \brief change parameters
        ///
        void interp(interpolation interp) { m_interpolation = interp; }

        ///
        /// \brief access functions
        ///
        auto interp() const { return m_interpolation; }

    private:

        // attributes
        interpolation   m_interpolation{interpolation::cubic};  ///<
    };
}
