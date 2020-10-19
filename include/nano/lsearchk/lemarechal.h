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
        /// \brief default constructor
        ///
        lsearchk_lemarechal_t() = default;

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
        void tau1(const scalar_t tau1) { m_tau1 = tau1; }
        void interp(const interpolation interp) { m_interpolation = interp; }

        ///
        /// \brief access functions
        ///
        auto tau1() const { return m_tau1.get(); }
        auto interp() const { return m_interpolation; }

    private:

        // attributes
        interpolation   m_interpolation{interpolation::cubic};                      ///<
        sparam1_t       m_tau1{"lsearchk::lemarechal::tau1", 2, LT, 9, LT, 1e+6};   ///< see (1)
    };
}
