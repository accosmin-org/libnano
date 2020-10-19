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
        /// \brief default constructor
        ///
        lsearchk_fletcher_t() = default;

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
        void tau23(const scalar_t tau2, const scalar_t tau3) { m_tau23.set(tau2, tau3); }
        void interp(const interpolation interp) { m_interpolation = interp; }

        ///
        /// \brief access functions
        ///
        auto tau1() const { return m_tau1.get(); }
        auto tau2() const { return m_tau23.get1(); }
        auto tau3() const { return m_tau23.get2(); }
        auto interp() const { return m_interpolation; }

    private:

        bool zoom(const solver_state_t&, lsearch_step_t lo, lsearch_step_t hi, solver_state_t&) const;

        // attributes
        interpolation   m_interpolation{interpolation::cubic};                              ///<
        sparam1_t       m_tau1{"lsearchk::fletcher::tau1", 2, LT, 9.0, LT, 1e+6};           ///< see (1)
        sparam2_t       m_tau23{"lsearchk::fletcher::tau23", 0, LT, 0.1, LT, 0.5, LE, 0.5}; ///< see (1)
    };
}
