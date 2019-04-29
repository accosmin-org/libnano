#pragma once

#include <nano/lsearch/lsearchk.h>

namespace nano
{
    ///
    /// \brief the More&Thuente-like line-search algorithm described here:
    ///     see (1) "Practical methods of optimization", Fletcher, 2nd edition, p.34
    ///     see (2) "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
    ///
    /// NB: the implementation follows the notation from (1).
    ///
    class lsearchk_fletcher_t final : public lsearchk_t
    {
    public:

        lsearchk_fletcher_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        bool get(const solver_state_t& state0, solver_state_t& state) final;

    private:

        bool zoom(const solver_state_t&, lsearch_step_t lo, lsearch_step_t hi, solver_state_t&) const;

        // attributes
        interpolation   m_interpolation{interpolation::cubic};  ///<
        scalar_t        m_tau1{9.0};                            ///< see (1)
        scalar_t        m_tau2{0.1};                            ///< see (1)
        scalar_t        m_tau3{0.5};                            ///< see (1)
    };
}
