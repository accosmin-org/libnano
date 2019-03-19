#pragma once

#include <nano/lsearch/init.h>

namespace nano
{
    ///
    /// \brief CG_DESCENT initial step length strategy.
    ///
    class lsearch_cgdescent_init_t final : public lsearch_init_t
    {
    public:

        lsearch_cgdescent_init_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        scalar_t get(const solver_state_t&) final;

    private:

        // attributes
        scalar_t    m_phi0{static_cast<scalar_t>(0.01)};    ///<
        scalar_t    m_phi1{static_cast<scalar_t>(0.1)};     ///<
        scalar_t    m_phi2{static_cast<scalar_t>(2.0)};     ///<
    };
}
