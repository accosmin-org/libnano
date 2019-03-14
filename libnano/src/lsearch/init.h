#pragma once

#include <nano/lsearch_init.h>

namespace nano
{
    ///
    /// \brief initial line-search step length strategies.
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
    ///     see CG_DESCENT papers
    ///

    ///
    /// \brief unit step length (useful for LBFGS, Quasi-Newton and Newton methods).
    ///
    class lsearch_unit_init_t final : public lsearch_init_t
    {
    public:

        lsearch_unit_init_t() = default;
        scalar_t get(const solver_state_t&, const int iteration) final;
    };

    ///
    /// \brief use linear interpolation of the previous line-search step lengths.
    ///
    class lsearch_linear_init_t final : public lsearch_init_t
    {
    public:

        lsearch_linear_init_t() = default;
        scalar_t get(const solver_state_t&, const int iteration) final;

    private:

        // attributes
        scalar_t    m_prevdg{1};    ///< previous direction dot product
    };

    ///
    /// \brief use quadratic interpolation of the previous line-search step lengths.
    ///
    class lsearch_quadratic_init_t final : public lsearch_init_t
    {
    public:

        lsearch_quadratic_init_t() = default;
        scalar_t get(const solver_state_t&, const int iteration) final;

    private:

        // attributes
        scalar_t    m_prevf{0};     ///< previous function value
    };

    ///
    /// \brief CG_DESCENT initial step length strategy.
    ///
    class lsearch_cgdescent_init_t final : public lsearch_init_t
    {
    public:

        lsearch_cgdescent_init_t() = default;
        scalar_t get(const solver_state_t&, const int iteration) final;

        void to_json(json_t&) const final;
        void from_json(const json_t&) final;

    private:

        // attributes
        scalar_t    m_phi0{static_cast<scalar_t>(0.01)};    ///<
        scalar_t    m_phi1{static_cast<scalar_t>(0.1)};     ///<
        scalar_t    m_phi2{static_cast<scalar_t>(2.0)};     ///<
    };
}
