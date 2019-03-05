#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief initial step length strategies
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59-60
    ///
    class lsearch_unit_init_t final : public lsearch_init_t
    {
    public:

        lsearch_unit_init_t() = default;
        void to_json(json_t&) const final {}
        void from_json(const json_t&) final {}
        scalar_t get(const solver_state_t&, const int iteration) final;
    };

    class lsearch_linear_init_t final : public lsearch_init_t
    {
    public:

        lsearch_linear_init_t() = default;
        void to_json(json_t&) const final {}
        void from_json(const json_t&) final {}
        scalar_t get(const solver_state_t&, const int iteration) final;

    private:

        // attributes
        scalar_t    m_prevdg{1};    ///< previous direction dot product
    };

    class lsearch_quadratic_init_t final : public lsearch_init_t
    {
    public:

        lsearch_quadratic_init_t() = default;
        void to_json(json_t&) const final {}
        void from_json(const json_t&) final {}
        scalar_t get(const solver_state_t&, const int iteration) final;

    private:

        // attributes
        scalar_t    m_prevf{0};     ///< previous function value
    };

    ///
    /// \brief CG_DESCENT initial step length strategy
    ///
    class lsearch_cgdescent_init_t final : public lsearch_init_t
    {
    public:

        lsearch_cgdescent_init_t() = default;
        void to_json(json_t&) const final {}
        void from_json(const json_t&) final {}
        scalar_t get(const solver_state_t&, const int iteration) final;
    };
}
