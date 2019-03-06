#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief gradient descent with line-search.
    ///
    class solver_gd_t final : public solver_t
    {
    public:

        solver_gd_t() = default;

        void to_json(json_t&) const final;
        void from_json(const json_t&) final;

        solver_state_t minimize(
            const size_t max_iterations, const scalar_t epsilon,
            const solver_function_t&, const vector_t& x0, const logger_t&) const final;

    private:

        // attributes
        lsearch_t::initializer  m_init{lsearch_t::initializer::quadratic};
        lsearch_t::strategy     m_strat{lsearch_t::strategy::morethuente};
        scalar_t                m_c1{static_cast<scalar_t>(1e-1)};
        scalar_t                m_c2{static_cast<scalar_t>(9e-1)};
    };
}
