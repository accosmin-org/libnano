#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief limited memory BGFS (l-BGFS).
    ///     see "Updating Quasi-Newton Matrices with Limited Storage",
    ///     by J. Nocedal, 1980
    ///     see "Numerical Optimization",
    ///     by J. Nocedal, S. Wright, 2006
    ///
    class solver_lbfgs_t final : public solver_t
    {
    public:

        solver_lbfgs_t() = default;

        void to_json(json_t&) const final;
        void from_json(const json_t&) final;

        solver_state_t minimize(
            const size_t max_iterations, const scalar_t epsilon,
            const solver_function_t&, const vector_t& x0, const logger_t&) const final;

    private:

        // attributes
        lsearch_t::initializer  m_init{lsearch_t::initializer::quadratic};
        lsearch_t::strategy     m_strat{lsearch_t::strategy::morethuente};
        scalar_t                m_c1{static_cast<scalar_t>(1e-4)};
        scalar_t                m_c2{static_cast<scalar_t>(9e-1)};
        size_t                  m_history_size{6};      ///< history size (number of previous gradients to use)
    };
}
