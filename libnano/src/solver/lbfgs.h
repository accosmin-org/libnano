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

        solver_lbfgs_t();
        json_t config() const final;
        void config(const json_t&) final;
        solver_state_t minimize(const solver_function_t&, const vector_t& x0) const final;

    private:

        // attributes
        size_t          m_history_size{6};      ///< history size (number of previous gradients to use)
    };
}
