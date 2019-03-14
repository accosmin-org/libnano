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

        solver_gd_t();
        void to_json(json_t&) const final;
        void from_json(const json_t&) final;
        solver_state_t minimize(const solver_function_t&, const vector_t& x0) const final;
    };
}
