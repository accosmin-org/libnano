#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief quasi-Newton methods.
    ///     see (1) "Practical methods of optimization", Fletcher, 2nd edition
    ///     see (2) "Numerical optimization", Nocedal & Wright, 2nd edition
    ///     see (3) "Introductory Lectures on Convex Optimization (Applied Optimization)", Nesterov, 2013
    ///
    template <typename tquasi>
    class solver_quasi_t final : public solver_t
    {
    public:

        solver_quasi_t();
        json_t config() const final;
        void config(const json_t&) final;
        solver_state_t minimize(const solver_function_t&, const lsearch_t&, const vector_t& x0) const final;
    };

    struct quasi_step_DFP;
    struct quasi_step_SR1;
    struct quasi_step_BFGS;
    struct quasi_step_Hoshino;

    // create various quasi-Newton algorithms
    using solver_quasi_dfp_t = solver_quasi_t<quasi_step_DFP>;
    using solver_quasi_sr1_t = solver_quasi_t<quasi_step_SR1>;
    using solver_quasi_bfgs_t = solver_quasi_t<quasi_step_BFGS>;
    using solver_quasi_hoshino_t = solver_quasi_t<quasi_step_Hoshino>;
}
