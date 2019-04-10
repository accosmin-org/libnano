#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief quasi-Newton methods.
    ///     see "Numerical Optimization",
    ///     by J. Nocedal, S. Wright, 2006
    ///
    ///     see "Introductory Lectures on Convex Optimization (Applied Optimization)",
    ///     by Y. Nesterov, 2013
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

    ///
    /// \brief Davidon-Fletcher-Powell (DFP).
    ///
    struct quasi_step_DFP
    {
        static auto get(const matrix_t& H, const solver_state_t& prev, const solver_state_t& curr)
        {
            const auto dx = curr.x - prev.x;
            const auto dg = curr.g - prev.g;

            return  H + (dx * dx.transpose()) / dx.dot(dg) -
                    (H * dg * dg.transpose() * H) / (dg.transpose() * H * dg);
        }
    };

    ///
    /// \brief Symmetric Rank 1 (SR1).
    ///
    struct quasi_step_SR1
    {
        static auto get(const matrix_t& H, const solver_state_t& prev, const solver_state_t& curr)
        {
            const auto dx = curr.x - prev.x;
            const auto dg = curr.g - prev.g;

            return  H + (dx - H * dg) * (dx - H * dg).transpose() /
                    (dx - H * dg).dot(dg);
        }
    };

    ///
    /// \brief Broyden–Fletcher–Goldfarb–Shanno (BFGS).
    ///
    struct quasi_step_BFGS
    {
        static auto get(const matrix_t& H, const solver_state_t& prev, const solver_state_t& curr)
        {
            const auto dx = curr.x - prev.x;
            const auto dg = curr.g - prev.g;

            const auto I = matrix_t::Identity(H.rows(), H.cols());

            return  (I - dx * dg.transpose() / dx.dot(dg)) * H * (I - dg * dx.transpose() / dx.dot(dg)) +
                    dx * dx.transpose() / dx.dot(dg);
        }
    };

    ///
    /// \brief Broyden's method.
    ///
    struct quasi_step_broyden
    {
        static auto get(const matrix_t& H, const solver_state_t& prev, const solver_state_t& curr)
        {
            const auto dx = curr.x - prev.x;
            const auto dg = curr.g - prev.g;

            return  H + (dx - H * dg) * dx.transpose() * H / (dx.transpose() * H * dg);
        }
    };

    // create various quasi-Newton algorithms
    using solver_quasi_dfp_t = solver_quasi_t<quasi_step_DFP>;
    using solver_quasi_sr1_t = solver_quasi_t<quasi_step_SR1>;
    using solver_quasi_bfgs_t = solver_quasi_t<quasi_step_BFGS>;
    using solver_quasi_broyden_t = solver_quasi_t<quasi_step_broyden>;
}
