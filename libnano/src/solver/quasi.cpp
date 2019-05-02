#include "quasi.h"

using namespace nano;

namespace nano
{
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
    /// \brief Symmetric Rank One (SR1).
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
    /// \brief Broyden-Fletcher-Goldfarb-Shanno (BFGS).
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
}

template <typename tquasi>
solver_quasi_t<tquasi>::solver_quasi_t() :
    solver_t(1e-4, 9e-1)
{
}

template <typename tquasi>
json_t solver_quasi_t<tquasi>::config() const
{
    json_t json = solver_t::config();
    return json;
}

template <typename tquasi>
void solver_quasi_t<tquasi>::config(const json_t& json)
{
    solver_t::config(json);
}

template <typename tquasi>
solver_state_t solver_quasi_t<tquasi>::minimize(const solver_function_t& function, const lsearch_t& lsearch,
    const vector_t& x0) const
{
    auto cstate = solver_state_t{function, x0};
    auto pstate = cstate;
    log(cstate);

    // current approximation of the Hessian
    matrix_t H = matrix_t::Identity(function.size(), function.size());

    for (int i = 0; i < max_iterations(); ++ i)
    {
        // descent direction
        cstate.d = -H * cstate.g;

        // restart:
        //  - if not a descent direction
        if (!cstate.has_descent())
        {
            cstate.d = -cstate.g;
            H.setIdentity();
        }

        // line-search
        pstate = cstate;
        const auto iter_ok = lsearch.get(cstate);
        if (solver_t::done(function, cstate, iter_ok))
        {
            break;
        }

        // update approximation of the Hessian
        H = tquasi::get(H, pstate, cstate);
    }

    return cstate;
}

template class nano::solver_quasi_t<quasi_step_DFP>;
template class nano::solver_quasi_t<quasi_step_SR1>;
template class nano::solver_quasi_t<quasi_step_BFGS>;
