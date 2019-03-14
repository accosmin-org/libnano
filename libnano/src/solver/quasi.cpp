#include "quasi.h"

using namespace nano;

template <typename tquasi>
solver_quasi_t<tquasi>::solver_quasi_t() :
    solver_t(1e-4, 9e-1)
{
}

template <typename tquasi>
void solver_quasi_t<tquasi>::to_json(json_t& json) const
{
    solver_t::to_json(json);
}

template <typename tquasi>
void solver_quasi_t<tquasi>::from_json(const json_t& json)
{
    solver_t::from_json(json);
}

template <typename tquasi>
solver_state_t solver_quasi_t<tquasi>::minimize(const solver_function_t& function, const vector_t& x0) const
{
    auto cstate = solver_state_t{function, x0};
    auto pstate = cstate;

    // current approximation of the Hessian
    matrix_t H = matrix_t::Identity(function.size(), function.size());

    for (int i = 0; i < max_iterations(); ++ i, ++ cstate.m_iterations)
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
        const auto iter_ok = lsearch(cstate);
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
template class nano::solver_quasi_t<quasi_step_broyden>;
