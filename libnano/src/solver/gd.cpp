#include "gd.h"

using namespace nano;

solver_gd_t::solver_gd_t() :
    solver_t(1e-1, 9e-1)
{
}

void solver_gd_t::to_json(json_t& json) const
{
    solver_t::to_json(json);
}

void solver_gd_t::from_json(const json_t& json)
{
    solver_t::from_json(json);
}

solver_state_t solver_gd_t::minimize(const solver_function_t& function, const vector_t& x0) const
{
    auto cstate = solver_state_t{function, x0};
    for (int i = 0; i < max_iterations(); ++ i, ++ cstate.m_iterations)
    {
        // descent direction
        cstate.d = -cstate.g;

        // line-search
        const auto iter_ok = lsearch(cstate);
        if (solver_t::done(function, cstate, iter_ok))
        {
            break;
        }
    }

    return cstate;
}
