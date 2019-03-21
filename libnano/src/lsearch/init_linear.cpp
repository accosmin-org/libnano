#include "init_linear.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch_linear_init_t::config() const
{
    json_t json;
    return json;
}

void lsearch_linear_init_t::config(const json_t&)
{
}

scalar_t lsearch_linear_init_t::get(const solver_state_t& state)
{
    scalar_t t0;

    const auto dg = state.dg();
    if (state.m_iterations <= 1)
    {
        t0 = 1;
    }
    else
    {
        t0 = clamp(state.t * m_prevdg / dg, state.t * 0.25, 1);
    }

    m_prevdg = dg;
    return t0;
}
