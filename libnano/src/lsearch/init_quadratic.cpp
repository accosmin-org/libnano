#include <nano/numeric.h>
#include "init_quadratic.h"

using namespace nano;

json_t lsearch_quadratic_init_t::config() const
{
    json_t json;
    return json;
}

void lsearch_quadratic_init_t::config(const json_t&)
{
}

scalar_t lsearch_quadratic_init_t::get(const solver_state_t& state)
{
    scalar_t t0;

    if (state.m_iterations <= 1)
    {
        t0 = 1;
    }
    else
    {
        t0 = clamp(2 * (state.f - m_prevf) / m_prevdg, state.t * 0.25, 1);
    }

    m_prevf = state.f;
    m_prevdg = state.dg();
    return t0;
}
