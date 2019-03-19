#include <nano/numeric.h>
#include "init_quadratic.h"

using namespace nano;

json_t lsearch_quadratic_init_t::config() const
{
    json_t json;
    // todo: alpha=1.01 as parameter
    return json;
}

void lsearch_quadratic_init_t::config(const json_t&)
{
}

scalar_t lsearch_quadratic_init_t::get(const solver_state_t& state)
{
    scalar_t t0;

    switch (state.m_iterations)
    {
    case 0:
    case 1:
        t0 = 1;
        break;

    default:
        t0 = scalar_t(1.01) * 2 * (state.f - m_prevf) / state.dg();
        break;
    }

    m_prevf = state.f;
    return t0;
}
