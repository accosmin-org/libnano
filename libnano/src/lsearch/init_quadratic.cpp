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

scalar_t lsearch_quadratic_init_t::get(const solver_state_t& state, const int iteration)
{
    scalar_t t0;

    switch (iteration)
    {
    case 0:
        t0 = 1;
        break;

    default:
        t0 = scalar_t(1.01) * 2 * (state.f - m_prevf) / state.dg();
        break;
    }

    m_prevf = state.f;
    return t0;
}
