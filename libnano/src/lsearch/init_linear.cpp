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

scalar_t lsearch_linear_init_t::get(const solver_state_t& state, const int iteration)
{
    scalar_t t0;

    const auto dg = state.dg();
    switch (iteration)
    {
    case 0:
        t0 = 1;
        break;

    default:
        // NB: the line-search length is from the previous iteration!
        t0 = state.t * m_prevdg / dg;
        break;
    }

    m_prevdg = dg;
    return t0;
}
