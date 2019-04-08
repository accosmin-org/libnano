#include "init_linear.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch_linear_init_t::config() const
{
    json_t json;
    json["tro"] = strcat(m_tro, "(0,1)");
    return json;
}

void lsearch_linear_init_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    from_json_range(json, "tro", m_tro, 1 + eps, inf);
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
        t0 = std::min(scalar_t(1), m_tro * state.t * m_prevdg / dg);
    }

    m_prevdg = dg;

    log(state, t0);
    return t0;
}
