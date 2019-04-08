#include "init_linear.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch_linear_init_t::config() const
{
    json_t json;
    json["t0"] = strcat(m_t0, "(0,inf)");
    json["tro"] = strcat(m_tro, "(0,1)");
    json["tmax"] = strcat(m_tmax, "[1,inf)");
    return json;
}

void lsearch_linear_init_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    from_json_range(json, "t0", m_t0, eps, inf);
    from_json_range(json, "tro", m_tro, eps, 1 - eps);
    from_json_range(json, "tmax", m_tmax, 1 - eps, inf);
}

scalar_t lsearch_linear_init_t::get(const solver_state_t& state)
{
    scalar_t t0;

    const auto dg = state.dg();
    if (state.m_iterations <= 1)
    {
        t0 = m_t0;
    }
    else
    {
        t0 = clamp(state.t * m_prevdg / dg, state.t * m_tro, m_tmax);
    }

    m_prevdg = dg;

    log(state, t0);
    return t0;
}
