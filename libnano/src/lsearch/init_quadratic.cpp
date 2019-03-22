#include <nano/numeric.h>
#include "init_quadratic.h"

using namespace nano;

json_t lsearch_quadratic_init_t::config() const
{
    json_t json;
    json["t0"] = strcat(m_t0, "(0,inf)");
    json["tro"] = strcat(m_tro, "(0,1)");
    json["tmax"] = strcat(m_tmax, "[1,inf)");
    return json;
}

void lsearch_quadratic_init_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    from_json_range(json, "t0", m_t0, eps, inf);
    from_json_range(json, "tro", m_tro, eps, 1 - eps);
    from_json_range(json, "tmax", m_tmax, 1 - eps, inf);
}

scalar_t lsearch_quadratic_init_t::get(const solver_state_t& state)
{
    scalar_t t0;

    if (state.m_iterations <= 1)
    {
        t0 = m_t0;
    }
    else
    {
        t0 = clamp(2 * (state.f - m_prevf) / m_prevdg, state.t * m_tro, m_tmax);
    }

    m_prevf = state.f;
    m_prevdg = state.dg();
    return t0;
}
