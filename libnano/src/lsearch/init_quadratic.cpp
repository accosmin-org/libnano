#include <nano/numeric.h>
#include "init_quadratic.h"

using namespace nano;

json_t lsearch_quadratic_init_t::config() const
{
    json_t json;
    json["tro"] = strcat(m_tro, "(1,inf)");
    return json;
}

void lsearch_quadratic_init_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    from_json_range(json, "tro", m_tro, 1 + eps, inf);
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
        t0 = std::min(scalar_t(1), m_tro * 2 * (state.f - m_prevf) / m_prevdg);
    }

    m_prevf = state.f;
    m_prevdg = state.dg();

    log(state, t0);
    return t0;
}
