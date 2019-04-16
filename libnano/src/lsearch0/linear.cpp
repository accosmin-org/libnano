#include "linear.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch0_linear_t::config() const
{
    json_t json;
    json["tro"] = strcat(m_tro, "(1,inf)");
    return json;
}

void lsearch0_linear_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    from_json_range(json, "tro", m_tro, 1 + eps, inf);
}

scalar_t lsearch0_linear_t::get(const solver_state_t& state)
{
    scalar_t t0;

    const auto dg = state.dg();
    if (state.m_iterations <= 1)
    {
        t0 = 1;
    }
    else
    {
        t0 = std::min(scalar_t(1), - m_tro * std::max(- state.t * m_prevdg, 10 * epsilon()) / dg);
    }

    m_prevdg = dg;

    log(state, t0);
    return t0;
}
