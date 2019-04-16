#include "linear.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch0_linear_t::config() const
{
    json_t json;
    json["_"] = "t0 := min(1, -alpha * max(-t_{k-1} * dg_{k-1}, beta * epsilon) / dg_{k})";
    json["alpha"] = strcat(m_alpha, "(1,inf)");
    json["beta"] = strcat(m_beta, "(1, inf)");
    return json;
}

void lsearch0_linear_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    from_json_range(json, "alpha", m_alpha, 1 + eps, inf);
    from_json_range(json, "beta", m_beta, 1 + eps, inf);
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
        t0 = std::min(scalar_t(1), - m_alpha * std::max(- state.t * m_prevdg, m_beta * epsilon()) / dg);
    }

    m_prevdg = dg;

    log(state, t0);
    return t0;
}
