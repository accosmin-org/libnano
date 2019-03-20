#include <nano/numeric.h>
#include "init_quadratic.h"

using namespace nano;

json_t lsearch_quadratic_init_t::config() const
{
    json_t json;
    json["ro"] = strcat(m_ro, "(1,2)");
    return json;
}

void lsearch_quadratic_init_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    from_json_range(json, "ro", m_ro, 1 + eps, 2 - eps);
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
        t0 = std::min(scalar_t(1), m_ro * 2 * (state.f - m_prevf) / state.dg());
    }

    m_prevf = state.f;
    return t0;
}
