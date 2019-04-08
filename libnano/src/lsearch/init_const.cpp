#include "init_const.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch_const_init_t::config() const
{
    json_t json;
    json["t0"] = strcat(m_t0, "(0,inf)");
    return json;
}

void lsearch_const_init_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    from_json_range(json, "t0", m_t0, eps, inf);
}

scalar_t lsearch_const_init_t::get(const solver_state_t& state)
{
    log(state, m_t0);
    return m_t0;
}
