#include "init_const.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch_const_init_t::config() const
{
    json_t json;
    json["alpha"] = strcat(m_alpha, "(0,10)");
    return json;
}

void lsearch_const_init_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    from_json_range(json, "alpha", m_alpha, eps, 10 - eps);
}

scalar_t lsearch_const_init_t::get(const solver_state_t&)
{
    return m_alpha;
}
