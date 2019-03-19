#include "backtrack.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch_backtrack_t::config() const
{
    json_t json;
    json["ro"] = strcat(m_ro, "(0,1)");
    return json;
}

void lsearch_backtrack_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    nano::from_json_range(json, "ro", m_ro, eps, 1 - eps);
}

bool lsearch_backtrack_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
    for (int i = 0; i < max_iterations() && t > stpmin(); ++ i)
    {
        const auto ok = state.update(state0, t);
        log(state);

        if (!ok)
        {
            t *= m_ro;
        }
        else if (!state.has_armijo(state0, c1()))
        {
            t *= m_ro;
        }
        else
        {
            return true;
        }
    }

    return false;
}
