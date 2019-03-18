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
        if (!scale(state0, t, state, m_ro, stpmin(), stpmax()))
        {
            return false;
        }
        else if (!state.has_armijo(state0, c1()))
        {
            log(state);
            t *= m_ro;
        }
        else
        {   log(state);
            return true;
        }
    }

    return false;
}
