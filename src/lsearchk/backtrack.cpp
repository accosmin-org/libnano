#include "backtrack.h"

using namespace nano;

json_t lsearchk_backtrack_t::config() const
{
    json_t json;
    json["interpolation"] = strcat(m_interpolation, join(enum_values<interpolation>()));
    return json;
}

void lsearchk_backtrack_t::config(const json_t& json)
{
    nano::from_json(json, "interpolation", m_interpolation);
}

bool lsearchk_backtrack_t::get(const solver_state_t& state0, solver_state_t& state)
{
    for (int i = 0; i < max_iterations() && state; ++ i)
    {
        if (state.has_armijo(state0, c1()))
        {
            return true;
        }

        // next trial
        state.update(state0, lsearch_step_t::interpolate(state0, state, m_interpolation));
        log(state0, state);
    }

    return false;
}
