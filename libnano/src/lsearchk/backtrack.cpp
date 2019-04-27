#include "backtrack.h"

using namespace nano;

json_t lsearchk_backtrack_t::config() const
{
    json_t json;
    json["interpolation"] = strcat(m_method, join(enum_values<lsearchk_backtrack_t::interpolation>()));
    return json;
}

void lsearchk_backtrack_t::config(const json_t& json)
{
    nano::from_json(json, "interpolation", m_method);
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
        scalar_t t = 0;
        switch (m_method)
        {
        case interpolation::cubic:
            t = lsearch_step_t::cubic(state0, state);
            if (std::isfinite(t) && t < state0.t && t > state.t)
            {
                break;
            }
            // NB: if cubic interpolation fails, fallback to bisection!

        case interpolation::bisect:
        default:
            t = lsearch_step_t::bisect(state0, state);
            break;
        }

        state.update(state0, t);
        log(state0, state);
    }

    return false;
}
