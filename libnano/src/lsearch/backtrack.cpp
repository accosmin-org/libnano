#include "backtrack.h"

using namespace nano;

json_t lsearch_backtrack_t::config() const
{
    json_t json;
    return json;
}

void lsearch_backtrack_t::config(const json_t&)
{
}

bool lsearch_backtrack_t::get(const solver_state_t& state0, solver_state_t& state)
{
    for (int i = 0; i < max_iterations(); ++ i)
    {
        if (state.has_armijo(state0, c1()))
        {
            return true;
        }

        // next trial
        state.update(state0, lsearch_step_t::interpolate(state0, state));
        log(state0, state);
    }

    return false;
}
