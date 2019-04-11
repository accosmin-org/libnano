#include "lemarechal.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearchk_lemarechal_t::config() const
{
    json_t json;
    json["ro"] = strcat(m_ro, "(1,inf)");
    return json;
}

void lsearchk_lemarechal_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    nano::from_json_range(json, "ro", m_ro, 1 + eps, inf);
}

bool lsearchk_lemarechal_t::get(const solver_state_t& state0, solver_state_t& state)
{
    lsearch_step_t L = state0;
    lsearch_step_t R = L;

    bool R_updated = false;
    for (int i = 1; i < max_iterations(); ++ i)
    {
        if (state.has_armijo(state0, c1()))
        {
            if (state.has_wolfe(state0, c2()))
            {
                return true;
            }
            else
            {
                L = state;
                if (!R_updated)
                {
                    state.t *= m_ro;
                }
                else
                {
                    state.t = lsearch_step_t::interpolate(L, R);
                }
            }
        }
        else
        {
            R = state;
            R_updated = true;
            state.t = lsearch_step_t::interpolate(L, R);
        }

        // next trial
        state.update(state0, state.t);
        log(state0, state);
    }

    return false;
}
