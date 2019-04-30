#include "lemarechal.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearchk_lemarechal_t::config() const
{
    json_t json;
    json["tau1"] = strcat(m_tau1, "(2,inf)");
    json["interpolation"] = strcat(m_interpolation, join(enum_values<interpolation>()));
    return json;
}

void lsearchk_lemarechal_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    nano::from_json_range(json, "tau1", m_tau1, 2 + eps, inf);
    nano::from_json(json, "interpolation", m_interpolation);
}

bool lsearchk_lemarechal_t::get(const solver_state_t& state0, solver_state_t& state)
{
    lsearch_step_t L = state0;
    lsearch_step_t R = L;

    bool R_updated = false;
    for (int i = 1; i < max_iterations(); ++ i)
    {
        scalar_t tmin, tmax;
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
                    tmin = std::max(L.t, R.t) + 2 * std::fabs(L.t - R.t);
                    tmax = std::max(L.t, R.t) + m_tau1 * std::fabs(L.t - R.t);
                }
                else
                {
                    tmin = std::min(L.t, R.t);
                    tmax = std::max(L.t, R.t);
                }
            }
        }
        else
        {
            R = state;
            R_updated = true;
            tmin = std::min(L.t, R.t);
            tmax = std::max(L.t, R.t);
        }

        // next trial
        const auto next = lsearch_step_t::interpolate(L, R, m_interpolation);
        const auto ok = state.update(state0, clamp(next, tmin, tmax));
        log(state0, state);

        if (!ok)
        {
            return false;
        }
    }

    return false;
}
