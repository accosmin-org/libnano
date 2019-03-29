#include "lemarechal.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch_lemarechal_t::config() const
{
    json_t json;
    json["ro"] = strcat(m_ro, "(1,inf)");
    return json;
}

void lsearch_lemarechal_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    nano::from_json_range(json, "ro", m_ro, 1 + eps, inf);
}

static scalar_t safeguard(const lsearch_step_t& a, const lsearch_step_t& b)
{
    const auto tmin = std::min(a.t, b.t);
    const auto tmax = std::max(a.t, b.t);
    const auto tmin_safe = tmin + (tmax - tmin) / 20;
    const auto tmax_safe = tmax - (tmax - tmin) / 20;

    const auto tc = lsearch_step_t::cubic(a, b);
    const auto tb = lsearch_step_t::bisect(a, b);
    return (std::isfinite(tc) && tc > tmin_safe && tc < tmax_safe) ? tc : tb;
}

bool lsearch_lemarechal_t::get(const solver_state_t& state0, solver_state_t& state)
{
    lsearch_step_t L = state0;
    lsearch_step_t R = L;

    assert(L.t < epsilon0<scalar_t>());
    assert(R.t < epsilon0<scalar_t>());

    bool R_updated = false;
    for (int i = 1; i < max_iterations() && state.t > stpmin() && state.t < stpmax(); ++ i)
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
                    state.t = L.t * m_ro;
                }
                else
                {
                    state.t = safeguard(L, R);
                }
            }
        }
        else
        {
            R = state;
            R_updated = true;
            state.t = safeguard(L, R);
        }

        // next trial
        state.update(state0, state.t);
        log(state0, state);
    }

    return false;
}
