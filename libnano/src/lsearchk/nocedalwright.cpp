#include <nano/numeric.h>
#include "nocedalwright.h"

using namespace nano;

json_t lsearchk_nocedalwright_t::config() const
{
    json_t json;
    json["ro"] = strcat(m_ro, "(1,inf)");
    json["interpolation"] = strcat(m_interpolation, join(enum_values<interpolation>()));
    return json;
}

void lsearchk_nocedalwright_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    nano::from_json_range(json, "ro", m_ro, 1 + eps, inf);
    nano::from_json(json, "interpolation", m_interpolation);
}

bool lsearchk_nocedalwright_t::zoom(const solver_state_t& state0,
    lsearch_step_t lo, lsearch_step_t hi, solver_state_t& state) const
{
    for (int i = 0; i < max_iterations() && std::fabs(lo.t - hi.t) > epsilon0<scalar_t>(); ++ i)
    {
        const auto ok = state.update(state0, lsearch_step_t::interpolate(lo, hi, m_interpolation));
        log(state0, state);

        if (!ok)
        {
            return false;
        }
        else if (!state.has_armijo(state0, c1()) || state.f >= lo.f)
        {
            hi = state;
        }
        else
        {
            if (state.has_strong_wolfe(state0, c2()))
            {
                return true;
            }

            if (state.dg() * (hi.t - lo.t) >= scalar_t(0))
            {
                hi = lo;
            }
            lo = state;
        }
    }

    return false;
}

bool lsearchk_nocedalwright_t::get(const solver_state_t& state0, solver_state_t& state)
{
    lsearch_step_t prev = state0;
    lsearch_step_t curr = state;

    for (int i = 1; i < max_iterations(); ++ i)
    {
        if (!state.has_armijo(state0, c1()) || (state.f >= prev.f && i > 1))
        {
            return zoom(state0, prev, curr, state);
        }
        else if (state.has_strong_wolfe(state0, c2()))
        {
            return true;
        }
        else if (!state.has_descent())
        {
            return zoom(state0, curr, prev, state);
        }

        prev = curr;
        state.t *= m_ro;

        // next trial
        const auto ok = state.update(state0, state.t);
        log(state0, state);
        if (!ok)
        {
            return false;
        }
        curr = state;
    }

    return false;
}
