#include "fletcher.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearchk_fletcher_t::config() const
{
    json_t json;
    json["tau1"] = strcat(m_tau1, "(1,inf)");
    json["tau2"] = strcat(m_tau2, "(0,0.5)");
    json["tau3"] = strcat(m_tau3, "(tau2,0.5)");
    json["interpolation"] = strcat(m_interpolation, join(enum_values<interpolation>()));
    return json;
}

void lsearchk_fletcher_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    nano::from_json_range(json, "tau1", m_tau1, 1 + eps, inf);
    nano::from_json_range(json, "tau2", m_tau2, eps, 0.5 + eps);
    nano::from_json_range(json, "tau3", m_tau3, m_tau2 + eps, 0.5 + eps);
    nano::from_json(json, "interpolation", m_interpolation);
}

bool lsearchk_fletcher_t::zoom(const solver_state_t& state0,
    lsearch_step_t lo, lsearch_step_t hi, solver_state_t& state) const
{
    for (int i = 0; i < max_iterations() && std::fabs(lo.t - hi.t) > epsilon0<scalar_t>(); ++ i)
    {
        const auto tmin = lo.t + std::min(m_tau2, c2()) * (hi.t - lo.t);
        const auto tmax = hi.t - m_tau3 * (hi.t - lo.t);
        const auto next = lsearch_step_t::interpolate(lo, hi, m_interpolation);
        const auto ok = state.update(state0, clamp(next, std::min(tmin, tmax), std::max(tmin, tmax)));
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

bool lsearchk_fletcher_t::get(const solver_state_t& state0, solver_state_t& state)
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

        // next trial
        const auto tmin = 2 * curr.t - prev.t;
        const auto tmax = curr.t + m_tau1 * (curr.t - prev.t);
        const auto next = state.t * 3;//lsearch_step_t::interpolate(prev, curr, m_interpolation);
        const auto ok = state.update(state0, clamp(next, tmin, tmax));
        log(state0, state);

        if (!ok)
        {
            return false;
        }
        prev = curr;
        curr = state;
    }

    return false;
}
