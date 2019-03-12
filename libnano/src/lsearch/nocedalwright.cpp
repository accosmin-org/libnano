#include <nano/numeric.h>
#include "nocedalwright.h"

using namespace nano;

void lsearch_nocedalwright_t::to_json(json_t& json) const
{
    nano::to_json(json, "ro", strcat(m_ro, "(1,inf)"));
}

void lsearch_nocedalwright_t::from_json(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    nano::from_json_range(json, "ro", m_ro, 1 + eps, inf);
}

bool lsearch_nocedalwright_t::zoom(const solver_state_t& state0,
    lsearch_step_t lo, lsearch_step_t hi, solver_state_t& state) const
{
    for (int i = 0; i < max_iterations() && std::fabs(lo.t - hi.t) > epsilon0<scalar_t>(); ++ i)
    {
        if (!state.update(state0, lsearch_step_t::cubic(lo, hi)))
        {
            log(state);
            return false;
        }

        else if (!state.has_armijo(state0, c1()) || state.f >= lo.f)
        {
            log(state);
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

bool lsearch_nocedalwright_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
    lsearch_step_t prev = state0;
    lsearch_step_t curr = prev;

    for (int i = 1; i < max_iterations() && t < stpmax(); ++ i)
    {
        if (!state.update(state0, t))
        {
            log(state);
            return false;
        }
        log(state);
        curr = state;

        if (!state.has_armijo(state0, c1()) || (state.f >= prev.f && i > 1))
        {
            return zoom(state0, prev, curr, state);
        }

        else if (state.has_strong_wolfe(state0, c2()))
        {
            return true;
        }

        if (!state.has_descent())
        {
            return zoom(state0, curr, prev, state);
        }

        prev = curr;
        t *= m_ro;
    }

    return false;
}
