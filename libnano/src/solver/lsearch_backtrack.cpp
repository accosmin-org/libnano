#include "lsearch_backtrack.h"

using namespace nano;

void lsearch_backtrack_t::to_json(json_t& json) const
{
    nano::to_json(json, "decrement", m_decrement);
}

void lsearch_backtrack_t::from_json(const json_t& json)
{
    nano::from_json(json, "decrement", m_decrement);
    // todo: check parameters!
}

bool lsearch_backtrack_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
    const auto decmin = scalar_t(0.1);
    const auto decmax = scalar_t(0.9);

    if (m_decrement < decmin || m_decrement > decmax)
    {
        return false;
    }

    for (int i = 0; i < max_iterations() && t > stpmin(); ++ i)
    {
        if (!state.update(state0, t))
        {
            return false;
        }
        else if (!state.has_armijo(state0, c1()))
        {
            t *= m_decrement;
        }
        else
        {
            return true;
        }
    }

    return false;
}
