#include "backtrack.h"
#include <nano/numeric.h>

using namespace nano;

void lsearch_backtrack_t::to_json(json_t& json) const
{
    nano::to_json(json,
        "ro", strcat(m_ro, "(0,1)"));
}

void lsearch_backtrack_t::from_json(const json_t& json)
{
    nano::from_json(json,
        "ro", m_ro);

    if (m_ro < epsilon0<scalar_t>() || m_ro > 1 - epsilon0<scalar_t>())
    {
        throw std::runtime_error("invalid 'ro' parameter");
    }
}

bool lsearch_backtrack_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
    for (int i = 0; i < max_iterations() && t > stpmin(); ++ i)
    {
        if (!state.update(state0, t))
        {
            return false;
        }
        else if (!state.has_armijo(state0, c1()))
        {
            t *= m_ro;
        }
        else
        {
            return true;
        }
    }

    return false;
}
