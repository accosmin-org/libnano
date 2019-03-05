#include <nano/numeric.h>
#include "lsearch_lemarechal.h"

using namespace nano;

void lsearch_lemarechal_t::to_json(json_t& json) const
{
    nano::to_json(json, "increment", m_increment);
}

void lsearch_lemarechal_t::from_json(const json_t& json)
{
    nano::from_json(json, "increment", m_increment);
    // todo: check parameters!
}

bool lsearch_lemarechal_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
    lsearch_step_t L = state0;
    lsearch_step_t R = L;

    assert(L.t < epsilon0<scalar_t>());
    assert(R.t < epsilon0<scalar_t>());

    for (int i = 1; i < max_iterations() && t < stpmax(); ++ i)
    {
        if (!state.update(state0, t))
        {
            return false;
        }

        if (state.has_armijo(state0, c1()))
        {
            if (state.has_wolfe(state0, c2()))
            {
                return true;
            }
            else
            {
                L = state;

                if (R.t < epsilon0<scalar_t>())
                {
                    t *= m_increment;
                }
                else
                {
                    t = lsearch_step_t::cubic(L, R);
                }
            }
        }
        else
        {
            R = state;
            t = lsearch_step_t::cubic(L, R);
        }
    }

    return false;
}
