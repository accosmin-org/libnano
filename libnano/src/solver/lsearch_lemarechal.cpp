#include <nano/numeric.h>
#include "lsearch_lemarechal.h"

using namespace nano;

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
                    t *= 3;
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
