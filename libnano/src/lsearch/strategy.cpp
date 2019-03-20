#include <mutex>
#include "backtrack.h"
#include "cgdescent.h"
#include "lemarechal.h"
#include "morethuente.h"
#include "nocedalwright.h"
#include <nano/numeric.h>

using namespace nano;

lsearch_strategy_factory_t& lsearch_strategy_t::all()
{
    static lsearch_strategy_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<lsearch_backtrack_t>("backtrack", "backtracking using Armijo conditions");
        manager.add<lsearch_cgdescent_t>("cgdescent", "CG-DESCENT using strong Wolfe conditions");
        manager.add<lsearch_lemarechal_t>("lemarechal", "LeMarechal using regular Wolfe conditions");
        manager.add<lsearch_morethuente_t>("morethuente", "More&Thuente using strong Wolfe conditions");
        manager.add<lsearch_nocedalwright_t>("nocedalwright", "Nocedal&Wright using strong Wolfe conditions");
    });

    return manager;
}

bool lsearch_strategy_t::get(solver_state_t& state, scalar_t t)
{
    // check descent direction
    if (!state.has_descent())
    {
        return false;
    }

    // adjust the initial step length if it produces an invalid state
    const auto state0 = state;

    t = std::isfinite(t) ? nano::clamp(t, stpmin(), scalar_t(1)) : scalar_t(1);
    for (int i = 0; i < max_iterations(); ++ i)
    {
        const auto ok = state.update(state0, t);
        log(state0, state);

        if (!ok)
        {
            t *= 0.5;
        }
        else
        {
            break;
        }
    }

    // line-search step length
    return get(state0, state) && state && (state < state0);
}
