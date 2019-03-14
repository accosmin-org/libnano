#include <mutex>
#include "lsearch/init.h"
#include "lsearch/backtrack.h"
#include "lsearch/cgdescent.h"
#include "lsearch/lemarechal.h"
#include "lsearch/morethuente.h"
#include "lsearch/nocedalwright.h"

using namespace nano;

lsearch_init_factory_t& nano::get_lsearch_inits()
{
    static lsearch_init_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<lsearch_unit_init_t>("unit", "unit line-search step length");
        manager.add<lsearch_linear_init_t>("linear", "linearly interpolate the previous line-search step");
        manager.add<lsearch_quadratic_init_t>("quadratic", "quadratically interpolate the previous line-search step");
        manager.add<lsearch_cgdescent_init_t>("cgdescent", "the initial line-search step length described in CG-DESCENT");
    });

    return manager;
}

lsearch_strategy_factory_t& nano::get_lsearch_strategies()
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
