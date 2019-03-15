#include <mutex>
#include "backtrack.h"
#include "cgdescent.h"
#include "lemarechal.h"
#include "morethuente.h"
#include "nocedalwright.h"

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
