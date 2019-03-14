#include <mutex>
#include "lsearch/backtrack.h"
#include "lsearch/cgdescent.h"
#include "lsearch/lemarechal.h"
#include "lsearch/morethuente.h"
#include "lsearch/nocedalwright.h"

using namespace nano;

lsearch_algo_factory_t& nano::get_lsearch_algos()
{
    static lsearch_algo_factory_t manager;

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
