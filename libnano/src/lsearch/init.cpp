#include <mutex>
#include "init_unit.h"
#include "init_linear.h"
#include "init_cgdescent.h"
#include "init_quadratic.h"

using namespace nano;

lsearch_init_factory_t& lsearch_init_t::all()
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
