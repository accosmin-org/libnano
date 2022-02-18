#include <mutex>
#include <nano/lsearch0/linear.h>
#include <nano/lsearch0/constant.h>
#include <nano/lsearch0/cgdescent.h>
#include <nano/lsearch0/quadratic.h>

using namespace nano;

lsearch0_t::lsearch0_t()
{
    register_parameter(parameter_t::make_float("lsearch0::epsilon", 0, LT, 1e-6, LT, 1));
}

lsearch0_factory_t& lsearch0_t::all()
{
    static lsearch0_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<lsearch0_linear_t>("linear", "linearly interpolate the previous line-search step length");
        manager.add<lsearch0_constant_t>("constant", "constant line-search step length");
        manager.add<lsearch0_quadratic_t>("quadratic", "quadratically interpolate the previous line-search step length");
        manager.add<lsearch0_cgdescent_t>("cgdescent", "the initial line-search step length described in CG-DESCENT");
    });

    return manager;
}
