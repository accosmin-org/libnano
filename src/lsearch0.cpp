#include <lsearch0/cgdescent.h>
#include <lsearch0/constant.h>
#include <lsearch0/linear.h>
#include <lsearch0/quadratic.h>
#include <mutex>

using namespace nano;

lsearch0_t::lsearch0_t(string_t id)
    : typed_t(std::move(id))
{
}

factory_t<lsearch0_t>& lsearch0_t::all()
{
    static auto manager = factory_t<lsearch0_t>{};
    const auto  op      = []()
    {
        manager.add<lsearch0_linear_t>("linearly interpolate the previous line-search step size");
        manager.add<lsearch0_constant_t>("constant line-search step size");
        manager.add<lsearch0_quadratic_t>("quadratically interpolate the previous line-search step size");
        manager.add<lsearch0_cgdescent_t>("the initial line-search step size described in CG-DESCENT");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
