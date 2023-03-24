#include <mutex>
#include <nano/lsearch0/cgdescent.h>
#include <nano/lsearch0/constant.h>
#include <nano/lsearch0/linear.h>
#include <nano/lsearch0/quadratic.h>

using namespace nano;

lsearch0_t::lsearch0_t(string_t id)
    : clonable_t(std::move(id))
{
    register_parameter(parameter_t::make_scalar("lsearch0::epsilon", 0, LT, 1e-6, LT, 1));
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
