#include <mutex>
#include <nano/splitter/kfold.h>

using namespace nano;

splitter_t::splitter_t(string_t id)
    : clonable_t(std::move(id))
{
    register_parameter(parameter_t::make_integer("splitter::folds", 2, LE, 10, LE, 100));
    register_parameter(parameter_t::make_integer("splitter::seed", 0, LE, 42, LE, 1024));
}

factory_t<splitter_t>& splitter_t::all()
{
    static auto manager = factory_t<splitter_t>{};
    const auto  op      = []() { manager.add<kfold_splitter_t>("k-fold cross-validation"); };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
