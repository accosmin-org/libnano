#include <mutex>
#include "dataset/iris.h"

using namespace nano;

dataset_factory_t& dataset_t::all()
{
    static dataset_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<iris_dataset_t>("iris", "Iris dataset - classify flowers by sepal and petal (Fisher, 1936)");
    });

    return manager;
}
