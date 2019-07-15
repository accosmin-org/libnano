#include <mutex>
#include <nano/dataset.h>

using namespace nano;

dataset_factory_t& dataset_t::all()
{
    static dataset_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        //manager.add<tabular_dataset_t>("tabular", "load tabular dataset from CSV file");
    });

    return manager;
}
