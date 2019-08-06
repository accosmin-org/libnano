#include <mutex>
#include "dataset/iris.h"
#include "dataset/wine.h"
#include "dataset/adult.h"

using namespace nano;

dataset_factory_t& dataset_t::all()
{
    static dataset_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<iris_dataset_t>("iris", "Iris dataset - classify flowers by sepal and petal (Fisher, 1936)");
        manager.add<wine_dataset_t>("wine", "Wine dataset - predict the wine type from its constituents (Aeberhard, Coomans & de Vel, 1992)");
        manager.add<adult_dataset_t>("adult", "Adult dataset - predict if a person makes more than 50K per year (Kohavi & Becker, 1994)");
    });

    return manager;
}
