#include <mutex>
#include "dataset/iris.h"
#include "dataset/wine.h"
#include "dataset/adult.h"
#include "dataset/poker_hand.h"
#include "dataset/forest_fires.h"
#include "dataset/breast_cancer.h"

using namespace nano;

dataset_factory_t& dataset_t::all()
{
    static dataset_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<iris_dataset_t>("iris", "classify flowers by sepal and petal (Fisher, 1936)");
        manager.add<wine_dataset_t>("wine", "predict the wine type from its constituents (Aeberhard, Coomans & de Vel, 1992)");
        manager.add<adult_dataset_t>("adult", "predict if a person makes more than 50K per year (Kohavi & Becker, 1994)");
        manager.add<poker_hand_dataset_t>("poker-hand", "predict the poker hand from 5 cards (Cattral, Oppacher & Deugo, 2002)");
        manager.add<forest_fires_dataset_t>("forest-fires", "predict the burned area of the forest (Cortez & Morais, 2007)");
        manager.add<breast_cancer_dataset_t>("breast-cancer", "diagnostic breast cancer using measurements of cell nucleai (Street, Wolberg & Mangasarian, 1992)");
    });

    return manager;
}
