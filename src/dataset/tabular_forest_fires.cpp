#include <nano/dataset/tabular_forest_fires.h>

using namespace nano;

forest_fires_dataset_t::forest_fires_dataset_t()
{
    features(
    {
        feature_t{"X"}.labels(9),
        feature_t{"Y"}.labels(8),
        feature_t{"month"}.labels(12),
        feature_t{"day"}.labels(7),
        feature_t{"FFMC"},
        feature_t{"DMC"},
        feature_t{"DC"},
        feature_t{"ISI"},
        feature_t{"temp"},
        feature_t{"RH"},
        feature_t{"wind"},
        feature_t{"rain"},
        feature_t{"area"}
    }, 12);

    const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/forest-fires");
    csvs(
    {
        csv_t{dir + "/forestfires.csv"}.delim(",").header(true).expected(517)
    });
}
