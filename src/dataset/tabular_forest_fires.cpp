#include <nano/dataset/tabular_forest_fires.h>

using namespace nano;

forest_fires_dataset_t::forest_fires_dataset_t()
{
    features(
    {
        feature_t{"X"}.labels({"1", "2", "3", "4", "5", "6", "7", "8", "9"}),
        feature_t{"Y"}.labels({"2", "3", "4", "5", "6", "7", "8", "9"}),
        feature_t{"month"}.labels({"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}),
        feature_t{"day"}.labels({"mon", "tue", "wed", "thu", "fri", "sat", "sun"}),
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
