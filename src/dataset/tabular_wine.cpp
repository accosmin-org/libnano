#include <nano/dataset/tabular_wine.h>

using namespace nano;

wine_dataset_t::wine_dataset_t()
{
    features(
    {
        feature_t{"class"}.labels({"1", "2", "3"}),
        feature_t{"Alcohol"},
        feature_t{"Malic acid"},
        feature_t{"Ash"},
        feature_t{"Alcalinity of ash"},
        feature_t{"Magnesium"},
        feature_t{"Total phenols"},
        feature_t{"Flavanoids"},
        feature_t{"Nonflavanoid phenols"},
        feature_t{"Proanthocyanins"},
        feature_t{"Color intensity"},
        feature_t{"Hue"},
        feature_t{"OD280/OD315 of diluted wines"},
        feature_t{"Proline"},
    }, 0);

    const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/wine");
    csvs(
    {
        csv_t{dir + "/wine.data"}.delim(",").header(false).expected(178)
    });
}
