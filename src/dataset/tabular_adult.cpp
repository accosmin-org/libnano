#include <nano/dataset/tabular_adult.h>

using namespace nano;

adult_dataset_t::adult_dataset_t()
{
    features(
    {
        feature_t{"age"},
        feature_t{"workclass"}.placeholder("?").labels(8),
        feature_t{"fnlwgt"},
        feature_t{"education"}.labels(16),
        feature_t{"education-num"},
        feature_t{"marital-status"}.labels(7),
        feature_t{"occupation"}.placeholder("?").labels(14),
        feature_t{"relationship"}.labels(6),
        feature_t{"race"}.labels(5),
        feature_t{"sex"}.labels({"Female", "Male"}),
        feature_t{"capital-gain"},
        feature_t{"capital-loss"},
        feature_t{"hours-per-week"},
        feature_t{"native-country"}.placeholder("?").labels(41),
        feature_t{"income"}.labels(2),
    }, 14);

    const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/adult");
    csvs(
    {
        csv_t{dir + "/adult.data"}.skip('|').delim(", .").header(false).expected(32561),
        csv_t{dir + "/adult.test"}.skip('|').delim(", .").header(false).expected(16281).testing(make_range(0, 16281))
    });
}
