#include <nano/dataset/tabular_iris.h>

using namespace nano;

iris_dataset_t::iris_dataset_t()
{
    features(
    {
        feature_t{"sepal_length_cm"},
        feature_t{"sepal_width_cm"},
        feature_t{"petal_length_cm"},
        feature_t{"petal_width_cm"},
        feature_t{"class"}.labels(3),
    }, 4);

    const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/iris");
    csvs(
    {
        csv_t{dir + "/iris.data"}.delim(",").header(false).expected(150)
    });
}
