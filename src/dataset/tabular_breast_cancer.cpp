#include <nano/dataset/tabular_breast_cancer.h>

using namespace nano;

breast_cancer_dataset_t::breast_cancer_dataset_t()
{
    features(
    {
        feature_t{"ID"},
        feature_t{"Diagnosis"}.labels(2),

        feature_t{"radius1"},
        feature_t{"texture1"},
        feature_t{"perimeter1"},
        feature_t{"area1"},
        feature_t{"smoothness1"},
        feature_t{"compactness1"},
        feature_t{"concavity1"},
        feature_t{"concave_points1"},
        feature_t{"symmetry1"},
        feature_t{"fractal_dimension1"},

        feature_t{"radius2"},
        feature_t{"texture2"},
        feature_t{"perimeter2"},
        feature_t{"area2"},
        feature_t{"smoothness2"},
        feature_t{"compactness2"},
        feature_t{"concavity2"},
        feature_t{"concave_points2"},
        feature_t{"symmetry2"},
        feature_t{"fractal_dimension2"},

        feature_t{"radius3"},
        feature_t{"texture3"},
        feature_t{"perimeter3"},
        feature_t{"area3"},
        feature_t{"smoothness3"},
        feature_t{"compactness3"},
        feature_t{"concavity3"},
        feature_t{"concave_points3"},
        feature_t{"symmetry3"},
        feature_t{"fractal_dimension3"}
    }, 1);

    const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/breast-cancer");
    csvs(
    {
        csv_t{dir + "/wdbc.data"}.delim(",").header(false).expected(569)
    });
}
