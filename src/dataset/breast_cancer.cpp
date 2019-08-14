#include <nano/mlearn.h>
#include "breast_cancer.h"

using namespace nano;

breast_cancer_dataset_t::breast_cancer_dataset_t() :
    m_dir(std::getenv("HOME"))
{
    m_dir.append("/libnano/datasets/breast-cancer");

    features(
    {
        feature_t{"ID"},
        feature_t{"Diagnosis"}.labels({"M", "B"}),

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

    config(config());
}

json_t breast_cancer_dataset_t::config() const
{
    json_t json;
    json["dir"] = m_dir;
    json["folds"] = strcat(m_folds, "[1,100]");
    json["train_per"] = strcat(m_train_per, "[10,90]");
    json["valid_per"] = strcat(m_valid_per, "[10,90]");
    return json;
}

void breast_cancer_dataset_t::config(const json_t& json)
{
    from_json(json, "dir", m_dir);
    from_json_range(json, "folds", m_folds, 1, 100);
    from_json_range(json, "train_per", m_train_per, 10, 90);
    from_json_range(json, "valid_per", m_valid_per, 10, 90);

    if (m_train_per + m_valid_per >= 100)
    {
        throw std::invalid_argument(
            "invalid JSON attributes 'train_per' and 'valid_per', expected to sum to less than 100");
    }

    csvs(
    {
        csv_t{m_dir + "/wdbc.data"}.delim(",").header(false).expected(569)
    });
    folds(m_folds);
}

void breast_cancer_dataset_t::split(const tensor_size_t samples, split_t& split) const
{
    assert(samples == 569);

    std::tie(split.m_tr_indices, split.m_vd_indices, split.m_te_indices) =
        nano::split3(samples, m_train_per, m_valid_per);
}
