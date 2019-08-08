#include <nano/mlearn.h>
#include "breast_cancer.h"

using namespace nano;

breast_cancer_dataset_t::breast_cancer_dataset_t() :
    m_dir(std::getenv("HOME"))
{
    m_dir.append("/experiments/datasets/breast-cancer");

    features(
    {
        feature_t::make_scalar("ID"),
        feature_t::make_discrete("Diagnosis", {"M", "B"}),

        feature_t::make_scalar("radius1"),
        feature_t::make_scalar("texture1"),
        feature_t::make_scalar("perimeter1"),
        feature_t::make_scalar("area1"),
        feature_t::make_scalar("smoothness1"),
        feature_t::make_scalar("compactness1"),
        feature_t::make_scalar("concavity1"),
        feature_t::make_scalar("concave_points1"),
        feature_t::make_scalar("symmetry1"),
        feature_t::make_scalar("fractal_dimension1"),

        feature_t::make_scalar("radius2"),
        feature_t::make_scalar("texture2"),
        feature_t::make_scalar("perimeter2"),
        feature_t::make_scalar("area2"),
        feature_t::make_scalar("smoothness2"),
        feature_t::make_scalar("compactness2"),
        feature_t::make_scalar("concavity2"),
        feature_t::make_scalar("concave_points2"),
        feature_t::make_scalar("symmetry2"),
        feature_t::make_scalar("fractal_dimension2"),

        feature_t::make_scalar("radius3"),
        feature_t::make_scalar("texture3"),
        feature_t::make_scalar("perimeter3"),
        feature_t::make_scalar("area3"),
        feature_t::make_scalar("smoothness3"),
        feature_t::make_scalar("compactness3"),
        feature_t::make_scalar("concavity3"),
        feature_t::make_scalar("concave_points3"),
        feature_t::make_scalar("symmetry3"),
        feature_t::make_scalar("fractal_dimension3")
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
        csv_t{m_dir + "/wdbc.data"}.delim(",").header(false)
    });
    folds(m_folds);
}

void breast_cancer_dataset_t::split(const tensor_size_t samples, split_t& split) const
{
    if (samples != 569)
    {
        throw std::invalid_argument(strcat("breast_cancer dataset: received ", samples, " samples, expecting 569"));
    }

    std::tie(split.m_tr_indices, split.m_vd_indices, split.m_te_indices) =
        nano::split3(samples, m_train_per, m_valid_per);
}
