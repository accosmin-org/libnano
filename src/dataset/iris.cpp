#include "iris.h"
#include <nano/mlearn.h>

using namespace nano;

iris_dataset_t::iris_dataset_t() :
    m_dir(std::getenv("HOME"))
{
    m_dir.append("/experiments/datasets/iris");

    features(
    {
        feature_t{"sepal_length_cm"},
        feature_t{"sepal_width_cm"},
        feature_t{"petal_length_cm"},
        feature_t{"petal_width_cm"},
        feature_t{"class"}.labels({"Iris-setosa", "Iris-versicolor", "Iris-virginica"})
    }, 7);

    config(config());
}

json_t iris_dataset_t::config() const
{
    json_t json;
    json["dir"] = m_dir;
    json["folds"] = strcat(m_folds, "[1,100]");
    json["train_per"] = strcat(m_train_per, "[10,90]");
    json["valid_per"] = strcat(m_valid_per, "[10,90]");
    return json;
}

void iris_dataset_t::config(const json_t& json)
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
        csv_t{m_dir + "/iris.data"}.delim(",").header(false).expected(150)
    });
    folds(m_folds);
}

void iris_dataset_t::split(const tensor_size_t samples, split_t& split) const
{
    assert(samples == 150);

    std::tie(split.m_tr_indices, split.m_vd_indices, split.m_te_indices) =
        nano::split3(samples, m_train_per, m_valid_per);
}
