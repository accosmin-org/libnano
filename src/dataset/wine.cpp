#include "wine.h"
#include <nano/mlearn.h>

using namespace nano;

wine_dataset_t::wine_dataset_t() :
    m_dir(std::getenv("HOME"))
{
    m_dir.append("/experiments/datasets/wine");

    features(
    {
        feature_t::make_discrete("class", {"1", "2", "3"}),
        feature_t::make_scalar("Alcohol"),
        feature_t::make_scalar("Malic acid"),
        feature_t::make_scalar("Ash"),
        feature_t::make_scalar("Alcalinity of ash"),
        feature_t::make_scalar("Magnesium"),
        feature_t::make_scalar("Total phenols"),
        feature_t::make_scalar("Flavanoids"),
        feature_t::make_scalar("Nonflavanoid phenols"),
        feature_t::make_scalar("Proanthocyanins"),
        feature_t::make_scalar("Color intensity"),
        feature_t::make_scalar("Hue"),
        feature_t::make_scalar("OD280/OD315 of diluted wines"),
        feature_t::make_scalar("Proline"),
    }, 0);

    config(config());
}

json_t wine_dataset_t::config() const
{
    json_t json;
    json["dir"] = m_dir;
    json["folds"] = strcat(m_folds, "[1,100]");
    json["train_per"] = strcat(m_train_per, "[10,90]");
    json["valid_per"] = strcat(m_valid_per, "[10,90]");
    return json;
}

void wine_dataset_t::config(const json_t& json)
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
        csv_t{m_dir + "/wine.data"}.delim(",").header(false)
    });
    folds(m_folds);
}

void wine_dataset_t::split(const tensor_size_t samples, split_t& split) const
{
    if (samples != 178)
    {
        throw std::invalid_argument(strcat("wine dataset: received ", samples, " samples, expecting 178"));
    }

    std::tie(split.m_tr_indices, split.m_vd_indices, split.m_te_indices) =
        nano::split3(samples, m_train_per, m_valid_per);
}
