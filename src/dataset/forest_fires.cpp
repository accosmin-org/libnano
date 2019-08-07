#include "forest_fires.h"
#include <nano/mlearn.h>

using namespace nano;

forest_fires_dataset_t::forest_fires_dataset_t() :
    m_dir(std::getenv("HOME"))
{
    m_dir.append("/experiments/datasets/forest-fires");

    delim(",");
    header(true);
    paths({m_dir + "/forestfires.csv"});
    features(
    {
        feature_t::make_discrete("X", {"1", "2", "3", "4", "5", "6", "7", "8", "9"}),
        feature_t::make_discrete("Y", {"2", "3", "4", "5", "6", "7", "8", "9"}),
        feature_t::make_discrete("month", {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}),
        feature_t::make_discrete("day", {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}),
        feature_t::make_scalar("FFMC"),
        feature_t::make_scalar("DMC"),
        feature_t::make_scalar("DC"),
        feature_t::make_scalar("ISI"),
        feature_t::make_scalar("temp"),
        feature_t::make_scalar("RH"),
        feature_t::make_scalar("wind"),
        feature_t::make_scalar("rain"),
        feature_t::make_scalar("area")
    }, 12);
    folds(m_folds);
}

json_t forest_fires_dataset_t::config() const
{
    json_t json;
    json["dir"] = m_dir;
    json["folds"] = strcat(m_folds, "[1,100]");
    json["train_per"] = strcat(m_train_per, "[10,90]");
    json["valid_per"] = strcat(m_valid_per, "[10,90]");
    return json;
}

void forest_fires_dataset_t::config(const json_t& json)
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

    folds(m_folds);
}

void forest_fires_dataset_t::split(const tensor_size_t samples, split_t& split) const
{
    if (samples != 517)
    {
        throw std::invalid_argument(strcat("wine dataset: received ", samples, " samples, expecting 517"));
    }

    std::tie(split.m_tr_indices, split.m_vd_indices, split.m_te_indices) =
        nano::split3(samples, m_train_per, m_valid_per);
}
