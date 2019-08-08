#include "poker_hand.h"
#include <nano/mlearn.h>

using namespace nano;

poker_hand_dataset_t::poker_hand_dataset_t() :
    m_dir(std::getenv("HOME"))
{
    m_dir.append("/experiments/datasets/poker-hand");

    features(
    {
        feature_t::make_discrete("S1", {"1", "2", "3", "4"}),
        feature_t::make_discrete("C1", {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t::make_discrete("S2", {"1", "2", "3", "4"}),
        feature_t::make_discrete("C2", {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t::make_discrete("S3", {"1", "2", "3", "4"}),
        feature_t::make_discrete("C3", {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t::make_discrete("S4", {"1", "2", "3", "4"}),
        feature_t::make_discrete("C4", {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t::make_discrete("S5", {"1", "2", "3", "4"}),
        feature_t::make_discrete("C5", {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t::make_discrete("CLASS", {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
    }, 10);

    config(config());
}

json_t poker_hand_dataset_t::config() const
{
    json_t json;
    json["dir"] = m_dir;
    json["folds"] = strcat(m_folds, "[1,100]");
    json["train_per"] = strcat(m_train_per, "[10,90]");
    return json;
}

void poker_hand_dataset_t::config(const json_t& json)
{
    from_json(json, "dir", m_dir);
    from_json_range(json, "folds", m_folds, 1, 100);
    from_json_range(json, "train_per", m_train_per, 10, 90);

    csvs(
    {
        csv_t{m_dir + "/poker-hand-training-true.data"}.delim(",\r").header(false),
        csv_t{m_dir + "/poker-hand-testing.data"}.delim(",\r").header(false)
    });
    folds(m_folds);
}

void poker_hand_dataset_t::split(const tensor_size_t samples, split_t& split) const
{
    const auto tr_vd_size = 25010;
    const auto te_size = 1000000;

    if (samples != tr_vd_size + te_size)
    {
        throw std::invalid_argument(strcat("adult dataset: received ", samples, " samples, expecting ", tr_vd_size + te_size));
    }

    split.m_te_indices = indices_t::LinSpaced(te_size, tr_vd_size, tr_vd_size + te_size);
    std::tie(split.m_tr_indices, split.m_vd_indices) = nano::split2(tr_vd_size, m_train_per);
}
