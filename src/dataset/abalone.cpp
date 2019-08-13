#include "abalone.h"
#include <nano/mlearn.h>

using namespace nano;

abalone_dataset_t::abalone_dataset_t() :
    m_dir(std::getenv("HOME"))
{
    m_dir.append("/experiments/datasets/abalone");

    features(
    {
        feature_t{"sex"}.labels({"M", "F", "I"}),
        feature_t{"length"},
        feature_t{"diameter"},
        feature_t{"height"},
        feature_t{"whole_weight"},
        feature_t{"shucked_weight"},
        feature_t{"viscera_weight"},
        feature_t{"shell_weight"},
        feature_t{"rings"}.labels({
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29"
        }),
    }, 8);

    config(config());
}

json_t abalone_dataset_t::config() const
{
    json_t json;
    json["dir"] = m_dir;
    json["folds"] = strcat(m_folds, "[1,100]");
    json["train_per"] = strcat(m_train_per, "[10,90]");
    return json;
}

void abalone_dataset_t::config(const json_t& json)
{
    from_json(json, "dir", m_dir);
    from_json_range(json, "folds", m_folds, 1, 100);
    from_json_range(json, "train_per", m_train_per, 10, 90);

    csvs(
    {
        csv_t{m_dir + "/abalone.data"}.delim(",").header(false).expected(4177)
    });
    folds(m_folds);
}

void abalone_dataset_t::split(const tensor_size_t samples, split_t& split) const
{
    const auto tr_vd_size = 3133, te_size = 1044;
    NANO_UNUSED1_RELEASE(samples);
    assert(samples == tr_vd_size + te_size);

    split.m_te_indices = indices_t::LinSpaced(te_size, tr_vd_size, tr_vd_size + te_size);
    std::tie(split.m_tr_indices, split.m_vd_indices) = nano::split2(tr_vd_size, m_train_per);
}
