#include <nano/mlearn.h>
#include "bank_marketing.h"

using namespace nano;

bank_marketing_dataset_t::bank_marketing_dataset_t() :
    m_dir(std::getenv("HOME"))
{
    m_dir.append("/libnano/datasets/bank-marketing");

    features(
    {
        feature_t{"age"},
        feature_t{"job"}.labels({
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
            "services", "student", "technician", "unemployed", "unknown"
        }),
        feature_t{"marital"}.labels({"divorced", "married", "single", "unknown"}),
        feature_t{"education"}.labels({
            "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
            "professional.course", "university.degree", "unknown"
        }),
        feature_t{"default"}.labels({"no", "yes", "unknown"}),
        feature_t{"housing"}.labels({"no", "yes", "unknown"}),
        feature_t{"loan"}.labels({"no", "yes", "unknown"}),
        feature_t{"contact"}.labels({"cellular", "telephone"}),
        feature_t{"month"}.labels({"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}),
        feature_t{"day_of_week"}.labels({"mon", "tue", "wed", "thu", "fri"}),
        feature_t{"duration"},
        feature_t{"campaign"},
        feature_t{"pdays"},
        feature_t{"previous"},
        feature_t{"poutcome"}.labels({"failure", "nonexistent", "success"}),
        feature_t{"emp.var.rate"},
        feature_t{"cons.price.idx"},
        feature_t{"cons.conf.idx"},
        feature_t{"euribor3m"},
        feature_t{"nr.employed"},
        feature_t{"y"}.labels({"yes", "no"})
    }, 20);

    config(config());
}

json_t bank_marketing_dataset_t::config() const
{
    json_t json;
    json["dir"] = m_dir;
    json["folds"] = strcat(m_folds, "[1,100]");
    json["train_per"] = strcat(m_train_per, "[10,90]");
    json["valid_per"] = strcat(m_valid_per, "[10,90]");
    return json;
}

void bank_marketing_dataset_t::config(const json_t& json)
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
        csv_t{m_dir + "/bank-additional-full.csv"}.delim(";\"\r").header(true).expected(41188)
    });
    folds(m_folds);
}

void bank_marketing_dataset_t::split(const tensor_size_t samples, split_t& split) const
{
    assert(samples == 41188);

    std::tie(split.m_tr_indices, split.m_vd_indices, split.m_te_indices) =
        nano::split3(samples, m_train_per, m_valid_per);
}
