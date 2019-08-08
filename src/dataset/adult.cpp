#include "adult.h"
#include <nano/mlearn.h>

using namespace nano;

adult_dataset_t::adult_dataset_t() :
    m_dir(std::getenv("HOME"))
{
    m_dir.append("/experiments/datasets/adult");

    features(
    {
        feature_t::make_scalar("age"),
        feature_t::make_discrete("workclass",
        {
            "Private",
            "Self-emp-not-inc", "Self-emp-inc",
            "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"
        }, "?"),
        feature_t::make_scalar("fnlwgt"),
        feature_t::make_discrete("education",
        {
            "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc",
            "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
        }),
        feature_t::make_scalar("education-num"),
        feature_t::make_discrete("marital-status",
        {
            "Married-civ-spouse", "Divorced",
            "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        }),
        feature_t::make_discrete("occupation",
        {
            "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
            "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
            "Priv-house-serv", "Protective-serv", "Armed-Forces"
        }, "?"),
        feature_t::make_discrete("relationship",
        {
            "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
        }),
        feature_t::make_discrete("race",
        {
            "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
        }),
        feature_t::make_discrete("sex", {"Female", "Male"}),
        feature_t::make_scalar("capital-gain"),
        feature_t::make_scalar("capital-loss"),
        feature_t::make_scalar("hours-per-week"),
        feature_t::make_discrete("native-country",
        {
            "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
            "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
            "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
            "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
            "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
            "Peru", "Hong", "Holand-Netherlands"
        }, "?"),
        feature_t::make_discrete("income", {">50K", "<=50K"})
    }, 14);

    config(config());
}

json_t adult_dataset_t::config() const
{
    json_t json;
    json["dir"] = m_dir;
    json["folds"] = strcat(m_folds, "[1,100]");
    json["train_per"] = strcat(m_train_per, "[10,90]");
    return json;
}

void adult_dataset_t::config(const json_t& json)
{
    from_json(json, "dir", m_dir);
    from_json_range(json, "folds", m_folds, 1, 100);
    from_json_range(json, "train_per", m_train_per, 10, 90);

    csvs(
    {
        csv_t{m_dir + "/adult.data"}.skip('|').delim(", .").header(false),
        csv_t{m_dir + "/adult.test"}.skip('|').delim(", .").header(false)
    });
    folds(m_folds);
}

void adult_dataset_t::split(const tensor_size_t samples, split_t& split) const
{
    const auto tr_vd_size = 32561;
    const auto te_size = 16281;

    if (samples != tr_vd_size + te_size)
    {
        throw std::invalid_argument(strcat("adult dataset: received ", samples, " samples, expecting ", tr_vd_size + te_size));
    }

    split.m_te_indices = indices_t::LinSpaced(te_size, tr_vd_size, tr_vd_size + te_size);
    std::tie(split.m_tr_indices, split.m_vd_indices) = nano::split2(tr_vd_size, m_train_per);
}
