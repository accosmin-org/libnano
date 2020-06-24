#include <nano/dataset/tabular_bank_marketing.h>

using namespace nano;

bank_marketing_dataset_t::bank_marketing_dataset_t()
{
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

    const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/bank-marketing");
    csvs(
    {
        csv_t{dir + "/bank-additional-full.csv"}.delim(";\"\r").header(true).expected(41188)
    });
}

split_t bank_marketing_dataset_t::make_split() const
{
    assert(samples() == 41188);

    return {
        nano::split3(samples(), train_percentage(), (100 - train_percentage()) / 2)
    };
}
