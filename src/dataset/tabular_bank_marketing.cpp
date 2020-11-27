#include <nano/dataset/tabular_bank_marketing.h>

using namespace nano;

bank_marketing_dataset_t::bank_marketing_dataset_t()
{
    features(
    {
        feature_t{"age"},
        feature_t{"job"}.labels(12),
        feature_t{"marital"}.labels(4),
        feature_t{"education"}.labels(8),
        feature_t{"default"}.labels(3),
        feature_t{"housing"}.labels(3),
        feature_t{"loan"}.labels(3),
        feature_t{"contact"}.labels(2),
        feature_t{"month"}.labels(12),
        feature_t{"day_of_week"}.labels(5),
        feature_t{"duration"},
        feature_t{"campaign"},
        feature_t{"pdays"},
        feature_t{"previous"},
        feature_t{"poutcome"}.labels(3),
        feature_t{"emp.var.rate"},
        feature_t{"cons.price.idx"},
        feature_t{"cons.conf.idx"},
        feature_t{"euribor3m"},
        feature_t{"nr.employed"},
        feature_t{"y"}.labels(2),
    }, 20);

    const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/bank-marketing");
    csvs(
    {
        csv_t{dir + "/bank-additional-full.csv"}.delim(";\"\r").header(true).expected(41188)
    });
}
