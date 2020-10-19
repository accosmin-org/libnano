#include <nano/dataset/tabular_adult.h>

using namespace nano;

adult_dataset_t::adult_dataset_t()
{
    features(
    {
        feature_t{"age"},
        feature_t{"workclass"}.placeholder("?").labels(
        {
            "Private",
            "Self-emp-not-inc", "Self-emp-inc",
            "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"
        }),
        feature_t{"fnlwgt"},
        feature_t{"education"}.labels(
        {
            "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc",
            "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
        }),
        feature_t{"education-num"},
        feature_t{"marital-status"}.labels(
        {
            "Married-civ-spouse", "Divorced",
            "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        }),
        feature_t{"occupation"}.placeholder("?").labels(
        {
            "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
            "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
            "Priv-house-serv", "Protective-serv", "Armed-Forces"
        }),
        feature_t{"relationship"}.labels(
        {
            "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
        }),
        feature_t{"race"}.labels(
        {
            "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
        }),
        feature_t{"sex"}.labels({"Female", "Male"}),
        feature_t{"capital-gain"},
        feature_t{"capital-loss"},
        feature_t{"hours-per-week"},
        feature_t{"native-country"}.placeholder("?").labels(
        {
            "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
            "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
            "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
            "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
            "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
            "Peru", "Hong", "Holand-Netherlands"
        }),
        feature_t{"income"}.labels({">50K", "<=50K"})
    }, 14);

    const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/adult");
    csvs(
    {
        csv_t{dir + "/adult.data"}.skip('|').delim(", .").header(false).expected(32561),
        csv_t{dir + "/adult.test"}.skip('|').delim(", .").header(false).expected(16281).testing(make_range(0, 16281))
    });
}
