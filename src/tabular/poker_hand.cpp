#include <nano/tabular/poker_hand.h>

using namespace nano;

poker_hand_dataset_t::poker_hand_dataset_t()
{
    features(
    {
        feature_t{"S1"}.labels({"1", "2", "3", "4"}),
        feature_t{"C1"}.labels({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t{"S2"}.labels({"1", "2", "3", "4"}),
        feature_t{"C2"}.labels({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t{"S3"}.labels({"1", "2", "3", "4"}),
        feature_t{"C3"}.labels({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t{"S4"}.labels({"1", "2", "3", "4"}),
        feature_t{"C4"}.labels({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t{"S5"}.labels({"1", "2", "3", "4"}),
        feature_t{"C5"}.labels({"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"}),
        feature_t{"CLASS"}.labels({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
    }, 10);

    const auto dir = strcat(std::getenv("HOME"), "/libnano/datasets/poker-hand");
    csvs(
    {
        csv_t{dir + "/poker-hand-training-true.data"}.delim(",\r").header(false).expected(25010),
        csv_t{dir + "/poker-hand-testing.data"}.delim(",\r").header(false).expected(1000000)
    });
}

split_t poker_hand_dataset_t::make_split() const
{
    const auto tr_vd_size = 25010, te_size = 1000000;
    assert(samples() == tr_vd_size + te_size);

    return {
        nano::split2(tr_vd_size, train_percentage()),
        indices_t::LinSpaced(te_size, tr_vd_size, tr_vd_size + te_size)
    };
}
