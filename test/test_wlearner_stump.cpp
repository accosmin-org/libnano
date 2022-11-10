#include "fixture/wlearner.h"
#include <nano/wlearner/stump.h>

using namespace nano;

class fixture_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 2)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    static auto expected_feature() { return 6; }

    static auto expected_features() { return make_indices(expected_feature()); }

    static auto expected_threshold() { return 2.5; }

    static auto expected_pred_lower() { return +3.0; }

    static auto expected_pred_upper() { return -2.1; }

    static auto expected_tables()
    {
        return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), expected_pred_lower(), expected_pred_upper());
    }

    static void check_wlearner(const stump_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
        UTEST_CHECK_CLOSE(wlearner.threshold(), expected_threshold(), 1e-8);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto feature    = expected_feature();
        const auto threshold  = expected_threshold();
        const auto pred_lower = expected_pred_lower();
        const auto pred_upper = expected_pred_upper();
        const auto fvalues    = make_random_tensor<int32_t>(make_dims(this->samples()), -5, +4);

        set_targets(feature, [&](const tensor_size_t sample)
                    { return make_stump_target(fvalues(sample), threshold, pred_lower, pred_upper); });
    }
};

UTEST_BEGIN_MODULE(test_wlearner_stump)

UTEST_CASE(fit_predict)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>(300);
    const auto datasourceX = make_random_datasource(make_features_all_discrete());

    check_wlearner<stump_wlearner_t>(datasource0, datasourceX);
}

UTEST_END_MODULE()
