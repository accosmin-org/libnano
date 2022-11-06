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

    static auto expected_threshold() { return 2.5; }

    static auto expected_pred_lower() { return +3.0; }

    static auto expected_pred_upper() { return -2.1; }

    static auto expected_tables()
    {
        return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), expected_pred_lower(), expected_pred_upper());
    }

    void set_stump_target(const tensor_size_t feature, const scalar_t threshold, const scalar_t pred_lower,
                          const scalar_t pred_upper)
    {
        const auto hits    = this->hits();
        const auto samples = this->samples();
        const auto itarget = this->features();
        const auto fvalues = make_random_tensor<int32_t>(make_dims(samples), -5, +4);

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            if (hits(sample, feature) != 0)
            {
                const auto fvalue = fvalues(sample);
                const auto target = fvalue <= threshold ? pred_lower : pred_upper;

                set(sample, feature, fvalue);
                set(sample, itarget, target);
                assign(sample, fvalue <= threshold ? 0 : 1);
            }
        }
    }

    static void check_wlearner(const stump_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
        UTEST_CHECK_CLOSE(wlearner.threshold(), expected_threshold(), 1e-8);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        set_stump_target(expected_feature(), expected_threshold(), expected_pred_lower(), expected_pred_upper());
    }
};

UTEST_BEGIN_MODULE(test_wlearner_stump)

UTEST_CASE(fit_predict)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>();
    const auto datasourceX = make_random_datasource(make_features_all_discrete());

    check_wlearner<stump_wlearner_t>(datasource0, datasourceX);
}

UTEST_END_MODULE()
