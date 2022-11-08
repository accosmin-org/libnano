#include "fixture/wlearner.h"
#include <nano/wlearner/affine.h>

using namespace nano;

class fixture_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 1)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    static auto expected_feature() { return 6; }

    static auto expected_features() { return make_indices(expected_feature()); }

    static auto expected_weight() { return +1.42; }

    static auto expected_bias() { return -0.573; }

    static auto expected_tables()
    {
        return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), expected_weight(), expected_bias());
    }

    void set_affine_target(const tensor_size_t feature, const scalar_t weight, const scalar_t bias)
    {
        const auto fvalues = make_random_tensor<scalar_t>(make_dims(this->samples()), -1.0, +0.8);

        set_targets(feature,
                    [&](const tensor_size_t sample)
                    {
                        const auto fvalue = fvalues(sample);
                        const auto target = weight * fvalue + bias;
                        return std::make_tuple(fvalue, target, 0);
                    });
    }

    static void check_wlearner(const affine_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        set_affine_target(expected_feature(), expected_weight(), expected_bias());
    }
};

UTEST_BEGIN_MODULE(test_wlearner_affine)

UTEST_CASE(fit_predict)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>(300);
    const auto datasourceX = make_random_datasource(make_features_all_discrete());

    check_wlearner<affine_wlearner_t>(datasource0, datasourceX);
}

UTEST_END_MODULE()
