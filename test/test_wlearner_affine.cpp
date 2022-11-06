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

    static auto expected_tables() { return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), +1.42, -0.573); }

    void set_affine_target(const tensor_size_t feature, const tensor4d_t& tables)
    {
        const auto hits    = this->hits();
        const auto samples = this->samples();
        const auto itarget = this->features();
        const auto fvalues = make_random_tensor<scalar_t>(make_dims(samples), -1.0, +0.8);

        assert(2 == tables.size<0>());

        const auto weight = tables(0);
        const auto bias   = tables(1);

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            if (hits(sample, feature) != 0)
            {
                const auto fvalue = fvalues(sample);
                const auto target = weight * fvalue + bias;

                set(sample, feature, fvalue);
                set(sample, itarget, target);
                assign(sample, 0);
            }
        }
    }

    static void check_wlearner(const affine_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        set_affine_target(expected_feature(), expected_tables());
    }
};

UTEST_BEGIN_MODULE(test_wlearner_affine)

UTEST_CASE(fit_predict)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>();
    const auto datasourceX = make_random_datasource(make_features_all_discrete());

    check_wlearner<affine_wlearner_t>(datasource0, datasourceX);
}

UTEST_END_MODULE()
