#include "fixture/wlearner.h"
#include <nano/wlearner/dstep.h>

using namespace nano;

class fixture_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 2)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    static auto expected_fvalue() { return 2; }

    static auto expected_feature() { return 1; }

    static auto expected_tables() { return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), -1.42, 0.0); }

    void set_dstep_target(const tensor_size_t feature, const tensor_size_t fvalueX, const tensor4d_t& tables)
    {
        const auto hits    = this->hits();
        const auto samples = this->samples();
        const auto itarget = this->features();
        const auto classes = this->feature(feature).classes();
        const auto fvalues = make_random_tensor<int32_t>(make_dims(samples), tensor_size_t{0}, classes - 1);

        assert(2 == tables.size<0>());
        assert(fvalueX >= 0 && fvalueX < classes);

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            if (hits(sample, feature) != 0)
            {
                const auto fvalue = fvalues(sample);
                const auto target = tables(fvalue == fvalueX ? 0 : 1);

                set(sample, feature, fvalue);
                set(sample, itarget, target);
                assign(sample, fvalue == fvalueX ? 0 : 1);
            }
        }
    }

    static void check_wlearner(const dstep_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.fvalue(), expected_fvalue());
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        set_dstep_target(expected_feature(), expected_fvalue(), expected_tables());
    }
};

UTEST_BEGIN_MODULE(test_wlearner_dstep)

UTEST_CASE(fit_predict)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>();
    const auto datasourceX = make_random_datasource(make_features_all_continuous());

    check_wlearner<dstep_wlearner_t>(datasource0, datasourceX);
}

UTEST_END_MODULE()
