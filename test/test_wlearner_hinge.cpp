#include "fixture/wlearner.h"
#include <nano/wlearner/hinge.h>

using namespace nano;

class fixture_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples, const hinge_type hinge)
        : wlearner_datasource_t(samples, 1)
        , m_hinge(hinge)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    static auto expected_feature() { return 6; }

    static auto expected_features() { return make_indices(expected_feature()); }

    static auto expected_threshold() { return 2.5; }

    static auto expected_beta() { return -1.1; }

    static auto expected_tables()
    {
        return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), expected_beta(), -expected_threshold() * expected_beta());
    }

    void set_hinge_target(const tensor_size_t feature, const scalar_t threshold, const scalar_t beta)
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
                const auto target = (m_hinge == hinge_type::left)
                                      ? ((fvalue < threshold) ? (beta * (fvalue - threshold)) : 0.0)
                                      : ((fvalue < threshold) ? 0.0 : (beta * (fvalue - threshold)));

                set(sample, feature, fvalue);
                set(sample, itarget, target);
                if ((m_hinge == hinge_type::left && fvalue < threshold) ||
                    (m_hinge == hinge_type::right && fvalue >= threshold))
                {
                    assign(sample, 0);
                }
            }
        }
    }

    void check_wlearner(const hinge_wlearner_t& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.hinge(), m_hinge);
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
        UTEST_CHECK_CLOSE(wlearner.threshold(), expected_threshold(), 1e-8);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        set_hinge_target(expected_feature(), expected_threshold(), expected_beta());
    }

    hinge_type m_hinge{hinge_type::left};
};

UTEST_BEGIN_MODULE(test_wlearner_hinge)

UTEST_CASE(str_enum)
{
    {
        std::ostringstream stream;
        stream << hinge_type::left;
        UTEST_CHECK_EQUAL(stream.str(), "left");
    }
    {
        std::ostringstream stream;
        stream << hinge_type::right;
        UTEST_CHECK_EQUAL(stream.str(), "right");
    }
}

UTEST_CASE(fit_predict_left)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>(300, hinge_type::left);
    const auto datasourceX = make_random_datasource(make_features_all_discrete());

    check_wlearner<hinge_wlearner_t>(datasource0, datasourceX);
}

UTEST_CASE(fit_predict_right)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>(300, hinge_type::right);
    const auto datasourceX = make_random_datasource(make_features_all_discrete());

    check_wlearner<hinge_wlearner_t>(datasource0, datasourceX);
}

UTEST_END_MODULE()
