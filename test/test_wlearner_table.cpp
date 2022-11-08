#include "fixture/wlearner.h"
#include <nano/wlearner/table.h>

using namespace nano;

class fixture_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 3)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    static auto expected_feature() { return 1; }

    static auto expected_features() { return make_indices(expected_feature()); }

    static auto expected_tables() { return make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -1.42, +1.42, -0.42); }

    void set_table_target(const tensor_size_t feature, const tensor4d_t& tables)
    {
        const auto classes = this->feature(feature).classes();
        const auto fvalues = make_random_tensor<int32_t>(make_dims(this->samples()), tensor_size_t{0}, classes - 1);

        assert(classes == tables.size<0>());

        set_targets(feature,
                    [&](const tensor_size_t sample)
                    {
                        const auto fvalue = fvalues(sample);
                        const auto target = tables.tensor(fvalue);
                        return std::make_tuple(fvalue, target, fvalue);
                    });
    }

    static void check_wlearner(const table_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        set_table_target(expected_feature(), expected_tables());
    }
};

UTEST_BEGIN_MODULE(test_wlearner_table)

UTEST_CASE(fit_predict)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>(300);
    const auto datasourceX = make_random_datasource(make_features_all_continuous());

    check_wlearner<table_wlearner_t>(datasource0, datasourceX);
}

UTEST_END_MODULE()
