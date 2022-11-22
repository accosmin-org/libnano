#include "fixture/wlearner.h"
#include <nano/wlearner/table.h>

using namespace nano;

class sclass_fixture_datasource_t final : public wlearner_datasource_t
{
public:
    explicit sclass_fixture_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 3)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<sclass_fixture_datasource_t>(*this); }

    static auto expected_feature() { return 1; }

    static auto expected_features() { return make_indices(expected_feature()); }

    static auto expected_mhashes() { return mhashes_t{}; }

    static auto expected_tables() { return make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -1.42, +1.42, -0.42); }

    static auto make_wlearner() { return table_wlearner_t{}; }

    static void check_wlearner(const table_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
        UTEST_CHECK_EQUAL(wlearner.mhashes(), expected_mhashes());
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto feature = expected_feature();
        const auto tables  = expected_tables();
        const auto classes = this->feature(feature).classes();
        const auto fvalues = make_random_tensor<int32_t>(make_dims(this->samples()), tensor_size_t{0}, classes - 1);

        assert(classes == tables.size<0>());

        set_targets(feature, [&](const tensor_size_t sample) { return make_table_target(fvalues(sample), tables); });
    }
};

class mclass_fixture_datasource_t final : public wlearner_datasource_t
{
public:
    explicit mclass_fixture_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 8)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<mclass_fixture_datasource_t>(*this); }

    static auto expected_feature() { return 3; }

    static auto expected_features() { return make_indices(expected_feature()); }

    static auto expected_mhashes()
    {
        static const auto fvalues = make_tensor<int8_t, 2>(make_dims(8, 3), 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
                                                           1, 0, 1, 1, 1, 0, 1, 1, 1);

        static const auto mhashes = make_mhashes(fvalues);
        UTEST_CHECK_EQUAL(mhashes.size(), 8);
        return mhashes;
    }

    static auto expected_tables()
    {
        return make_tensor<scalar_t>(make_dims(8, 1, 1, 1), +2.42, -1.42, +1.42, -0.42, +0.42, +3.42, -2.42, +5.42);
    }

    static auto make_wlearner() { return table_wlearner_t{}; }

    static void check_wlearner(const table_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
        UTEST_CHECK_EQUAL(wlearner.mhashes(), expected_mhashes());
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto feature = expected_feature();
        const auto mhashes = expected_mhashes();
        const auto tables  = expected_tables();
        const auto classes = this->feature(feature).classes();
        const auto fvalues = make_random_tensor<int8_t, 2>(make_dims(this->samples(), classes), 0, 1);

        assert(mhashes.size() == tables.size<0>());

        set_targets(feature, [&](const tensor_size_t sample)
                    { return make_table_target(fvalues.tensor(sample), tables, mhashes); });
    }
};

UTEST_BEGIN_MODULE(test_wlearner_table)

UTEST_CASE(fit_predict_sclass)
{
    const auto datasource0 = make_datasource<sclass_fixture_datasource_t>(100);
    const auto datasourceX = make_random_datasource(make_features_all_continuous());

    check_wlearner(datasource0, datasourceX);
}

UTEST_CASE(fit_predict_mclass)
{
    const auto datasource0 = make_datasource<mclass_fixture_datasource_t>(200);
    const auto datasourceX = make_random_datasource(make_features_all_continuous());

    check_wlearner(datasource0, datasourceX);
}

UTEST_END_MODULE()
