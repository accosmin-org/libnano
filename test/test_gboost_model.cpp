#include "fixture/gboost.h"
#include <nano/wlearner/affine.h>
#include <nano/wlearner/table.h>

using namespace nano;

class fixture_bias_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_bias_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 1)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_bias_datasource_t>(*this); }

    static auto expected_bias() { return -0.3; }

    static void check_gbooster(const gboost_model_t& model)
    {
        UTEST_CHECK_EQUAL(model.wlearners().size(), 0U);
        UTEST_CHECK_EQUAL(model.features(), indices_t{});
        UTEST_CHECK_CLOSE(model.bias()(0), expected_bias(), 1e-6);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto bias    = expected_bias();
        const auto samples = this->samples();
        const auto itarget = this->features(); // NB: the last feature is the target!

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            set(sample, itarget, bias);
        }
    }
};

class fixture_affine_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_affine_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 1)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_affine_datasource_t>(*this); }

    static auto expected_weight1() { return +0.5; }

    static auto expected_weight2() { return -0.1; }

    static auto expected_bias1() { return -0.3; }

    static auto expected_bias2() { return +0.7; }

    static auto expected_feature1() { return 5; }

    static auto expected_feature2() { return 7; }

    static void check_gbooster(const gboost_model_t& model)
    {
        UTEST_CHECK_EQUAL(model.wlearners().size(), 2U);
        UTEST_CHECK_EQUAL(model.features(), make_indices(expected_feature1(), expected_feature2()));

        scalar_t weight1 = 0.0, bias = model.bias()(0);
        scalar_t weight2 = 0.0;

        for (const auto& wlearner : model.wlearners())
        {
            UTEST_CHECK_EQUAL(wlearner->type_id(), "affine");

            const affine_wlearner_t* pwlearner = nullptr;
            UTEST_REQUIRE_NOTHROW(pwlearner = dynamic_cast<const affine_wlearner_t*>(wlearner.get()));

            if (pwlearner->feature() == expected_feature1())
            {
                weight1 += pwlearner->vector(0)(0);
            }
            else
            {
                weight2 += pwlearner->vector(0)(0);
            }
            bias += pwlearner->vector(1)(0);
        }

        UTEST_CHECK_CLOSE(weight1, expected_weight1(), 1e-5);
        UTEST_CHECK_CLOSE(weight2, expected_weight2(), 1e-5);
        UTEST_CHECK_CLOSE(bias, expected_bias1() + expected_bias2(), 1e-5);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto samples  = this->samples();
        const auto feature1 = expected_feature1();
        const auto feature2 = expected_feature2();
        const auto itarget  = this->features(); // NB: the last feature is the target!

        const auto fvalues1 = make_random_tensor<scalar_t>(make_dims(samples), -1.0, +0.8);
        const auto fvalues2 = make_random_tensor<scalar_t>(make_dims(samples), +1.1, +2.4);

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            const auto [fvalue1, target1, cluster1] =
                make_affine_target(fvalues1(sample), expected_weight1(), expected_bias1());

            const auto [fvalue2, target2, cluster2] =
                make_affine_target(fvalues2(sample), expected_weight2(), expected_bias2());

            set(sample, feature1, fvalue1);
            set(sample, feature2, fvalue2);
            set(sample, itarget, target1 + target2);
        }
    }
};

class fixture_tables_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_tables_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 1)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_tables_datasource_t>(*this); }

    static auto expected_tables1() { return make_tensor<scalar_t>(make_dims(3, 1, 1, 1), +0.5, -0.3, +0.9); }

    static auto expected_tables2() { return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), +2.5, -1.2); }

    static auto expected_feature1() { return 1; }

    static auto expected_feature2() { return 0; }

    static void check_gbooster(const gboost_model_t& model)
    {
        UTEST_CHECK_EQUAL(model.wlearners().size(), 2U);
        UTEST_CHECK_EQUAL(model.features(), make_indices(expected_feature2(), expected_feature1()));

        for (const auto& wlearner : model.wlearners())
        {
            UTEST_CHECK_EQUAL(wlearner->type_id(), "dense-table");
        }
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto samples  = this->samples();
        const auto feature1 = expected_feature1();
        const auto feature2 = expected_feature2();
        const auto itarget  = this->features(); // NB: the last feature is the target!

        const auto classes1 = this->feature(feature1).classes();
        const auto fvalues1 = make_random_tensor<int32_t>(make_dims(samples), tensor_size_t{0}, classes1 - 1);

        const auto classes2 = this->feature(feature2).classes();
        const auto fvalues2 = make_random_tensor<int32_t>(make_dims(samples), tensor_size_t{0}, classes2 - 1);

        const auto tables1 = expected_tables1();
        const auto tables2 = expected_tables2();

        assert(classes1 == tables1.size<0>());
        assert(classes2 == tables2.size<0>());

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            const auto [fvalue1, target1, cluster1] = make_table_target(fvalues1(sample), tables1);
            const auto [fvalue2, target2, cluster2] = make_table_target(fvalues2(sample), tables2);

            set(sample, feature1, fvalue1);
            set(sample, feature2, fvalue2);
            set(sample, itarget, target1(0) + target2(0));
        }
    }
};

template <typename... targs>
static auto make_gbooster_to_fit(const targs... args)
{
    auto model = make_gbooster();
    model.add("affine");
    model.add("dense-table");
    config(model, args...);
    return model;
}

// TODO: check that fitting twice with the same seed works as expected

UTEST_BEGIN_MODULE(test_gboost_model)

UTEST_CASE(empty)
{
    auto model = make_gbooster();

    UTEST_CHECK_EQUAL(model.bias().size(), 0);
    UTEST_CHECK_EQUAL(model.features().size(), 0);
    UTEST_CHECK_EQUAL(model.wlearners().size(), 0U);

    check_predict_throws(model);
}

UTEST_CASE(add_protos)
{
    auto model = make_gbooster();

    UTEST_CHECK_NOTHROW(model.add("affine"));
    UTEST_CHECK_NOTHROW(model.add(affine_wlearner_t{}));
    UTEST_CHECK_THROW(model.add("invalid"), std::runtime_error);

    check_predict_throws(model);
}

UTEST_CASE(fit_predict_bias)
{
    auto       model       = make_gbooster_to_fit();
    const auto param_names = strings_t{};
    const auto datasource  = make_datasource<fixture_bias_datasource_t>(100);

    const auto result = check_gbooster(std::move(model), datasource);
    check_result(result, param_names);
}

UTEST_CASE(fit_predict_affine)
{
    auto       model       = make_gbooster_to_fit();
    const auto param_names = strings_t{};
    const auto datasource  = make_datasource<fixture_affine_datasource_t>(200);

    const auto result = check_gbooster(std::move(model), datasource);
    check_result(result, param_names);
}

UTEST_CASE(fit_predict_tables)
{
    auto       model       = make_gbooster_to_fit();
    const auto param_names = strings_t{};
    const auto datasource  = make_datasource<fixture_tables_datasource_t>(300);

    const auto result = check_gbooster(std::move(model), datasource);
    check_result(result, param_names);
}

UTEST_CASE(fit_predict_bootstrap)
{
    auto       model       = make_gbooster_to_fit("gboost::bootstrap", "on");
    const auto param_names = strings_t{};
    const auto datasource  = make_datasource<fixture_affine_datasource_t>(300);

    const auto result = check_gbooster(std::move(model), datasource);
    check_result(result, param_names);
}

UTEST_CASE(fit_predict_tboost)
{
    auto       model       = make_gbooster_to_fit("gboost::wscale", "tboost");
    const auto param_names = strings_t{};
    const auto datasource  = make_datasource<fixture_tables_datasource_t>(400);

    const auto result = check_gbooster(std::move(model), datasource);
    check_result(result, param_names);
}

UTEST_CASE(tune_shrinkage)
{
    auto       model       = make_gbooster_to_fit("gboost::shrinkage", "on");
    const auto param_names = strings_t{"shrinkage"};
    const auto datasource  = make_datasource<fixture_affine_datasource_t>(300);

    const auto result = check_gbooster(std::move(model), datasource);
    check_result(result, param_names);
}

UTEST_CASE(tune_subsample)
{
    auto       model       = make_gbooster_to_fit("gboost::subsample", "on");
    const auto param_names = strings_t{"subsample"};
    const auto datasource  = make_datasource<fixture_affine_datasource_t>(300);

    const auto result = check_gbooster(std::move(model), datasource);
    check_result(result, param_names);
}

UTEST_CASE(tune_variance)
{
    auto model       = make_gbooster_to_fit("gboost::regularization", "variance");
    const auto param_names = strings_t{"vAreg"};
    const auto datasource  = make_datasource<fixture_affine_datasource_t>(300);

    const auto result = check_gbooster(std::move(model), datasource);
    check_result(result, param_names);
}

UTEST_END_MODULE()
