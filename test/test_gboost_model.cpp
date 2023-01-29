#include "fixture/gboost.h"
#include <nano/wlearner/affine.h>

using namespace nano;

class fixture_linear_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_linear_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 1)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_linear_datasource_t>(*this); }

    static auto expected_weight1() { return +0.5; }

    static auto expected_weight2() { return -0.1; }

    static auto expected_bias1() { return -0.3; }

    static auto expected_bias2() { return +0.7; }

    static auto expected_feature1() { return 5; }

    static auto expected_feature2() { return 7; }

    static void check_gbooster(const gboost_model_t& model)
    {
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

        const auto feature1 = expected_feature1();
        const auto feature2 = expected_feature2();

        const auto fvalues1 = make_random_tensor<scalar_t>(make_dims(this->samples()), -1.0, +0.8);
        const auto fvalues2 = make_random_tensor<scalar_t>(make_dims(this->samples()), +1.1, +2.4);

        const auto hits    = this->hits();
        const auto samples = this->samples();
        const auto itarget = this->features(); // NB: the last feature is the target!

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            const auto [fvalue1, target1, cluster1] =
                make_affine_target(fvalues1(sample), expected_weight1(), expected_bias1());

            const auto [fvalue2, target2, cluster2] =
                make_affine_target(fvalues2(sample), expected_weight2(), expected_bias2());

            set(sample, feature1, fvalue1);
            set(sample, feature2, fvalue2);
            set(sample, itarget, target1 + target2);
            assign(sample, cluster1);
        }
    }
};

// TODO: check with linear and table weak learners
// TODO: check fit results have the expected structure for various regularization methods

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

UTEST_CASE(fit_predict_linear)
{
    const auto datasource = make_datasource<fixture_linear_datasource_t>(100);

    auto model = make_gbooster();
    model.add("affine");
    model.add("dense-table");

    const auto fit_result = check_gbooster(std::move(model), datasource);
    // TODO check fit results - expecting perfect fitting!
}

/*UTEST_CASE(train_mixed)
{
    const auto loss    = make_loss();
    const auto solver  = make_solver();
    const auto dataset = make_dataset<gboost_mixed_dataset_t>(10, 1, 100);
    const auto samples = make_samples(dataset);

    auto wstump  = wlearner_stump_t{};
    auto wlinear = wlearner_lin1_t{};

    auto model = gboost_model_t{};
    UTEST_REQUIRE_NOTHROW(model.rounds(10));
    UTEST_REQUIRE_NOTHROW(model.epsilon(1e-8));
    UTEST_REQUIRE_NOTHROW(model.shrinkage(1.0));
    UTEST_REQUIRE_NOTHROW(model.subsample(1.0));
    UTEST_REQUIRE_NOTHROW(model.wscale(::nano::wscale::tboost));
    UTEST_REQUIRE_NOTHROW(model.add(wstump));
    UTEST_REQUIRE_NOTHROW(model.add(wlinear));

    UTEST_REQUIRE_NOTHROW(model.fit(*loss, dataset, samples, *solver));
    ::check_predict(dataset, model);
    ::check_features(dataset, *loss, model);
}*/

UTEST_END_MODULE()
