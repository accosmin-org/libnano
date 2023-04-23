#include "fixture/configurable.h"
#include "fixture/dataset.h"
#include "fixture/datasource/hits.h"
#include "fixture/datasource/random.h"
#include "fixture/loss.h"
#include "fixture/model.h"
#include "fixture/solver.h"
#include "fixture/splitter.h"
#include "fixture/tuner.h"

using namespace nano;

namespace
{
void check_stats(const fit_result_t::stats_t& stats, const scalar_t expected_mean, const scalar_t expected_stdev,
                 const scalar_t expected_count, const scalar_t expected_per01, const scalar_t expected_per05,
                 const scalar_t expected_per10, const scalar_t expected_per20, const scalar_t expected_per50,
                 const scalar_t expected_per80, const scalar_t expected_per90, const scalar_t expected_per95,
                 const scalar_t expected_per99, const scalar_t epsilon = 1e-12)
{
    UTEST_CHECK_CLOSE(stats.m_mean, expected_mean, epsilon);
    UTEST_CHECK_CLOSE(stats.m_stdev, expected_stdev, epsilon);
    UTEST_CHECK_CLOSE(stats.m_count, expected_count, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per01, expected_per01, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per05, expected_per05, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per10, expected_per10, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per20, expected_per20, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per50, expected_per50, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per80, expected_per80, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per90, expected_per90, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per95, expected_per95, epsilon);
    UTEST_CHECK_CLOSE(stats.m_per99, expected_per99, epsilon);
}

auto make_predictions(const dataset_t& dataset, const indices_t& samples)
{
    return make_full_tensor<scalar_t>(cat_dims(samples.size(), dataset.target_dims()), samples.mean());
}

auto make_features()
{
    return features_t{
        feature_t{"mclass"}.mclass(strings_t{"m00", "m01", "m02"}),
        feature_t{"sclass"}.sclass(strings_t{"s00", "s01", "s02"}),
        feature_t{"scalar"}.scalar(feature_type::int16),
        feature_t{"struct"}.scalar(feature_type::uint8, make_dims(1, 2, 2)),
    };
}

auto make_datasource(const tensor_size_t samples, const size_t target)
{
    const auto features = make_features();
    const auto hits     = make_random_hits(samples, static_cast<tensor_size_t>(features.size()), target);

    auto datasource = random_datasource_t{samples, features, target, hits};
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_EQUAL(datasource.samples(), samples);
    return datasource;
}

class fixture_model_t final : public model_t
{
public:
    fixture_model_t()
        : model_t("fixture")
    {
    }

    rmodel_t clone() const override { return std::make_unique<fixture_model_t>(*this); }

    tensor4d_t predict(const dataset_t& dataset, const indices_t& samples) const override
    {
        learner_t::critical_compatible(dataset);
        return make_predictions(dataset, samples);
    }

    fit_result_t fit(const dataset_t& dataset, const indices_t&, const loss_t&, const solver_t&, const splitter_t&,
                     const tuner_t&) override
    {
        learner_t::fit_dataset(dataset);
        return fit_result_t{};
    }
};
} // namespace

UTEST_BEGIN_MODULE(test_model)

UTEST_CASE(factory)
{
    const auto& models = model_t::all();
    UTEST_CHECK_EQUAL(models.ids().size(), 2U);
    UTEST_CHECK(models.get("gboost") != nullptr);
    UTEST_CHECK(models.get("linear") != nullptr);
}

UTEST_CASE(fit_predict)
{
    const auto rloss     = make_loss("mse");
    const auto rsolver   = make_solver("lbfgs");
    const auto rsplitter = make_splitter("k-fold", 2);
    const auto rtuner    = make_tuner("surrogate");

    const auto train_samples = arange(0, 80);
    const auto valid_samples = arange(80, 100);

    const auto datasource1 = make_datasource(100, 0U);
    const auto datasource2 = make_datasource(100, 1U);
    const auto datasource3 = make_datasource(100, 2U);

    const auto dataset1 = make_dataset(datasource1);
    const auto dataset2 = make_dataset(datasource2);
    const auto dataset3 = make_dataset(datasource3);

    {
        const auto model = check_stream(fixture_model_t{});

        check_predict_fails(model, dataset1, train_samples);
        check_predict_fails(model, dataset2, train_samples);
        check_predict_fails(model, dataset3, train_samples);
    }
    {
        const auto model =
            check_stream(check_fit<fixture_model_t>(dataset1, train_samples, *rloss, *rsolver, *rsplitter, *rtuner));

        check_predict(model, dataset1, train_samples, make_predictions(dataset1, train_samples));
        check_predict(model, dataset1, valid_samples, make_predictions(dataset1, valid_samples));

        check_predict_fails(model, dataset2, train_samples);
        check_predict_fails(model, dataset3, train_samples);
    }
    {
        const auto model =
            check_stream(check_fit<fixture_model_t>(dataset2, train_samples, *rloss, *rsolver, *rsplitter, *rtuner));

        check_predict(model, dataset2, train_samples, make_predictions(dataset2, train_samples));
        check_predict(model, dataset2, valid_samples, make_predictions(dataset2, valid_samples));

        check_predict_fails(model, dataset1, train_samples);
        check_predict_fails(model, dataset3, train_samples);
    }
    {
        const auto model =
            check_stream(check_fit<fixture_model_t>(dataset3, train_samples, *rloss, *rsolver, *rsplitter, *rtuner));

        check_predict(model, dataset3, train_samples, make_predictions(dataset3, train_samples));
        check_predict(model, dataset3, valid_samples, make_predictions(dataset3, valid_samples));

        check_predict_fails(model, dataset1, train_samples);
        check_predict_fails(model, dataset2, train_samples);
    }
}

UTEST_CASE(fit_result_empty)
{
    const auto param_names = strings_t{};

    const auto result = fit_result_t{param_names};
    UTEST_CHECK_EQUAL(result.optimum().params(), tensor1d_t{});
    UTEST_CHECK_EQUAL(result.param_results().size(), 0U);
    UTEST_CHECK_EQUAL(result.param_names(), param_names);
}

UTEST_CASE(fit_result_optimum)
{
    const auto param_names = strings_t{"l1reg", "l2reg"};

    auto result = fit_result_t{param_names};
    UTEST_CHECK_EQUAL(result.param_results().size(), 0U);
    UTEST_CHECK_EQUAL(result.param_names(), param_names);

    const auto make_errors_losses = [](const tensor_size_t min, const tensor_size_t max)
    {
        auto values = tensor2d_t{2, max - min + 1};
        for (tensor_size_t val = min; val <= max; ++val)
        {
            values(0, val - min) = 1e-3 * static_cast<scalar_t>(val - min);
            values(1, val - min) = 1e-4 * static_cast<scalar_t>(max - val);
        }
        return values;
    };

    {
        const auto* const closest = result.closest(make_tensor<scalar_t>(make_dims(2), 0.0, 0.99));
        UTEST_REQUIRE(closest == nullptr);
    }
    {
        auto param = fit_result_t::param_t{make_tensor<scalar_t>(make_dims(2), 0.0, 1.0), 3};
        param.evaluate(0, make_errors_losses(0, 100), make_errors_losses(1000, 1200), 1);
        param.evaluate(1, make_errors_losses(1, 101), make_errors_losses(1001, 1301), "2");
        param.evaluate(2, make_errors_losses(2, 102), make_errors_losses(1003, 1403), 3.14);

        UTEST_CHECK_EQUAL(std::any_cast<int>(param.extra(0)), 1);
        UTEST_CHECK_EQUAL(std::any_cast<const char*>(param.extra(1)), string_t{"2"});
        UTEST_CHECK_EQUAL(std::any_cast<double>(param.extra(2)), 3.14);

        check_stats(param.stats(0, split_type::train, value_type::errors), 1e-3 * 50, 0.002915475947, 101.0, 1e-3, 5e-3,
                    10e-3, 20e-3, 50e-3, 80e-3, 90e-3, 95e-3, 99e-3);
        check_stats(param.stats(1, split_type::valid, value_type::losses), 1e-4 * 150, 0.000501663898, 301.0, 3e-4,
                    15e-4, 30e-4, 60e-4, 150e-4, 240e-4, 270e-4, 285e-4, 297e-4);

        result.add(std::move(param));
        result.evaluate(make_errors_losses(0, 10));

        check_stats(result.stats(value_type::errors), 1e-3 * 5, 1e-3, 11.0, 5e-4, 5e-4, 10e-4, 20e-4, 50e-4, 80e-4,
                    90e-4, 95e-4, 95e-4);
        check_stats(result.stats(value_type::losses), 1e-4 * 5, 1e-4, 11.0, 5e-5, 5e-5, 10e-5, 20e-5, 50e-5, 80e-5,
                    90e-5, 95e-5, 95e-5);

        const auto expected_optimum = make_tensor<scalar_t>(make_dims(2), 0.0, 1.0);
        UTEST_CHECK_CLOSE(result.optimum().params(), expected_optimum, 1e-12);
    }
    {
        const auto* const closest = result.closest(make_tensor<scalar_t>(make_dims(2), 0.0, 0.99));
        UTEST_REQUIRE(closest != nullptr);

        const auto expected_closest_params = make_tensor<scalar_t>(make_dims(2), 0.0, 1.0);
        UTEST_CHECK_CLOSE(closest->params(), expected_closest_params, 1e-12);
    }
    {
        auto param = fit_result_t::param_t{make_tensor<scalar_t>(make_dims(2), 1.0, 2.0), 3};
        param.evaluate(0, make_errors_losses(10, 110), make_errors_losses(1000, 1100));
        param.evaluate(1, make_errors_losses(11, 111), make_errors_losses(1001, 1201));
        param.evaluate(2, make_errors_losses(12, 112), make_errors_losses(1003, 1303));
        result.add(std::move(param));

        const auto expected_optimum = make_tensor<scalar_t>(make_dims(2), 1.0, 2.0);
        UTEST_CHECK_CLOSE(result.optimum().params(), expected_optimum, 1e-12);
    }
    {
        const auto* const closest = result.closest(make_tensor<scalar_t>(make_dims(2), 0.0, 0.99));
        UTEST_REQUIRE(closest != nullptr);

        const auto expected_closest_params = make_tensor<scalar_t>(make_dims(2), 0.0, 1.0);
        UTEST_CHECK_CLOSE(closest->params(), expected_closest_params, 1e-12);
    }
    {
        auto param = fit_result_t::param_t{make_tensor<scalar_t>(make_dims(2), 0.5, 1.2), 3};
        param.evaluate(0, make_errors_losses(10, 110), make_errors_losses(1000, 1010));
        param.evaluate(1, make_errors_losses(11, 111), make_errors_losses(1001, 1021));
        param.evaluate(2, make_errors_losses(12, 112), make_errors_losses(1003, 1033));
        result.add(std::move(param));

        const auto expected_optimum = make_tensor<scalar_t>(make_dims(2), 0.5, 1.2);
        UTEST_CHECK_CLOSE(result.optimum().params(), expected_optimum, 1e-12);
    }
    {
        const auto* const closest = result.closest(make_tensor<scalar_t>(make_dims(2), 0.5, 1.21));
        UTEST_REQUIRE(closest != nullptr);

        const auto expected_closest_params = make_tensor<scalar_t>(make_dims(2), 0.5, 1.2);
        UTEST_CHECK_CLOSE(closest->params(), expected_closest_params, 1e-12);
    }
    {
        auto param = fit_result_t::param_t{make_tensor<scalar_t>(make_dims(2), 0.9, 1.1), 3};
        param.evaluate(0, make_errors_losses(10, 110), make_errors_losses(1000, 1040));
        param.evaluate(1, make_errors_losses(11, 111), make_errors_losses(1001, 1061));
        param.evaluate(2, make_errors_losses(12, 112), make_errors_losses(1003, 1033));
        result.add(std::move(param));

        const auto expected_optimum = make_tensor<scalar_t>(make_dims(2), 0.5, 1.2);
        UTEST_CHECK_CLOSE(result.optimum().params(), expected_optimum, 1e-12);
    }
}

UTEST_END_MODULE()
