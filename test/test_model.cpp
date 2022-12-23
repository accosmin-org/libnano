#include "fixture/dataset.h"
#include "fixture/datasource/hits.h"
#include "fixture/datasource/random.h"
#include "fixture/estimator.h"
#include "fixture/learner.h"
#include "fixture/loss.h"
#include "fixture/solver.h"
#include "fixture/splitter.h"
#include "fixture/tuner.h"

using namespace nano;

static auto make_predictions(const dataset_t& dataset, const indices_t& samples)
{
    return make_full_tensor<scalar_t>(cat_dims(samples.size(), dataset.target_dims()), samples.mean());
}

static auto make_features()
{
    return features_t{
        feature_t{"mclass"}.mclass(strings_t{"m00", "m01", "m02"}),
        feature_t{"sclass"}.sclass(strings_t{"s00", "s01", "s02"}),
        feature_t{"scalar"}.scalar(feature_type::int16),
        feature_t{"struct"}.scalar(feature_type::uint8, make_dims(1, 2, 2)),
    };
}

static auto make_datasource(const tensor_size_t samples, const size_t target)
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

    fit_result_t fit(const dataset_t& dataset, const indices_t&, const loss_t&, const solver_t&, const splitter_t&,
                     const tuner_t&) override
    {
        learner_t::fit(dataset);
        return fit_result_t{};
    }

    tensor4d_t predict(const dataset_t& dataset, const indices_t& samples) const override
    {
        learner_t::critical_compatible(dataset);
        return make_predictions(dataset, samples);
    }
};

UTEST_BEGIN_MODULE(test_model)

UTEST_CASE(fit_predict)
{
    const auto rloss     = make_loss("mse");
    const auto rsolver   = make_solver("lbfgs");
    const auto rsplitter = make_splitter("k-fold");
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

UTEST_END_MODULE()
