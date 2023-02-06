#include "fixture/solver.h"
#include "fixture/splitter.h"
#include "fixture/tuner.h"
#include "fixture/wlearner.h"
#include <nano/gboost/model.h>

using namespace nano;

static auto make_gbooster()
{
    auto model = gboost_model_t{};
    model.logger(model_t::make_logger_stdio());
    model.parameter("model::gboost::epsilon")  = 1e-8;
    model.parameter("model::gboost::patience") = 3;
    return model;
}

static void check_predict(const gboost_model_t& model, const dataset_t& dataset, const scalar_t epsilon = 1e-12)
{
    const auto samples = arange(0, dataset.samples());
    const auto outputs = model.predict(dataset, samples);

    targets_iterator_t iterator(dataset, samples);
    iterator.loop([&](const tensor_range_t& range, size_t, tensor4d_cmap_t targets)
                  { UTEST_CHECK_CLOSE(targets, outputs.slice(range), epsilon); });
}

static void check_predict_throws(const gboost_model_t& model)
{
    const auto datasource1 = make_random_datasource(make_features_too_few());
    const auto datasource2 = make_random_datasource(make_features_too_many());
    const auto datasource3 = make_random_datasource(make_features_invalid_target());

    const auto dataset1 = make_dataset(datasource1);
    const auto dataset2 = make_dataset(datasource2);
    const auto dataset3 = make_dataset(datasource3);

    UTEST_CHECK_THROW(model.predict(dataset1, arange(0, dataset1.samples())), std::runtime_error);
    UTEST_CHECK_THROW(model.predict(dataset2, arange(0, dataset2.samples())), std::runtime_error);
    UTEST_CHECK_THROW(model.predict(dataset3, arange(0, dataset3.samples())), std::runtime_error);
}

template <typename tdatasource>
auto check_gbooster(gboost_model_t model, const tdatasource& datasource0)
{
    const auto loss     = make_loss("mse");
    const auto solver   = make_solver("lbfgs", 1e-12);
    const auto dataset  = make_dataset(datasource0);
    const auto splitter = make_splitter("k-fold", 2, 42U);
    const auto tuner    = make_tuner("surrogate");
    const auto samples  = arange(0, dataset.samples());

    // fitting should fail if no weak learner to chose from
    UTEST_REQUIRE_THROW(make_gbooster().fit(dataset, samples, *loss, *solver, *splitter, *tuner), std::runtime_error);

    // fitting should work when properly setup
    auto fit_result = fit_result_t{};
    UTEST_REQUIRE_NOTHROW(fit_result = model.fit(dataset, samples, *loss, *solver, *splitter, *tuner));

    // check model
    datasource0.check_gbooster(model);
    check_predict(model, dataset, 1e-5);
    check_predict_throws(model);

    // check model loading and saving from and to binary streams
    const auto imodel = check_stream(model);
    datasource0.check_gbooster(model);
    check_predict(imodel, dataset, 1e-5);
    check_predict_throws(imodel);

    return fit_result;
}
