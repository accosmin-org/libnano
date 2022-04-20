#include <sstream>
#include <nano/model.h>
#include <utest/utest.h>
#include "fixture/generator.h"
#include "fixture/generator_dataset.h"
#include <nano/generator/elemwise_identity.h>

using namespace nano;

static auto make_predictions(const dataset_generator_t& dataset, const indices_t& samples)
{
    return make_full_tensor<scalar_t>(cat_dims(samples.size(), dataset.target_dims()), 0.0);
}

static auto make_generator(const dataset_t& dataset)
{
    auto generator = dataset_generator_t{dataset};
    add_generator<elemwise_generator_t<sclass_identity_t>>(generator);
    add_generator<elemwise_generator_t<mclass_identity_t>>(generator);
    add_generator<elemwise_generator_t<scalar_identity_t>>(generator);
    add_generator<elemwise_generator_t<struct_identity_t>>(generator);
    return generator;
}

class fixture_model_t final : public model_t
{
public:

    rmodel_t clone() const override
    {
        return std::make_unique<fixture_model_t>(*this);
    }

    fit_result_t do_fit(const dataset_generator_t&, const indices_t&, const loss_t&, const solver_t&) override
    {
        return fit_result_t{};
    }

    tensor4d_t do_predict(const dataset_generator_t& dataset, const indices_t& samples) const override
    {
        return make_predictions(dataset, samples);
    }
};

static auto check_stream(const fixture_model_t& model)
{
    string_t str;
    {
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(write(stream, model));
        str = stream.str();
    }
    {
        fixture_model_t xmodel;
        std::istringstream stream(str);
        UTEST_CHECK_NOTHROW(read(stream, xmodel));
        UTEST_CHECK_EQUAL(xmodel.parameters(), model.parameters());
        return xmodel;
    }
}

static auto check_fit(
    const dataset_generator_t& dataset, const indices_t& samples, const loss_t& loss, const solver_t& solver)
{
    auto model = fixture_model_t{};
    UTEST_CHECK_NOTHROW(model.fit(dataset, samples, loss, solver));
    return model;
}

static void check_predict(const model_t& model, const dataset_generator_t& dataset, const indices_t& samples)
{
    const auto expected_predictions = make_predictions(dataset, samples);
    UTEST_CHECK_EQUAL(model.predict(dataset, samples), expected_predictions);
}

static void check_predict_fails(const model_t& model, const dataset_generator_t& dataset, const indices_t& samples)
{
    UTEST_CHECK_THROW(model.predict(dataset, samples), std::runtime_error);
}

UTEST_BEGIN_MODULE(test_model)

UTEST_CASE(parameters)
{
    const auto model = fixture_model_t{};

    UTEST_CHECK_EQUAL(model.parameter("model::folds").value<int>(), 5);
    UTEST_CHECK_EQUAL(model.parameter("model::random_seed").value<int>(), 42);
}

UTEST_CASE(fit_predict)
{
    const auto rloss = loss_t::all().get("squared");
    const auto rsolver = solver_t::all().get("lbfgs");

    const auto train_samples = arange(0, 80);
    const auto valid_samples = arange(80, 100);

    const auto dataset1 = make_dataset(100, 0U);
    const auto dataset2 = make_dataset(100, 3U);
    const auto dataset3 = make_dataset(100, 8U);

    const auto gdataset1 = make_generator(dataset1);
    const auto gdataset2 = make_generator(dataset2);
    const auto gdataset3 = make_generator(dataset3);

    {
        const auto model = check_stream(fixture_model_t{});

        check_predict_fails(model, gdataset1, train_samples);
        check_predict_fails(model, gdataset2, train_samples);
        check_predict_fails(model, gdataset3, train_samples);
    }
    {
        const auto model = check_stream(check_fit(gdataset1, train_samples, *rloss, *rsolver));

        check_predict(model, gdataset1, train_samples);
        check_predict(model, gdataset1, valid_samples);

        check_predict_fails(model, gdataset2, train_samples);
        check_predict_fails(model, gdataset3, train_samples);
    }
    {
        const auto model = check_stream(check_fit(gdataset2, train_samples, *rloss, *rsolver));

        check_predict(model, gdataset2, train_samples);
        check_predict(model, gdataset2, valid_samples);

        check_predict_fails(model, gdataset1, train_samples);
        check_predict_fails(model, gdataset3, train_samples);
    }
    {
        const auto model = check_stream(check_fit(gdataset3, train_samples, *rloss, *rsolver));

        check_predict(model, gdataset3, train_samples);
        check_predict(model, gdataset3, valid_samples);

        check_predict_fails(model, gdataset1, train_samples);
        check_predict_fails(model, gdataset2, train_samples);
    }
}

UTEST_END_MODULE()
