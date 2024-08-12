#include "fixture/linear.h"
#include "fixture/loss.h"
#include "fixture/solver.h"
#include "fixture/splitter.h"
#include <nano/linear/util.h>

using namespace nano;
using namespace nano::ml;

namespace
{
auto make_smooth_solver()
{
    return make_solver("lbfgs");
}

auto make_nonsmooth_solver()
{
    return make_solver("rqb");
}

auto make_model(const string_t& id, const scaling_type scaling, const tensor_size_t batch = 100)
{
    auto model = linear_t::all().get(id);
    UTEST_REQUIRE(model);
    model->parameter("linear::batch")   = batch;
    model->parameter("linear::scaling") = scaling;
    return model;
}

auto make_fit_params(const rsolver_t& solver)
{
    return params_t{}.splitter(make_splitter("k-fold", 2)).solver(solver).logger(make_stdout_logger());
}

void check_outputs(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& outputs,
                   const scalar_t epsilon)
{
    auto iterator = flatten_iterator_t{dataset, samples};
    iterator.batch(7);
    iterator.scaling(scaling_type::none);
    iterator.loop([&](tensor_range_t range, size_t, tensor4d_cmap_t targets)
                  { UTEST_CHECK_CLOSE(targets, outputs.slice(range), epsilon); });
}

void check_model(const linear_t& model, const dataset_t& dataset, const indices_t& samples, const scalar_t epsilon)
{
    const auto outputs = model.predict(dataset, samples);
    check_outputs(dataset, samples, outputs, epsilon);

    UTEST_CHECK_EQUAL(model.weights().dims(), make_dims(1, dataset.columns()));
    UTEST_CHECK_EQUAL(model.bias().dims(), make_dims(1));

    string_t str;
    {
        std::ostringstream stream;
        UTEST_REQUIRE_NOTHROW(model.write(stream));
        str = stream.str();
    }
    {
        auto               new_model = make_model("ordinary", scaling_type::none);
        std::istringstream stream(str);
        UTEST_REQUIRE_NOTHROW(new_model->read(stream));
        const auto new_outputs = new_model->predict(dataset, samples);
        UTEST_CHECK_CLOSE(outputs, new_outputs, epsilon0<scalar_t>());
    }
}

void check_importance(const linear_t& model, const dataset_t& dataset, const indices_t& relevancy)
{
    const auto importance         = linear::feature_importance(dataset, model.weights());
    const auto sparsity           = linear::sparsity_ratio(importance);
    const auto expected_revelancy = static_cast<scalar_t>(relevancy.sum()) / static_cast<scalar_t>(dataset.features());

    UTEST_REQUIRE_EQUAL(relevancy.size(), dataset.features());
    UTEST_REQUIRE_EQUAL(relevancy.size(), importance.size());

    for (tensor_size_t feature = 0, features = dataset.features(); feature < features; ++feature)
    {
        if (relevancy(feature) != 0)
        {
            UTEST_CHECK_GREATER(importance(feature), 1e-1);
        }
        else
        {
            UTEST_CHECK_LESS(importance(feature), 1e-6);
        }
    }

    UTEST_CHECK_CLOSE(sparsity, 1.0 - expected_revelancy, 1e-15);
}
} // namespace

UTEST_BEGIN_MODULE(test_linear_model)

UTEST_CASE(ordinary)
{
    const auto datasource = make_linear_datasource(100, 1, 4, "datasource::linear::relevant", 70);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());
    const auto model      = make_model("ordinary", scaling_type::none, 21);

    const auto param_names = strings_t{};
    for (const auto& loss_id : strings_t{"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss       = make_loss(loss_id);
        const auto solver     = loss_id == "mse" ? make_smooth_solver() : make_nonsmooth_solver();
        const auto fit_params = make_fit_params(solver);
        const auto result     = model->fit(dataset, samples, *loss, fit_params);
        const auto epsilon    = loss_id == "mse" ? 1e-6 : 1e-4;

        check_result(result, param_names, 2, epsilon);
        check_model(*model, dataset, samples, epsilon);
        check_importance(*model, dataset, datasource.relevant_feature_mask());
    }
}

// FIXME: enable the test when an efficient optimizer with strong theoretical guarantees is available!
/*UTEST_CASE(lasso)
{
    const auto datasource = make_linear_datasource(100, 1, 4, "datasource::linear::relevant", 70);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());
    const auto model      = make_model("lasso", scaling_type::standard, 50);

    const auto param_names = strings_t{"l1reg"};
    for (const auto& loss_id : strings_t{"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss       = make_loss(loss_id);
        const auto solver     = make_nonsmooth_solver();
        const auto fit_params = make_fit_params(solver);
        const auto result     = model.fit(dataset, samples, *loss, fit_params);
        const auto epsilon    = 1e-4;

        check_result(result, param_names, 2, epsilon);
        check_model(model, dataset, samples, epsilon);
        check_importance(model, dataset, datasource.relevant_feature_mask());
    }
}*/

UTEST_CASE(ridge)
{
    const auto datasource = make_linear_datasource(100, 1, 4, "datasource::linear::relevant", 70);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());
    const auto model      = make_model("ridge", scaling_type::mean, 100);

    const auto param_names = strings_t{"l2reg"};
    for (const auto& loss_id : strings_t{"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss       = make_loss(loss_id);
        const auto solver     = loss_id == "mse" ? make_smooth_solver() : make_nonsmooth_solver();
        const auto fit_params = make_fit_params(solver);
        const auto result     = model->fit(dataset, samples, *loss, fit_params);
        const auto epsilon    = loss_id == "mse" ? 1e-6 : 1e-4;

        check_result(result, param_names, 2, epsilon);
        check_model(*model, dataset, samples, epsilon);
        check_importance(*model, dataset, datasource.relevant_feature_mask());
    }
}

// FIXME: enable the test when an efficient optimizer with strong theoretical guarantees is available!
/*UTEST_CASE(elasticnet)
{
    const auto datasource = make_linear_datasource(100, 1, 4, "datasource::linear::relevant", 70);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());
    const auto model      = make_model("elasticnet", scaling_type::minmax, 100);

    const auto param_names = strings_t{"l1reg", "l2reg"};
    for (const auto& loss_id : strings_t{"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss       = make_loss(loss_id);
        const auto solver     = make_nonsmooth_solver();
        const auto fit_params = make_fit_params(solver);
        const auto result     = model.fit(dataset, samples, *loss, fit_params);
        const auto epsilon    = 1e-4;

        check_result(result, param_names, 2, epsilon);
        check_model(model, dataset, samples, epsilon);
        check_importance(model, dataset, datasource.relevant_feature_mask());
    }
}*/

UTEST_END_MODULE()
