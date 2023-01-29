#include "fixture/linear.h"
#include "fixture/loss.h"
#include "fixture/solver.h"
#include "fixture/splitter.h"
#include "fixture/tuner.h"
#include <nano/linear/enums.h>

using namespace nano;

using split_type = fit_result_t::split_type;
using value_type = fit_result_t::value_type;

static void check_outputs(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& outputs,
                          scalar_t epsilon)
{
    auto iterator = flatten_iterator_t{dataset, samples, 1U};
    iterator.batch(7);
    iterator.scaling(scaling_type::none);
    iterator.loop([&](tensor_range_t range, size_t, tensor4d_cmap_t targets)
                  { UTEST_CHECK_CLOSE(targets, outputs.slice(range), epsilon); });
}

static auto make_smooth_solver()
{
    auto solver                            = make_solver("lbfgs");
    solver->parameter("solver::max_evals") = 1000;
    solver->parameter("solver::epsilon")   = 1e-10;
    solver->lsearchk("cgdescent");
    return solver;
}

static auto make_nonsmooth_solver()
{
    auto solver                            = make_solver("osga");
    solver->parameter("solver::max_evals") = 2000;
    solver->parameter("solver::epsilon")   = 1e-6;
    return solver;
}

static auto make_model()
{
    auto model                              = linear_model_t{};
    model.parameter("model::linear::batch") = 10;
    model.logger(model_t::make_logger_stdio());
    return model;
}

static void check_result(const fit_result_t& result, const strings_t& expected_param_names,
                         const size_t min_param_results_size, const scalar_t epsilon)
{
    const auto& param_names        = result.param_names();
    const auto& param_results      = result.param_results();
    const auto  optim_errors_stats = result.stats(value_type::errors);
    const auto  optim_losses_stats = result.stats(value_type::losses);

    UTEST_CHECK_EQUAL(param_names, expected_param_names);
    UTEST_CHECK_CLOSE(optim_errors_stats.m_mean, 0.0, epsilon);
    UTEST_CHECK_CLOSE(optim_losses_stats.m_mean, 0.0, epsilon);

    UTEST_REQUIRE_GREATER_EQUAL(param_results.size(), min_param_results_size);

    const auto opt_losses = make_full_tensor<scalar_t>(make_dims(2), 0.0);
    const auto opt_errors = make_full_tensor<scalar_t>(make_dims(2), 0.0);

    tensor_size_t hits = 0;
    for (const auto& param_result : param_results)
    {
        const auto& params = param_result.params();
        UTEST_CHECK_EQUAL(params.size(), static_cast<tensor_size_t>(expected_param_names.size()));
        if (params.size() > 0)
        {
            UTEST_CHECK_GREATER(params.min(), 0.0);
        }

        const auto folds = param_result.folds();

        tensor1d_t train_losses(folds), train_errors(folds);
        tensor1d_t valid_losses(folds), valid_errors(folds);

        for (tensor_size_t fold = 0; fold < folds; ++fold)
        {
            train_losses(fold) = param_result.stats(fold, split_type::train, value_type::losses).m_mean;
            train_errors(fold) = param_result.stats(fold, split_type::train, value_type::errors).m_mean;
            valid_losses(fold) = param_result.stats(fold, split_type::valid, value_type::losses).m_mean;
            valid_errors(fold) = param_result.stats(fold, split_type::valid, value_type::errors).m_mean;
        }

        if (close(train_errors, opt_errors, epsilon))
        {
            ++hits;
            UTEST_CHECK_CLOSE(train_losses, opt_losses, 1.0 * epsilon);
            UTEST_CHECK_CLOSE(train_errors, opt_errors, 1.0 * epsilon);
            UTEST_CHECK_CLOSE(valid_losses, opt_losses, 5.0 * epsilon);
            UTEST_CHECK_CLOSE(valid_errors, opt_errors, 5.0 * epsilon);
        }
    }

    UTEST_CHECK_GREATER(hits, 0);
}

static void check_model(const linear_model_t& model, const dataset_t& dataset, const indices_t& samples,
                        const scalar_t epsilon)
{
    const auto outputs = model.predict(dataset, samples);
    check_outputs(dataset, samples, outputs, epsilon);

    string_t str;
    {
        std::ostringstream stream;
        UTEST_REQUIRE_NOTHROW(model.write(stream));
        str = stream.str();
    }
    {
        auto               new_model = linear_model_t{};
        std::istringstream stream(str);
        UTEST_REQUIRE_NOTHROW(new_model.read(stream));
        const auto new_outputs = model.predict(dataset, samples);
        UTEST_CHECK_CLOSE(outputs, new_outputs, epsilon0<scalar_t>());
    }
}

UTEST_BEGIN_MODULE(test_linear_model)

UTEST_CASE(regularization_none)
{
    const auto datasource = make_linear_datasource(100, 1, 4);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto model                                       = make_model();
    model.parameter("model::linear::scaling")        = scaling_type::none;
    model.parameter("model::linear::regularization") = linear::regularization_type::none;

    const auto param_names = strings_t{};
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss     = make_loss(loss_id);
        const auto solver   = string_t(loss_id) == "mse" ? make_smooth_solver() : make_nonsmooth_solver();
        const auto splitter = make_splitter("k-fold", 2);
        const auto tuner    = make_tuner();
        const auto result   = model.fit(dataset, samples, *loss, *solver, *splitter, *tuner);
        const auto epsilon  = string_t{loss_id} == "mse" ? 1e-6 : 1e-3;

        check_result(result, param_names, 0U, epsilon);
        check_model(model, dataset, samples, epsilon);
    }
}

UTEST_CASE(regularization_lasso)
{
    const auto datasource = make_linear_datasource(100, 1, 4);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto model                                       = make_model();
    model.parameter("model::linear::scaling")        = scaling_type::standard;
    model.parameter("model::linear::regularization") = linear::regularization_type::lasso;

    const auto param_names = strings_t{"l1reg"};
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss     = make_loss(loss_id);
        const auto solver   = make_nonsmooth_solver();
        const auto splitter = make_splitter("k-fold", 2);
        const auto tuner    = make_tuner();
        const auto result   = model.fit(dataset, samples, *loss, *solver, *splitter, *tuner);
        const auto epsilon  = 1e-3;

        check_result(result, param_names, 6U, epsilon);
        check_model(model, dataset, samples, epsilon);
    }
}

UTEST_CASE(regularization_ridge)
{
    const auto datasource = make_linear_datasource(100, 1, 4);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto model                                       = make_model();
    model.parameter("model::linear::scaling")        = scaling_type::mean;
    model.parameter("model::linear::regularization") = linear::regularization_type::ridge;

    const auto param_names = strings_t{"l2reg"};
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss     = make_loss(loss_id);
        const auto solver   = string_t(loss_id) == "mse" ? make_smooth_solver() : make_nonsmooth_solver();
        const auto splitter = make_splitter("k-fold", 2);
        const auto tuner    = make_tuner();
        const auto result   = model.fit(dataset, samples, *loss, *solver, *splitter, *tuner);
        const auto epsilon  = string_t{loss_id} == "mse" ? 1e-6 : 1e-3;

        check_result(result, param_names, 6U, epsilon);
        check_model(model, dataset, samples, epsilon);
    }
}

UTEST_CASE(regularization_variance)
{
    const auto datasource = make_linear_datasource(100, 1, 4);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto model                                       = make_model();
    model.parameter("model::linear::scaling")        = scaling_type::minmax;
    model.parameter("model::linear::regularization") = linear::regularization_type::variance;

    const auto param_names = strings_t{"vAreg"};
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss     = make_loss(loss_id);
        const auto solver   = string_t(loss_id) == "mse" ? make_smooth_solver() : make_nonsmooth_solver();
        const auto splitter = make_splitter("k-fold", 2);
        const auto tuner    = make_tuner();
        const auto result   = model.fit(dataset, samples, *loss, *solver, *splitter, *tuner);
        const auto epsilon  = string_t{loss_id} == "mse" ? 1e-6 : 1e-3;

        check_result(result, param_names, 6U, epsilon);
        check_model(model, dataset, samples, epsilon);
    }
}

UTEST_CASE(regularization_elasticnet)
{
    const auto datasource = make_linear_datasource(100, 1, 4);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto model                                       = make_model();
    model.parameter("model::linear::scaling")        = scaling_type::minmax;
    model.parameter("model::linear::regularization") = linear::regularization_type::elasticnet;

    const auto param_names = strings_t{"l1reg", "l2reg"};
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss     = make_loss(loss_id);
        const auto solver   = make_nonsmooth_solver();
        const auto splitter = make_splitter("k-fold", 2);
        const auto tuner    = make_tuner();
        const auto result   = model.fit(dataset, samples, *loss, *solver, *splitter, *tuner);
        const auto epsilon  = 1e-3;

        check_result(result, param_names, 15U, epsilon);
        check_model(model, dataset, samples, epsilon);
    }
}

UTEST_END_MODULE()
