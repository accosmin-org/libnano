#include "fixture/loss.h"
#include "fixture/linear.h"
#include <nano/linear/model.h>
#include <nano/linear/regularization.h>

using namespace nano;

static auto make_smooth_solver()
{
    auto solver = solver_t::all().get("lbfgs");
    UTEST_REQUIRE(solver);
    solver->parameter("solver::max_evals") = 1000;
    solver->parameter("solver::epsilon") = 1e-12;
    solver->lsearchk("cgdescent");
    return solver;
}

static auto make_nonsmooth_solver()
{
    auto solver = solver_t::all().get("osga");
    UTEST_REQUIRE(solver);
    solver->parameter("solver::max_evals") = 5000;
    solver->parameter("solver::epsilon") = 1e-10;
    return solver;
}

static auto make_solver(const string_t& loss_id)
{
    return loss_id == "squared" ? make_smooth_solver() : make_nonsmooth_solver();
}

static auto make_model()
{
    auto model = linear_model_t{};

    model.logger([] (const fit_result_t& result, const string_t& prefix)
    {
        auto&& logger = log_info();
        logger << std::fixed << std::setprecision(9) << std::fixed << prefix << ": ";

        const auto print_params = [&] (const tensor1d_t& param_values)
        {
            assert(result.m_param_names.size() == static_cast<size_t>(param_values.size()));
            for (size_t i = 0U, size = result.m_param_names.size(); i < size; ++ i)
            {
                logger << result.m_param_names[i] << "=" << param_values(static_cast<tensor_size_t>(i)) << ",";
            }
        };

        if (std::isfinite(result.m_refit_error))
        {
            print_params(result.m_refit_params);
            logger << "refit=" << result.m_refit_value << "/" << result.m_refit_error << ".";
        }
        else if (!result.m_cv_results.empty())
        {
            const auto& cv_result = *result.m_cv_results.rbegin();
            print_params(cv_result.m_params);
            logger << "train=" << cv_result.m_train_values.mean() << "/" << cv_result.m_train_errors.mean() << ",";
            logger << "valid=" << cv_result.m_valid_values.mean() << "/" << cv_result.m_valid_errors.mean() << ".";
        }
    });

    return model;
}

static void check_result(const fit_result_t& result,
    const strings_t& expected_param_names, size_t min_cv_results_size, scalar_t epsilon)
{
    UTEST_CHECK_CLOSE(result.m_refit_value, 0.0, epsilon);
    UTEST_CHECK_CLOSE(result.m_refit_error, 0.0, epsilon);
    UTEST_CHECK_EQUAL(result.m_param_names, expected_param_names);

    UTEST_REQUIRE_GREATER_EQUAL(result.m_cv_results.size(), min_cv_results_size);

    const auto opt_values = make_full_tensor<scalar_t>(make_dims(2), 0.0);
    const auto opt_errors = make_full_tensor<scalar_t>(make_dims(2), 0.0);

    tensor_size_t hits = 0;
    for (const auto& cv_result : result.m_cv_results)
    {
        UTEST_CHECK_GREATER(cv_result.m_params.min(), 0.0);
        UTEST_CHECK_EQUAL(cv_result.m_params.size(), static_cast<tensor_size_t>(expected_param_names.size()));
        if (close(cv_result.m_train_errors, opt_errors, epsilon))
        {
            ++ hits;
            UTEST_CHECK_CLOSE(cv_result.m_train_values, opt_values, epsilon);
            UTEST_CHECK_CLOSE(cv_result.m_train_errors, opt_errors, epsilon);
            UTEST_CHECK_CLOSE(cv_result.m_valid_values, opt_values, epsilon);
            UTEST_CHECK_CLOSE(cv_result.m_valid_errors, opt_errors, epsilon);
        }
    }
    if (!expected_param_names.empty())
    {
        UTEST_CHECK_GREATER(hits, 0);
    }
    else
    {
        UTEST_CHECK(result.m_cv_results.empty());
    }
}

static void check_model(const linear_model_t& model,
    const dataset_generator_t& dataset, const indices_t& samples, scalar_t epsilon)
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
        auto new_model = linear_model_t{};
        std::istringstream stream(str);
        UTEST_REQUIRE_NOTHROW(new_model.read(stream));
        const auto new_outputs = model.predict(dataset, samples);
        UTEST_CHECK_CLOSE(outputs, new_outputs, epsilon0<scalar_t>());
    }
}

UTEST_BEGIN_MODULE(test_linear_model)

UTEST_CASE(regularization_none)
{
    const auto dataset = make_dataset(100, 1, 4);
    const auto generator = make_generator(dataset);
    const auto samples = arange(0, dataset.samples());

    auto model = make_model();
    model.parameter("model::folds") = 3;
    model.parameter("model::linear::batch") = 10;
    model.parameter("model::linear::scaling") = scaling_type::none;
    model.parameter("model::linear::regularization") = linear::regularization_type::none;

    const auto param_names = strings_t{};
    for (const auto* const loss_id : {"squared", "absolute"})
    {
        [[maybe_unused]] const auto _ = utest_test_name_t{loss_id};

        const auto loss = make_loss(loss_id);
        const auto solver = make_solver(loss_id);
        const auto result = model.fit(generator, samples, *loss, *solver);
        const auto epsilon = string_t{loss_id} == "squared" ? 1e-6 : 1e-5;

        check_result(result, param_names, 0U, epsilon);
        check_model(model, generator, samples, epsilon);
    }
}

UTEST_CASE(regularization_lasso)
{
    const auto dataset = make_dataset(100, 1, 4);
    const auto generator = make_generator(dataset);
    const auto samples = arange(0, dataset.samples());

    auto model = make_model();
    model.parameter("model::folds") = 2;
    model.parameter("model::linear::batch") = 10;
    model.parameter("model::linear::scaling") = scaling_type::standard;
    model.parameter("model::linear::regularization") = linear::regularization_type::lasso;

    const auto param_names = strings_t{"l1reg"};
    for (const auto* const loss_id : {"squared", "absolute"})
    {
        [[maybe_unused]] const auto _ = utest_test_name_t{loss_id};

        const auto loss = make_loss(loss_id);
        const auto solver = make_nonsmooth_solver();
        const auto result = model.fit(generator, samples, *loss, *solver);
        const auto epsilon = 1e-5;

        check_result(result, param_names, 6U, epsilon);
        check_model(model, generator, samples, epsilon);
    }
}

UTEST_CASE(regularization_ridge)
{
    const auto dataset = make_dataset(100, 1, 4);
    const auto generator = make_generator(dataset);
    const auto samples = arange(0, dataset.samples());

    auto model = make_model();
    model.parameter("model::folds") = 2;
    model.parameter("model::linear::batch") = 10;
    model.parameter("model::linear::scaling") = scaling_type::mean;
    model.parameter("model::linear::regularization") = linear::regularization_type::ridge;

    const auto param_names = strings_t{"l2reg"};
    for (const auto* const loss_id : {"squared", "absolute"})
    {
        [[maybe_unused]] const auto _ = utest_test_name_t{loss_id};

        const auto loss = make_loss(loss_id);
        const auto solver = make_solver(loss_id);
        const auto result = model.fit(generator, samples, *loss, *solver);
        const auto epsilon = string_t{loss_id} == "squared" ? 1e-6 : 1e-5;

        check_result(result, param_names, 6U, epsilon);
        check_model(model, generator, samples, epsilon);
    }
}

UTEST_CASE(regularization_variance)
{
    const auto dataset = make_dataset(100, 1, 4);
    const auto generator = make_generator(dataset);
    const auto samples = arange(0, dataset.samples());

    auto model = make_model();
    model.parameter("model::folds") = 2;
    model.parameter("model::linear::batch") = 10;
    model.parameter("model::linear::scaling") = scaling_type::minmax;
    model.parameter("model::linear::regularization") = linear::regularization_type::variance;

    const auto param_names = strings_t{"vAreg"};
    for (const auto* const loss_id : {"squared", "absolute"})
    {
        [[maybe_unused]] const auto _ = utest_test_name_t{loss_id};

        const auto loss = make_loss(loss_id);
        const auto solver = make_solver(loss_id);
        const auto result = model.fit(generator, samples, *loss, *solver);
        const auto epsilon = string_t{loss_id} == "squared" ? 1e-6 : 1e-5;

        check_result(result, param_names, 6U, epsilon);
        check_model(model, generator, samples, epsilon);
    }
}

UTEST_CASE(regularization_elasticnet)
{
    const auto dataset = make_dataset(100, 1, 4);
    const auto generator = make_generator(dataset);
    const auto samples = arange(0, dataset.samples());

    auto model = make_model();
    model.parameter("model::folds") = 2;
    model.parameter("model::linear::batch") = 10;
    model.parameter("model::linear::scaling") = scaling_type::minmax;
    model.parameter("model::linear::regularization") = linear::regularization_type::elasticnet;

    const auto param_names = strings_t{"l1reg", "l2reg"};
    for (const auto* const loss_id : {"squared", "absolute"})
    {
        [[maybe_unused]] const auto _ = utest_test_name_t{loss_id};

        const auto loss = make_loss(loss_id);
        const auto solver = make_nonsmooth_solver();
        const auto result = model.fit(generator, samples, *loss, *solver);
        const auto epsilon = 1e-5;

        check_result(result, param_names, 15U, epsilon);
        check_model(model, generator, samples, epsilon);
    }
}

UTEST_END_MODULE()
