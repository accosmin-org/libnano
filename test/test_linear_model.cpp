#include "fixture/linear.h"
#include "fixture/loss.h"
#include "fixture/solver.h"
#include "fixture/splitter.h"
#include "fixture/tuner.h"
#include <nano/linear/enums.h>

using namespace nano;
using namespace nano::ml;

namespace
{
auto make_smooth_solver()
{
    auto solver                            = make_solver("hoshino");
    solver->parameter("solver::max_evals") = 1000;
    solver->parameter("solver::epsilon")   = 1e-10;
    solver->lsearchk("cgdescent");
    return solver;
}

auto make_nonsmooth_solver()
{
    auto solver                            = make_solver("hoshino");
    solver->parameter("solver::max_evals") = 1000;
    solver->parameter("solver::epsilon")   = 1e-10;
    return solver;
}

auto make_model()
{
    auto model                       = linear_model_t{};
    model.parameter("linear::batch") = 30;
    return model;
}

auto make_fit_params(const rsolver_t& solver)
{
    return params_t{}.splitter(make_splitter("k-fold", 2)).solver(solver).logger(params_t::make_logger_stdio());
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

void check_model(const linear_model_t& model, const dataset_t& dataset, const indices_t& samples,
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
} // namespace

UTEST_BEGIN_MODULE(test_linear_model)

UTEST_CASE(regularization_none)
{
    const auto datasource = make_linear_datasource(100, 1, 4);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto model                                = make_model();
    model.parameter("linear::scaling")        = scaling_type::none;
    model.parameter("linear::regularization") = linear::regularization_type::none;

    const auto param_names = strings_t{};
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss       = make_loss(loss_id);
        const auto solver     = string_t(loss_id) == "mse" ? make_smooth_solver() : make_nonsmooth_solver();
        const auto fit_params = make_fit_params(solver);
        const auto result     = model.fit(dataset, samples, *loss, fit_params);
        const auto epsilon    = string_t{loss_id} == "mse" ? 1e-6 : 1e-4;

        check_result(result, param_names, 2, epsilon);
        check_model(model, dataset, samples, epsilon);
    }
}

UTEST_CASE(regularization_lasso)
{
    const auto datasource = make_linear_datasource(100, 1, 4);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto model                                = make_model();
    model.parameter("linear::scaling")        = scaling_type::standard;
    model.parameter("linear::regularization") = linear::regularization_type::lasso;

    const auto param_names = strings_t{"l1reg"};
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss       = make_loss(loss_id);
        const auto solver     = make_nonsmooth_solver();
        const auto fit_params = make_fit_params(solver);
        const auto result     = model.fit(dataset, samples, *loss, fit_params);
        const auto epsilon    = 1e-4;

        check_result(result, param_names, 2, epsilon);
        check_model(model, dataset, samples, epsilon);
    }
}

UTEST_CASE(regularization_ridge)
{
    const auto datasource = make_linear_datasource(100, 1, 4);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto model                                = make_model();
    model.parameter("linear::scaling")        = scaling_type::mean;
    model.parameter("linear::regularization") = linear::regularization_type::ridge;

    const auto param_names = strings_t{"l2reg"};
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss       = make_loss(loss_id);
        const auto solver     = string_t(loss_id) == "mse" ? make_smooth_solver() : make_nonsmooth_solver();
        const auto fit_params = make_fit_params(solver);
        const auto result     = model.fit(dataset, samples, *loss, fit_params);
        const auto epsilon    = string_t{loss_id} == "mse" ? 1e-6 : 1e-4;

        check_result(result, param_names, 2, epsilon);
        check_model(model, dataset, samples, epsilon);
    }
}

UTEST_CASE(regularization_elasticnet)
{
    const auto datasource = make_linear_datasource(100, 1, 4);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto model                                = make_model();
    model.parameter("linear::scaling")        = scaling_type::minmax;
    model.parameter("linear::regularization") = linear::regularization_type::elasticnet;

    const auto param_names = strings_t{"l1reg", "l2reg"};
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss       = make_loss(loss_id);
        const auto solver     = make_nonsmooth_solver();
        const auto fit_params = make_fit_params(solver);
        const auto result     = model.fit(dataset, samples, *loss, fit_params);
        const auto epsilon    = 1e-4;

        check_result(result, param_names, 2, epsilon);
        check_model(model, dataset, samples, epsilon);
    }
}

UTEST_END_MODULE()
