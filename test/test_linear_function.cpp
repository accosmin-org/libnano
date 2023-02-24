#include "fixture/function.h"
#include "fixture/linear.h"
#include "fixture/loss.h"
#include "fixture/solver.h"
#include <nano/linear/function.h>
#include <nano/linear/util.h>

using namespace nano;

static auto make_loss(scaling_type scaling)
{
    return make_loss(scaling == scaling_type::mean ? "mae" : "mse");
}

static auto make_batch(scaling_type scaling)
{
    return (scaling == scaling_type::standard) ? 20 : 15;
}

static void check_vgrad(const linear::function_t& function, const flatten_iterator_t& iterator, const loss_t& loss,
                        const int trials = 100)
{
    const auto& dataset = iterator.dataset();
    const auto  samples = iterator.samples().size();

    tensor2d_t inputs(samples, dataset.columns());
    tensor4d_t outputs(cat_dims(samples, dataset.target_dims()));
    tensor4d_t targets(cat_dims(samples, dataset.target_dims()));

    iterator.loop(
        [&](tensor_range_t range, size_t, tensor2d_cmap_t input, tensor4d_cmap_t target)
        {
            inputs.slice(range)  = input;
            targets.slice(range) = target;
        });

    for (auto trial = 0; trial < trials; ++trial)
    {
        const auto x = make_random_vector<scalar_t>(function.size());
        linear::predict(inputs, function.weights(x), function.bias(x), outputs);

        tensor1d_t values(samples);
        loss.value(targets, outputs, values);
        UTEST_CHECK_CLOSE(function.vgrad(x), values.vector().mean(), epsilon1<scalar_t>());
    }
}

static auto check_minimize(const function_t& function)
{
    const auto* const solver_id      = function.smooth() ? "lbfgs" : "ellipsoid";
    const auto        epsilon_linear = function.smooth() ? 1e-7 : (function.strong_convexity() > 0.0 ? 1e-4 : 1e-2);
    const auto        epsilon_solver = function.smooth() ? 1e-10 : (function.strong_convexity() > 0.0 ? 1e-6 : 1e-4);
    const auto        solver         = make_solver(solver_id);
    solver->lsearchk("cgdescent");

    const auto x0    = vector_t{vector_t::Zero(function.size())};
    const auto state = check_minimize(*solver, function, x0, 20000, epsilon_solver);
    return std::make_pair(state, epsilon_linear);
}

UTEST_BEGIN_MODULE(test_linear_function)

UTEST_CASE(function_noreg)
{
    const auto trials   = 10;
    const auto targets  = tensor_size_t{1};
    const auto samples  = tensor_size_t{10};
    const auto features = tensor_size_t{4};
    const auto scaling  = scaling_type::none;
    const auto loss     = make_loss(scaling);
    const auto batch    = make_batch(scaling);

    const auto datasource = make_linear_datasource(samples, targets, features);
    const auto dataset    = make_dataset(datasource);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(batch);
    iterator.scaling(scaling);

    const auto expected_size = targets * (1 + (1 + 2 + 4 + 6));
    const auto function      = linear::function_t{iterator, *loss, 0.0, 0.0, 0.0};
    UTEST_CHECK_EQUAL(function.size(), expected_size);
    UTEST_CHECK(function.convex() || !loss->convex());
    UTEST_CHECK(function.smooth() || !loss->smooth());
    UTEST_CHECK_EQUAL(function.strong_convexity(), 0.0);

    check_vgrad(function, iterator, *loss, trials);
    check_gradient(function, trials);
    check_convexity(function, trials);
}

UTEST_CASE(function_l1reg)
{
    const auto trials   = 10;
    const auto targets  = tensor_size_t{1};
    const auto samples  = tensor_size_t{10};
    const auto features = tensor_size_t{4};
    const auto scaling  = scaling_type::mean;
    const auto loss     = make_loss(scaling);
    const auto batch    = make_batch(scaling);

    const auto datasource = make_linear_datasource(samples, targets, features);
    const auto dataset    = make_dataset(datasource);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(batch);
    iterator.scaling(scaling);

    const auto expected_size = targets * (1 + (1 + 2 + 4 + 6));
    const auto function      = linear::function_t{iterator, *loss, 1.0, 0.0, 0.0};
    UTEST_CHECK_EQUAL(function.size(), expected_size);
    UTEST_CHECK(function.convex() || !loss->convex());
    UTEST_CHECK(!function.smooth());
    UTEST_CHECK_EQUAL(function.strong_convexity(), 0.0);

    check_gradient(function, trials);
    check_convexity(function, trials);
}

UTEST_CASE(function_l2reg)
{
    const auto trials   = 10;
    const auto targets  = tensor_size_t{1};
    const auto samples  = tensor_size_t{10};
    const auto features = tensor_size_t{4};
    const auto scaling  = scaling_type::minmax;
    const auto loss     = make_loss(scaling);
    const auto batch    = make_batch(scaling);

    const auto datasource = make_linear_datasource(samples, targets, features);
    const auto dataset    = make_dataset(datasource);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(batch);
    iterator.scaling(scaling);

    const auto expected_size = targets * (1 + (1 + 2 + 4 + 6));
    const auto function      = linear::function_t{iterator, *loss, 0.0, 1.0, 0.0};
    UTEST_CHECK_EQUAL(function.size(), expected_size);
    UTEST_CHECK(function.convex() || !loss->convex());
    UTEST_CHECK(function.smooth() || !loss->smooth());
    UTEST_CHECK_EQUAL(function.strong_convexity(), 1.0 / targets / 13.0);

    check_gradient(function, trials);
    check_convexity(function, trials);
}

UTEST_CASE(function_vAreg)
{
    const auto trials   = 10;
    const auto targets  = tensor_size_t{1};
    const auto samples  = tensor_size_t{10};
    const auto features = tensor_size_t{4};
    const auto scaling  = scaling_type::standard;
    const auto loss     = make_loss(scaling);
    const auto batch    = make_batch(scaling);

    const auto datasource = make_linear_datasource(samples, targets, features);
    const auto dataset    = make_dataset(datasource);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(batch);
    iterator.scaling(scaling);

    const auto expected_size = targets * (1 + (1 + 2 + 4 + 6));
    const auto function      = linear::function_t{iterator, *loss, 0.0, 0.0, 1.0};
    UTEST_CHECK_EQUAL(function.size(), expected_size);
    UTEST_CHECK(!function.convex());
    UTEST_CHECK(function.smooth() || !loss->smooth());
    UTEST_CHECK_EQUAL(function.strong_convexity(), 0.0);

    check_gradient(function, trials);
    check_convexity(function, trials);
}

UTEST_CASE(minimize_noreg)
{
    const auto targets  = tensor_size_t{1};
    const auto samples  = tensor_size_t{50};
    const auto features = tensor_size_t{4};
    const auto scaling  = scaling_type::none;
    const auto loss     = make_loss(scaling);
    const auto batch    = make_batch(scaling);

    const auto datasource = make_linear_datasource(samples, targets, features);
    const auto dataset    = make_dataset(datasource);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(batch);
    iterator.scaling(scaling);

    const auto function = linear::function_t{iterator, *loss, 0.0, 0.0, 0.0};

    auto [state, epsilon] = check_minimize(function);
    UTEST_CHECK_CLOSE(state.f, 0.0, epsilon);
    UTEST_CHECK_GREATER(state.inner_iters, 10);

    ::nano::upscale(iterator.flatten_stats(), scaling, iterator.targets_stats(), scaling, function.weights(state.x),
                    function.bias(state.x));

    UTEST_CHECK_CLOSE(datasource.bias(), function.bias(state.x), epsilon);
    UTEST_CHECK_CLOSE(datasource.weights(), function.weights(state.x), epsilon);

    const auto datasource_bias    = datasource.bias().vector();
    const auto datasource_weights = datasource.weights().matrix();
    check_linear(dataset, datasource_weights, datasource_bias, 1e-15);

    const auto function_bias    = function.bias(state.x).vector();
    const auto function_weights = function.weights(state.x).matrix();
    check_linear(dataset, function_weights, function_bias, epsilon);
}

UTEST_CASE(minimize_l1reg)
{
    const auto targets  = tensor_size_t{1};
    const auto samples  = tensor_size_t{50};
    const auto features = tensor_size_t{4};
    const auto scaling  = scaling_type::mean;
    const auto loss     = make_loss(scaling);
    const auto batch    = make_batch(scaling);

    const auto datasource = make_linear_datasource(samples, targets, features);
    const auto dataset    = make_dataset(datasource);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(batch);
    iterator.scaling(scaling);

    const auto function = linear::function_t{iterator, *loss, 1.0, 0.0, 0.0};

    [[maybe_unused]] const auto [state, epsilon] = check_minimize(function);
    UTEST_CHECK_GREATER(state.inner_iters, 10);
}

UTEST_CASE(minimize_l2reg)
{
    const auto targets  = tensor_size_t{1};
    const auto samples  = tensor_size_t{50};
    const auto features = tensor_size_t{4};
    const auto scaling  = scaling_type::minmax;
    const auto loss     = make_loss(scaling);
    const auto batch    = make_batch(scaling);

    const auto datasource = make_linear_datasource(samples, targets, features);
    const auto dataset    = make_dataset(datasource);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(batch);
    iterator.scaling(scaling);

    const auto function = linear::function_t{iterator, *loss, 0.0, 1.0, 0.0};

    [[maybe_unused]] const auto [state, epsilon] = check_minimize(function);
    UTEST_CHECK_GREATER(state.inner_iters, 10);
}

UTEST_CASE(minimize_vAreg)
{
    const auto targets  = tensor_size_t{1};
    const auto samples  = tensor_size_t{50};
    const auto features = tensor_size_t{4};
    const auto scaling  = scaling_type::standard;
    const auto loss     = make_loss(scaling);
    const auto batch    = make_batch(scaling);

    const auto datasource = make_linear_datasource(samples, targets, features);
    const auto dataset    = make_dataset(datasource);

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.batch(batch);
    iterator.scaling(scaling);

    const auto function = linear::function_t{iterator, *loss, 0.0, 0.0, 1.0};

    [[maybe_unused]] const auto [state, epsilon] = check_minimize(function);
    UTEST_CHECK_GREATER(state.inner_iters, 10);
}

UTEST_END_MODULE()
