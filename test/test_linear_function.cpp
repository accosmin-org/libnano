#include "fixture/loss.h"
#include "fixture/linear.h"
#include "fixture/solver.h"
#include "fixture/function.h"
#include <nano/linear/util.h>
#include <nano/linear/function.h>

using namespace nano;

static auto make_loss(scaling_type scaling)
{
    return make_loss(
        (scaling == scaling_type::none) ? "squared" :
        (scaling == scaling_type::mean) ? "absolute" : "cauchy");
}

static auto make_batch(scaling_type scaling)
{
    return (scaling == scaling_type::standard) ? 5 : 3;
}

static auto make_execution(scaling_type scaling)
{
    return (scaling == scaling_type::minmax) ? execution_type::par : execution_type::seq;
}

static auto make_iterator(const dataset_generator_t& generator, scaling_type scaling)
{
    return make_iterator(generator, make_execution(scaling), make_batch(scaling), scaling);
}

static void check_vgrad(const linear::function_t& function, int trials = 100)
{
    const auto& loss = function.loss();
    const auto& iterator = function.iterator();
    const auto& generator = iterator.generator();
    const auto samples = iterator.samples().size();

    tensor2d_t inputs(samples, generator.columns());
    tensor4d_t outputs(cat_dims(samples, generator.target_dims()));
    tensor4d_t targets(cat_dims(samples, generator.target_dims()));

    iterator.loop([&] (tensor_range_t range, size_t, tensor2d_cmap_t input, tensor4d_cmap_t target)
    {
        inputs.slice(range) = input;
        targets.slice(range) = target;
    });

    for (auto trial = 0; trial < trials; ++ trial)
    {
        const vector_t x = vector_t::Random(function.size());
        linear::predict(inputs, function.weights(x), function.bias(x), outputs);

        tensor1d_t values(samples);
        loss.value(targets, outputs, values);
        UTEST_CHECK_CLOSE(function.vgrad(x), values.vector().mean(), epsilon1<scalar_t>());
    }
}

static auto check_minimize(const function_t& function)
{
    const auto* const solver_id = function.smooth() ? "lbfgs" : "asgm";
    const auto solver = make_solver(solver_id);
    solver->lsearchk("cgdescent");

    const vector_t x0 = vector_t::Random(function.size());
    return check_minimize(*solver, solver_id, function, x0, 10000, 1e-9);
}

UTEST_BEGIN_MODULE(test_linear_function)

UTEST_CASE(function)
{
    const auto trials = 10;
    const auto targets = tensor_size_t{1};
    const auto samples = tensor_size_t{10};
    const auto features = tensor_size_t{4};

    const auto dataset = make_dataset(samples, targets, features);
    const auto generator = make_generator(dataset);

    for (const auto scaling : enum_values<scaling_type>())
    {
        [[maybe_unused]] const auto _ = utest_test_name_t{scat(scaling)};

        const auto loss = make_loss(scaling);
        const auto iterator = make_iterator(generator, scaling);
        const auto expected_size = targets * (1 + (1 + 2 + 4 + 6));
        {
            const auto function = linear::function_t{iterator, *loss, 0.0, 0.0, 0.0};
            UTEST_CHECK_EQUAL(function.size(), expected_size);
            UTEST_CHECK(function.convex() || !loss->convex());
            UTEST_CHECK(function.smooth() || !loss->smooth());
            UTEST_CHECK_EQUAL(function.strong_convexity(), 0.0);

            check_vgrad(function, trials);
            check_gradient(function, trials);
            check_convexity(function, trials);
        }
        {
            const auto function = linear::function_t{iterator, *loss, 1.0, 0.0, 0.0};
            UTEST_CHECK_EQUAL(function.size(), expected_size);
            UTEST_CHECK(function.convex() || !loss->convex());
            UTEST_CHECK(!function.smooth());
            UTEST_CHECK_EQUAL(function.strong_convexity(), 0.0);

            check_gradient(function, trials);
            check_convexity(function, trials);
        }
        {
            const auto function = linear::function_t{iterator, *loss, 0.0, 1.0, 0.0};
            UTEST_CHECK_EQUAL(function.size(), expected_size);
            UTEST_CHECK(function.convex() || !loss->convex());
            UTEST_CHECK(function.smooth() || !loss->smooth());
            UTEST_CHECK_EQUAL(function.strong_convexity(), 1.0 / targets / 13.0);

            check_gradient(function, trials);
            check_convexity(function, trials);
        }
        {
            const auto function = linear::function_t{iterator, *loss, 0.0, 0.0, 1.0};
            UTEST_CHECK_EQUAL(function.size(), expected_size);
            UTEST_CHECK(!function.convex());
            UTEST_CHECK(function.smooth() || !loss->smooth());
            UTEST_CHECK_EQUAL(function.strong_convexity(), 0.0);

            check_convexity(function, trials);
            check_gradient(function, trials, 20.0);
        }
    }
}

UTEST_CASE(minimize)
{
    const auto targets = tensor_size_t{1};
    const auto samples = tensor_size_t{50};
    const auto features = tensor_size_t{4};

    const auto dataset = make_dataset(samples, targets, features);
    const auto generator = make_generator(dataset);

    for (const auto scaling : enum_values<scaling_type>())
    {
        [[maybe_unused]] const auto _ = utest_test_name_t{scat(scaling)};

        const auto loss = make_loss(scaling);
        const auto iterator = make_iterator(generator, scaling);
        {
            const auto function = linear::function_t{iterator, *loss, 0.0, 0.0, 0.0};

            auto state = check_minimize(function);
            UTEST_CHECK_CLOSE(state.f, 0.0, epsilon3<scalar_t>());

            ::nano::upscale(
                iterator.flatten_stats(), scaling,
                iterator.targets_stats(), scaling,
                function.weights(state.x),
                function.bias(state.x));

            UTEST_CHECK_CLOSE(dataset.bias(), function.bias(state.x), epsilon3<scalar_t>());
            UTEST_CHECK_CLOSE(dataset.weights(), function.weights(state.x), epsilon3<scalar_t>());

            const auto dataset_bias = dataset.bias().vector();
            const auto dataset_weights = dataset.weights().matrix();
            check_linear(generator, dataset_weights, dataset_bias, epsilon1<scalar_t>());

            const auto function_bias = function.bias(state.x).vector();
            const auto function_weights = function.weights(state.x).matrix();
            check_linear(generator, function_weights, function_bias, epsilon3<scalar_t>());
        }
        {
            const auto function = linear::function_t{iterator, *loss, 1.0, 0.0, 0.0};

            check_minimize(function);
        }
        {
            const auto function = linear::function_t{iterator, *loss, 0.0, 1.0, 0.0};

            check_minimize(function);
        }
        {
            const auto function = linear::function_t{iterator, *loss, 0.0, 0.0, 1.0};

            check_minimize(function);
        }
    }
}

UTEST_END_MODULE()
