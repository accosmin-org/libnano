#include <fixture/datasource/linear.h>
#include <fixture/linear.h>

using namespace nano;

namespace
{
auto make_inputs(const dataset_t& dataset)
{
    auto iterator = flatten_iterator_t{dataset, arange(0, dataset.samples())};
    iterator.scaling(scaling_type::none);

    auto values = tensor2d_t{dataset.samples(), dataset.columns()};
    iterator.loop([&](tensor_range_t range, size_t, tensor2d_cmap_t chunk) { values.slice(range) = chunk; });
    return values;
}

auto make_target(const dataset_t& dataset)
{
    auto iterator = targets_iterator_t{dataset, arange(0, dataset.samples())};
    iterator.scaling(scaling_type::none);

    auto values = tensor4d_t{cat_dims(dataset.samples(), dataset.target_dims())};
    iterator.loop([&](tensor_range_t range, size_t, tensor4d_cmap_t chunk) { values.slice(range) = chunk; });
    return values;
}
} // namespace

UTEST_BEGIN_MODULE(test_linear_dataset)

UTEST_CASE(dataset)
{
    const auto targets  = tensor_size_t{3};
    const auto samples  = tensor_size_t{10};
    const auto features = tensor_size_t{4};

    const auto datasource = make_linear_datasource(samples, targets, features);
    const auto dataset    = make_dataset(datasource);

    UTEST_CHECK_EQUAL(dataset.target(), feature_t{"Wx+b+eps"}.scalar(feature_type::float64, make_dims(targets, 1, 1)));

    const auto bias = datasource.bias().vector();
    UTEST_REQUIRE_EQUAL(bias.size(), targets);

    const auto weights = datasource.weights().matrix();
    UTEST_REQUIRE_EQUAL(weights.rows(), targets);
    UTEST_REQUIRE_EQUAL(weights.cols(), 13 * features / 4);

    UTEST_CHECK_EQUAL(dataset.type(), task_type::regression);
    UTEST_CHECK_EQUAL(dataset.features(), features);
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    UTEST_CHECK_EQUAL(datasource.test_samples(), arange(0, 0));
    UTEST_CHECK_EQUAL(datasource.train_samples(), arange(0, samples));

    check_linear(dataset, weights, bias, epsilon1<scalar_t>());
}

UTEST_CASE(seed)
{
    const auto targets  = tensor_size_t{3};
    const auto samples  = tensor_size_t{10};
    const auto features = tensor_size_t{4};
    const auto seed1    = 42U;
    const auto seed2    = 43U;

    const auto datasource11 = make_linear_datasource(samples, targets, features, "datasource::linear::seed", seed1);
    const auto datasource12 = make_linear_datasource(samples, targets, features, "datasource::linear::seed", seed1);
    const auto datasource21 = make_linear_datasource(samples, targets, features, "datasource::linear::seed", seed2);

    const auto dataset11 = make_dataset(datasource11);
    const auto dataset12 = make_dataset(datasource12);
    const auto dataset21 = make_dataset(datasource21);

    const auto inputs11 = make_inputs(dataset11);
    const auto inputs12 = make_inputs(dataset12);
    const auto inputs21 = make_inputs(dataset21);

    const auto target11 = make_target(dataset11);
    const auto target12 = make_target(dataset12);
    const auto target21 = make_target(dataset21);

    UTEST_CHECK_CLOSE(datasource11.bias(), datasource12.bias(), 1e-15);
    UTEST_CHECK_NOT_CLOSE(datasource11.bias(), datasource21.bias(), 1e-15);

    UTEST_CHECK_CLOSE(datasource11.weights(), datasource12.weights(), 1e-15);
    UTEST_CHECK_NOT_CLOSE(datasource11.weights(), datasource21.weights(), 1e-15);

    UTEST_CHECK_CLOSE(inputs11, inputs12, 1e-15);
    UTEST_CHECK_NOT_CLOSE(inputs11, inputs21, 1e-15);

    UTEST_CHECK_CLOSE(target11, target12, 1e-15);
    UTEST_CHECK_NOT_CLOSE(target11, target21, 1e-15);
}

UTEST_END_MODULE()
