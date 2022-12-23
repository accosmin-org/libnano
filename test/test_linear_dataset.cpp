#include "fixture/linear.h"

using namespace nano;

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

    UTEST_CHECK_EQUAL(dataset.features(), features);
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    UTEST_CHECK_EQUAL(datasource.test_samples(), arange(0, 0));
    UTEST_CHECK_EQUAL(datasource.train_samples(), arange(0, samples));

    check_linear(dataset, weights, bias, epsilon1<scalar_t>());
}

UTEST_END_MODULE()
