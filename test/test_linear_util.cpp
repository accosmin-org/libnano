#include <fixture/datasource/linear.h>
#include <fixture/linear.h>
#include <fixture/loss.h>
#include <nano/core/reduce.h>
#include <nano/linear/accumulator.h>
#include <nano/linear/util.h>

using namespace nano;

UTEST_BEGIN_MODULE()

UTEST_CASE(accumulator)
{
    const auto fill_accumulator = [](linear::accumulator_t& accumulator, const scalar_t value)
    {
        accumulator.m_fx = value;
        accumulator.m_gb.full(value);
        accumulator.m_gw.full(value);
        accumulator.m_hx.full(value);
    };

    const auto make_accumulators = [&]()
    {
        auto accumulators = std::vector<linear::accumulator_t>(3U, linear::accumulator_t{3, 2});
        fill_accumulator(accumulators[0], 1);
        fill_accumulator(accumulators[1], 2);
        fill_accumulator(accumulators[2], 3);
        return accumulators;
    };

    {
        auto accumulators = make_accumulators();

        const auto& accumulator0 = sum_reduce(accumulators, 6);
        UTEST_CHECK_CLOSE(accumulator0.m_fx, 6.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(accumulator0.m_gb, make_full_tensor<scalar_t>(make_dims(2), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(accumulator0.m_gw, make_full_tensor<scalar_t>(make_dims(2, 3), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(accumulator0.m_hx, make_full_tensor<scalar_t>(make_dims(8, 8), 6.0 / 6.0), 1e-12);
    }
}

UTEST_CASE(predict)
{
    const auto epsilon = epsilon1<scalar_t>();

    const auto bias    = make_random_tensor<scalar_t>(make_dims(3));
    const auto weights = make_random_tensor<scalar_t>(make_dims(3, 5));
    const auto inputs  = make_random_tensor<scalar_t>(make_dims(11, 5));

    tensor4d_t outputs;
    linear::predict(inputs, weights, bias, outputs);

    for (tensor_size_t sample = 0; sample < inputs.size<0>(); ++sample)
    {
        UTEST_CHECK_CLOSE(outputs.vector(sample), weights.matrix() * inputs.vector(sample) + bias.vector(), epsilon);
    }
}

UTEST_CASE(evaluate)
{
    const auto datasource = make_linear_datasource(20, 3, 4);
    const auto dataset    = make_dataset(datasource);
    const auto loss       = make_loss("mse");

    const auto samples         = arange(0, dataset.samples());
    const auto expected_values = make_full_tensor<scalar_t>(make_dims(2, samples.size()), 0.0);

    for (const auto batch : {1, 2, 3, 4})
    {
        const auto& bias    = datasource.bias();
        const auto& weights = datasource.weights();
        const auto  values  = linear::evaluate(dataset, samples, *loss, weights, bias, batch);

        UTEST_CHECK_CLOSE(values, expected_values, 1e-12);
    }
}

UTEST_CASE(feature_importance)
{
    const auto datasource = make_linear_datasource(20, 1, 4);
    const auto dataset    = make_dataset(datasource);
    {
        const auto weights = make_tensor<scalar_t>(make_dims(1, 13), 1, 0, -1, 0, 0, 1, -2, -3, 0, 0, 1, 1, 1);
        const auto feature_importance  = linear::feature_importance(dataset, weights);
        const auto expected_importance = make_tensor<scalar_t>(make_dims(4), 1, 2, 2, 6);

        UTEST_CHECK_CLOSE(feature_importance, expected_importance, 1e-15);
    }
    {
        const auto weights = make_tensor<scalar_t>(make_dims(1, 13), 0, 0, 0, 0, 0, 1, -2, -3, 0, 0, 0, 0, 0);
        const auto feature_importance  = linear::feature_importance(dataset, weights);
        const auto expected_importance = make_tensor<scalar_t>(make_dims(4), 0, 1, 2, 3);

        UTEST_CHECK_CLOSE(feature_importance, expected_importance, 1e-15);
    }
}

UTEST_CASE(sparsity_ratio)
{
    const auto feature_importance = make_tensor<scalar_t>(make_dims(8), 0.0, 1e-9, 1e-3, 1e-1, 1e-7, 1e-7, 1e-1, 1e+1);

    UTEST_CHECK_CLOSE(linear::sparsity_ratio(feature_importance, 1e-10), 1.0 / 8.0, 1e-15);
    UTEST_CHECK_CLOSE(linear::sparsity_ratio(feature_importance, 1e-8), 2.0 / 8.0, 1e-15);
    UTEST_CHECK_CLOSE(linear::sparsity_ratio(feature_importance, 1e-6), 4.0 / 8.0, 1e-15);
    UTEST_CHECK_CLOSE(linear::sparsity_ratio(feature_importance, 1e-2), 5.0 / 8.0, 1e-15);
}

UTEST_END_MODULE()
