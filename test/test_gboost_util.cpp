#include "fixture/linear.h"
#include "fixture/loss.h"
#include <nano/gboost/accumulator.h>
#include <nano/gboost/util.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_gboost_util)

UTEST_CASE(accumulator)
{
    auto accumulator0 = gboost::accumulator_t{3};
    auto accumulator1 = gboost::accumulator_t{3};

    UTEST_CHECK_CLOSE(accumulator0.m_vm1, 0.0, 1e-15);
    UTEST_CHECK_CLOSE(accumulator0.m_gb1, make_vector<scalar_t>(0.0, 0.0, 0.0), 1e-15);

    accumulator0.update(make_tensor<scalar_t>(make_dims(3), 1.0, 2.0, 3.0));

    UTEST_CHECK_CLOSE(accumulator0.m_vm1, 6.0, 1e-15);
    UTEST_CHECK_CLOSE(accumulator0.m_gb1, make_vector<scalar_t>(0.0, 0.0, 0.0), 1e-15);

    accumulator0.update(make_tensor<scalar_t>(make_dims(3), 1.0, 4.0, 0.0));

    UTEST_CHECK_CLOSE(accumulator0.m_vm1, 11.0, 1e-15);
    UTEST_CHECK_CLOSE(accumulator0.m_gb1, make_vector<scalar_t>(0.0, 0.0, 0.0), 1e-15);

    accumulator1.update(make_tensor<scalar_t>(make_dims(3), 3.0, 5.0, 4.0));

    UTEST_CHECK_CLOSE(accumulator1.m_vm1, 12.0, 1e-15);
    UTEST_CHECK_CLOSE(accumulator1.m_gb1, make_vector<scalar_t>(0.0, 0.0, 0.0), 1e-15);

    accumulator0 += accumulator1;

    UTEST_CHECK_CLOSE(accumulator0.m_vm1, 23.0, 1e-15);
    UTEST_CHECK_CLOSE(accumulator0.m_gb1, make_vector<scalar_t>(0.0, 0.0, 0.0), 1e-15);

    accumulator0.m_gb1(0) = 1.0;
    accumulator0 /= 5;

    UTEST_CHECK_CLOSE(accumulator0.m_vm1, 4.6, 1e-15);
    UTEST_CHECK_CLOSE(accumulator0.m_gb1, make_vector<scalar_t>(0.2, 0.0, 0.0), 1e-15);

    accumulator0.clear();

    UTEST_CHECK_CLOSE(accumulator0.m_vm1, 0.0, 1e-15);
    UTEST_CHECK_CLOSE(accumulator0.m_gb1, make_vector<scalar_t>(0.0, 0.0, 0.0), 1e-15);
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
        auto iterator = targets_iterator_t{dataset, samples};
        iterator.batch(batch);

        tensor4d_t outputs{cat_dims(samples.size(), dataset.target_dims())};
        iterator.loop([&](const auto& range, size_t, tensor4d_cmap_t targets) { outputs.slice(range) = targets; });

        tensor2d_t values{2, samples.size()};
        gboost::evaluate(iterator, *loss, outputs, values);

        UTEST_CHECK_CLOSE(values, expected_values, 1e-12);
    }
}

UTEST_CASE(mean)
{
    const auto errors_values = make_tensor<scalar_t>(make_dims(2, 5), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    const auto train_samples = make_indices(0, 1, 2);
    const auto valid_samples = make_indices(1, 3, 4);

    UTEST_CHECK_CLOSE(gboost::mean_loss(errors_values, train_samples), 6.0, 1e-12);
    UTEST_CHECK_CLOSE(gboost::mean_loss(errors_values, valid_samples), 23.0 / 3.0, 1e-12);

    UTEST_CHECK_CLOSE(gboost::mean_error(errors_values, train_samples), 1.0, 1e-12);
    UTEST_CHECK_CLOSE(gboost::mean_error(errors_values, valid_samples), 8.0 / 3.0, 1e-12);
}

UTEST_CASE(early_stopping)
{
    const auto epsilon       = 1.0;
    const auto patience      = 3;
    const auto train_samples = make_indices(0, 1, 2);
    const auto valid_samples = make_indices(1, 3, 4);

    auto optimum_round  = size_t{0U};
    auto optimum_value  = std::numeric_limits<scalar_t>::max();
    auto optimum_values = make_tensor<scalar_t>(make_dims(2, 5), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 9, 9, 9, 9, 9, 9, 9, 9, 9, 9);
        const auto wlearners = rwlearners_t{};

        UTEST_CHECK(!gboost::done(values, train_samples, valid_samples, wlearners, epsilon, patience, optimum_round,
                                  optimum_value, optimum_values));
        UTEST_CHECK_EQUAL(optimum_round, 0U);
        UTEST_CHECK_CLOSE(optimum_value, 9.0, 1e-12);
        UTEST_CHECK_CLOSE(optimum_values, values, 1e-12);
    }
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 8, 8, 8, 7, 6, 8, 8, 8, 8, 8);
        const auto wlearners = rwlearners_t{1U};

        UTEST_CHECK(!gboost::done(values, train_samples, valid_samples, wlearners, epsilon, patience, optimum_round,
                                  optimum_value, optimum_values));
        UTEST_CHECK_EQUAL(optimum_round, 1U);
        UTEST_CHECK_CLOSE(optimum_value, 7.0, 1e-12);
        UTEST_CHECK_CLOSE(optimum_values, values, 1e-12);
    }
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 8, 7, 8, 7, 6, 8, 8, 8, 8, 8);
        const auto wlearners = rwlearners_t{2U};

        UTEST_CHECK(!gboost::done(values, train_samples, valid_samples, wlearners, epsilon, patience, optimum_round,
                                  optimum_value, optimum_values));
        UTEST_CHECK_EQUAL(optimum_round, 1U);
        UTEST_CHECK_CLOSE(optimum_value, 7.0, 1e-12);
        UTEST_CHECK_NOT_CLOSE(optimum_values, values, 1e-12);
    }
    for (const auto rounds : {size_t{4U}, size_t{5U}})
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 8, 7, 8, 7, 6, 8, 8, 8, 8, 8);
        const auto wlearners = rwlearners_t{rounds};

        UTEST_CHECK(gboost::done(values, train_samples, valid_samples, wlearners, epsilon, patience, optimum_round,
                                 optimum_value, optimum_values));
        UTEST_CHECK_EQUAL(optimum_round, 1U);
        UTEST_CHECK_CLOSE(optimum_value, 7.0, 1e-12);
        UTEST_CHECK_NOT_CLOSE(optimum_values, values, 1e-12);
    }
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
        const auto wlearners = rwlearners_t{6U};

        UTEST_CHECK(!gboost::done(values, train_samples, indices_t{}, wlearners, epsilon, patience, optimum_round,
                                  optimum_value, optimum_values));
        UTEST_CHECK_EQUAL(optimum_round, 6U);
        UTEST_CHECK_CLOSE(optimum_value, 0.0, 1e-12);
        UTEST_CHECK_CLOSE(optimum_values, values, 1e-12);
    }
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 0, 0, 1, 1, 2, 2, 0, 0, 4, 4);
        const auto wlearners = rwlearners_t{3U};

        UTEST_CHECK(gboost::done(values, train_samples, valid_samples, wlearners, epsilon, patience, optimum_round,
                                 optimum_value, optimum_values));
        UTEST_CHECK_EQUAL(optimum_round, 3U);
        UTEST_CHECK_CLOSE(optimum_value, 1.0, 1e-12);
        UTEST_CHECK_CLOSE(optimum_values, values, 1e-12);
    }
}

UTEST_END_MODULE()
