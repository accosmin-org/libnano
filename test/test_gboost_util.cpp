#include <fixture/dataset.h>
#include <fixture/datasource/linear.h>
#include <fixture/loss.h>
#include <nano/gboost/accumulator.h>
#include <nano/gboost/early_stopping.h>
#include <nano/gboost/sampler.h>
#include <nano/gboost/util.h>

using namespace nano;

namespace
{
void check_samples(const indices_t& selected_samples, const indices_t& train_samples)
{
    UTEST_CHECK_EQUAL(selected_samples.size(), train_samples.size());
    UTEST_CHECK(std::is_sorted(selected_samples.begin(), selected_samples.end()));
    for (const auto sample : selected_samples)
    {
        UTEST_CHECK(std::find(train_samples.begin(), train_samples.end(), sample) != train_samples.end());
    }
}
} // namespace

UTEST_BEGIN_MODULE()

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

        UTEST_CHECK_CLOSE(values, expected_values, 1e-15);
    }
}

UTEST_CASE(mean)
{
    const auto errors_values = make_tensor<scalar_t>(make_dims(2, 5), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    const auto train_samples = make_indices(0, 1, 2);
    const auto valid_samples = make_indices(1, 3, 4);

    UTEST_CHECK_CLOSE(gboost::mean_loss(errors_values, train_samples), 6.0, 1e-15);
    UTEST_CHECK_CLOSE(gboost::mean_loss(errors_values, valid_samples), 23.0 / 3.0, 1e-15);

    UTEST_CHECK_CLOSE(gboost::mean_error(errors_values, train_samples), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(gboost::mean_error(errors_values, valid_samples), 8.0 / 3.0, 1e-15);
}

UTEST_CASE(sampler)
{
    const auto train_samples = make_indices(0, 1, 2, 5, 9, 7, 6);
    const auto errors_losses = make_full_tensor<scalar_t>(make_dims(2, 10), 0.0);
    const auto gradients     = make_full_tensor<scalar_t>(make_dims(10, 1, 1, 1), 0.0);

    for (const auto seed : {1U, 7U, 42U, 1000U})
    {
        auto sampler = gboost::sampler_t{train_samples, gboost_subsample::off, seed, 1.0};
        UTEST_CHECK_EQUAL(sampler.sample(errors_losses, gradients), train_samples);
    }
}

UTEST_CASE(bootstrap_sampler)
{
    const auto train_samples = make_indices(0, 1, 2, 5, 9, 7, 6);
    const auto errors_losses = make_full_tensor<scalar_t>(make_dims(2, 10), 1.42);
    const auto gradients     = make_full_tensor<scalar_t>(make_dims(10, 1, 1, 1), 4.2);

    for (const auto subsample :
         {gboost_subsample::bootstrap, gboost_subsample::wei_loss_bootstrap, gboost_subsample::wei_grad_bootstrap})
    {
        auto prev_samples = indices_t{};
        for (const auto seed : {1U, 7U, 42U, 1000U})
        {
            auto sampler = gboost::sampler_t{train_samples, subsample, seed, 1.0};

            const auto samples = sampler.sample(errors_losses, gradients);
            check_samples(samples, train_samples);
            if (prev_samples.size() == samples.size())
            {
                UTEST_CHECK_NOT_EQUAL(prev_samples, samples);
            }
            prev_samples = samples;
        }
    }
}

UTEST_CASE(early_stopping)
{
    const auto epsilon       = 1.0;
    const auto patience      = 3;
    const auto train_samples = make_indices(0, 1, 2);
    const auto valid_samples = make_indices(1, 3, 4);

    auto optimum = gboost::early_stopping_t{make_full_tensor<scalar_t>(make_dims(2, 5), 0.0)};
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 9, 9, 9, 9, 9, 9, 9, 9, 9, 9);
        const auto wlearners = rwlearners_t{};

        UTEST_CHECK(!optimum.done(values, train_samples, valid_samples, wlearners, epsilon, patience));
        UTEST_CHECK_EQUAL(optimum.round(), 0U);
        UTEST_CHECK_CLOSE(optimum.value(), 9.0, 1e-15);
        UTEST_CHECK_CLOSE(optimum.values(), values, 1e-15);
    }
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 8, 8, 8, 7, 6, 8, 8, 8, 8, 8);
        const auto wlearners = rwlearners_t{1U};

        UTEST_CHECK(!optimum.done(values, train_samples, valid_samples, wlearners, epsilon, patience));
        UTEST_CHECK_EQUAL(optimum.round(), 1U);
        UTEST_CHECK_CLOSE(optimum.value(), 7.0, 1e-15);
        UTEST_CHECK_CLOSE(optimum.values(), values, 1e-15);
    }
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 8, 7, 8, 7, 6, 8, 8, 8, 8, 8);
        const auto wlearners = rwlearners_t{2U};

        UTEST_CHECK(!optimum.done(values, train_samples, valid_samples, wlearners, epsilon, patience));
        UTEST_CHECK_EQUAL(optimum.round(), 1U);
        UTEST_CHECK_CLOSE(optimum.value(), 7.0, 1e-15);
        UTEST_CHECK_NOT_CLOSE(optimum.values(), values, 1e-15);
    }
    for (const auto rounds : {size_t{4U}, size_t{5U}})
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 8, 7, 8, 7, 6, 8, 8, 8, 8, 8);
        const auto wlearners = rwlearners_t{rounds};

        UTEST_CHECK(optimum.done(values, train_samples, valid_samples, wlearners, epsilon, patience));
        UTEST_CHECK_EQUAL(optimum.round(), 1U);
        UTEST_CHECK_CLOSE(optimum.value(), 7.0, 1e-15);
        UTEST_CHECK_NOT_CLOSE(optimum.values(), values, 1e-15);
    }
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
        const auto wlearners = rwlearners_t{6U};

        UTEST_CHECK(!optimum.done(values, train_samples, indices_t{}, wlearners, epsilon, patience));
        UTEST_CHECK_EQUAL(optimum.round(), 6U);
        UTEST_CHECK_CLOSE(optimum.value(), 0.0, 1e-15);
        UTEST_CHECK_CLOSE(optimum.values(), values, 1e-15);
    }
    {
        const auto values    = make_tensor<scalar_t>(make_dims(2, 5), 0, 0, 1, 1, 2, 2, 0, 0, 4, 4);
        const auto wlearners = rwlearners_t{3U};

        UTEST_CHECK(optimum.done(values, train_samples, valid_samples, wlearners, epsilon, patience));
        UTEST_CHECK_EQUAL(optimum.round(), 3U);
        UTEST_CHECK_CLOSE(optimum.value(), 1.0, 1e-15);
        UTEST_CHECK_CLOSE(optimum.values(), values, 1e-15);
    }
}

UTEST_CASE(tune_shrinkage)
{
    const auto datasource = make_linear_datasource(20, 3, 4);
    const auto dataset    = make_dataset(datasource);
    const auto loss       = make_loss("mse");

    const auto samples = make_indices(0, 3, 4, 5, 11, 17);

    auto outputs  = make_random_tensor<scalar_t>(cat_dims(dataset.samples(), dataset.target_dims()));
    auto woutputs = make_random_tensor<scalar_t>(cat_dims(dataset.samples(), dataset.target_dims()));

    for (const auto expected_shrinkage : {0.4, 0.1, 1.0, 0.6})
    {
        const auto iterator = targets_iterator_t{dataset, samples};

        iterator.loop(
            [&](const auto& range, size_t, tensor4d_cmap_t targets)
            {
                for (auto i = range.begin(); i < range.end(); ++i)
                {
                    const auto sample      = samples(i);
                    const auto offset      = i - range.begin();
                    outputs.vector(sample) = targets.vector(offset) - expected_shrinkage * woutputs.vector(sample);
                }
            });

        const auto shrinkage = gboost::tune_shrinkage(iterator, *loss, outputs, woutputs);
        UTEST_CHECK_CLOSE(shrinkage, expected_shrinkage, 1e-15);
    }
}

UTEST_END_MODULE()
