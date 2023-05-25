#include <nano/learner.h>
#include <nano/mlearn/result.h>
#include <utest/utest.h>

using namespace nano;
using namespace nano::ml;

template <typename tlearner, typename... tfit_args>
[[maybe_unused]] static auto check_fit(const dataset_t& dataset, const tfit_args&... fit_args)
{
    auto learner = tlearner{};
    UTEST_CHECK_NOTHROW(learner.fit(dataset, fit_args...));
    return learner;
}

[[maybe_unused]] static void check_predict(const learner_t& learner, const dataset_t& dataset, const indices_t& samples,
                                           const tensor4d_t& expected_predictions)
{
    UTEST_CHECK_EQUAL(learner.predict(dataset, samples), expected_predictions);
}

[[maybe_unused]] static void check_predict_fails(const learner_t& learner, const dataset_t& dataset,
                                                 const indices_t& samples)
{
    UTEST_CHECK_THROW(learner.predict(dataset, samples), std::runtime_error);
}

[[maybe_unused]] static void check_evaluate_fails(const learner_t& learner, const dataset_t& dataset,
                                                  const indices_t& samples, const loss_t& loss)
{
    UTEST_CHECK_THROW(learner.evaluate(dataset, samples, loss), std::runtime_error);
}

[[maybe_unused]] static void check_result(const result_t& result, const strings_t& expected_param_names,
                                          const size_t min_param_results_size, const tensor_size_t expected_folds,
                                          const scalar_t epsilon)
{
    const auto& param_names        = result.param_names();
    const auto& param_results      = result.param_results();
    const auto  optim_errors_stats = result.stats(value_type::errors);
    const auto  optim_losses_stats = result.stats(value_type::losses);

    UTEST_CHECK_EQUAL(param_names, expected_param_names);
    UTEST_CHECK_CLOSE(optim_errors_stats.m_mean, 0.0, epsilon);
    UTEST_CHECK_CLOSE(optim_losses_stats.m_mean, 0.0, epsilon);

    UTEST_REQUIRE_GREATER_EQUAL(param_results.size(), min_param_results_size);

    const auto optim_losses = make_full_tensor<scalar_t>(make_dims(expected_folds), 0.0);
    const auto optim_errors = make_full_tensor<scalar_t>(make_dims(expected_folds), 0.0);

    tensor_size_t hits = 0;
    for (const auto& param_result : param_results)
    {
        const auto& params = param_result.params();
        UTEST_CHECK_EQUAL(params.size(), static_cast<tensor_size_t>(expected_param_names.size()));
        if (params.size() > 0)
        {
            UTEST_CHECK_GREATER_EQUAL(params.min(), 0.0);
        }

        const auto folds = param_result.folds();
        UTEST_REQUIRE_EQUAL(folds, expected_folds);

        tensor1d_t train_losses(folds), train_errors(folds);
        tensor1d_t valid_losses(folds), valid_errors(folds);

        for (tensor_size_t fold = 0; fold < folds; ++fold)
        {
            train_losses(fold) = param_result.stats(fold, split_type::train, value_type::losses).m_mean;
            train_errors(fold) = param_result.stats(fold, split_type::train, value_type::errors).m_mean;
            valid_losses(fold) = param_result.stats(fold, split_type::valid, value_type::losses).m_mean;
            valid_errors(fold) = param_result.stats(fold, split_type::valid, value_type::errors).m_mean;
        }

        if (close(train_errors, optim_errors, epsilon))
        {
            ++hits;
            UTEST_CHECK_CLOSE(train_losses, optim_losses, 1.0 * epsilon);
            UTEST_CHECK_CLOSE(train_errors, optim_errors, 1.0 * epsilon);
            UTEST_CHECK_CLOSE(valid_losses, optim_losses, 5.0 * epsilon);
            UTEST_CHECK_CLOSE(valid_errors, optim_errors, 5.0 * epsilon);
        }
    }

    UTEST_CHECK_GREATER(hits, 0);
}
