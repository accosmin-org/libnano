#include <nano/model.h>
#include <utest/utest.h>

using namespace nano;

using split_type = fit_result_t::split_type;
using value_type = fit_result_t::value_type;

template <typename tmodel, typename... tfit_args>
[[maybe_unused]] static auto check_fit(const dataset_t& dataset, const tfit_args&... fit_args)
{
    auto model = tmodel{};
    UTEST_CHECK_NOTHROW(model.fit(dataset, fit_args...));
    return model;
}

[[maybe_unused]] static void check_predict(const model_t& model, const dataset_t& dataset, const indices_t& samples,
                                           const tensor4d_t& expected_predictions)
{
    UTEST_CHECK_EQUAL(model.predict(dataset, samples), expected_predictions);
}

[[maybe_unused]] static void check_predict_fails(const model_t& model, const dataset_t& dataset,
                                                 const indices_t& samples)
{
    UTEST_CHECK_THROW(model.predict(dataset, samples), std::runtime_error);
}

[[maybe_unused]] static void check_result(const fit_result_t& result, const strings_t& expected_param_names,
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

    const auto opt_losses = make_full_tensor<scalar_t>(make_dims(2), 0.0);
    const auto opt_errors = make_full_tensor<scalar_t>(make_dims(2), 0.0);

    tensor_size_t hits = 0;
    for (const auto& param_result : param_results)
    {
        const auto& params = param_result.params();
        UTEST_CHECK_EQUAL(params.size(), static_cast<tensor_size_t>(expected_param_names.size()));
        if (params.size() > 0)
        {
            UTEST_CHECK_GREATER(params.min(), 0.0);
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

        if (close(train_errors, opt_errors, epsilon))
        {
            ++hits;
            UTEST_CHECK_CLOSE(train_losses, opt_losses, 1.0 * epsilon);
            UTEST_CHECK_CLOSE(train_errors, opt_errors, 1.0 * epsilon);
            UTEST_CHECK_CLOSE(valid_losses, opt_losses, 5.0 * epsilon);
            UTEST_CHECK_CLOSE(valid_errors, opt_errors, 5.0 * epsilon);
        }
    }

    UTEST_CHECK_GREATER(hits, 0);
}
