#pragma once

#include <nano/core/parallel.h>
#include <nano/mlearn/config.h>
#include <nano/mlearn/result.h>

namespace nano::ml
{
NANO_PUBLIC result_t::params_t make_param_results(const tensor2d_t& all_params, tensor_size_t folds);

///
/// \brief tune hyper-parameters required to fit a machine learning model.
///
/// NB: each set of hyper-parameter values is evaluated using the given callback.
/// NB: the tuning is performed in parallel across the current set of hyper-parameter values to evaluate and the folds.
///
template <class tevaluator>
result_t tune(const string_t& prefix, const indices_t& samples, const config_t& config, strings_t param_names,
              const param_spaces_t& param_spaces, const tevaluator& evaluator)
{
    const auto splits = config.splitter().split(samples);
    const auto folds  = static_cast<tensor_size_t>(splits.size());

    auto thread_pool = parallel::pool_t{};
    auto fit_result  = result_t{std::move(param_names)};

    // tune hyper-parameters (if any) in parallel by hyper-parameter trials and folds
    const auto callback = [&](const tensor2d_t& all_params)
    {
        auto param_results = make_param_results(all_params, folds);

        const auto thread_callback = [&](const tensor_size_t index, size_t)
        {
            const auto fold  = index % folds;
            const auto trial = index / folds;
            const auto ifold = static_cast<size_t>(fold);

            const auto params                    = all_params.tensor(trial);

            const auto& [tr_samples, vd_samples] = splits[ifold];

            const auto* const closest = fit_result.closest(params);
            const auto        closest_

                auto [train_values, valid_values, extra] = evaluator(train_samples, valid_samples, params, closest);

            auto& param_result = param_results[static_cast<size_t>(trial)];
            param_result.evaluate(fold, std::move(train_values), std::move(valid_values), std::move(extra));
        };

        const auto trials = all_params.size<0>();
        thread_pool.map(folds * trials, thread_callback);

        tensor1d_t values{trials};
        for (tensor_size_t trial = 0; trial < trials; ++trial)
        {
            auto& param_result = param_results[static_cast<size_t>(trial)];
            values(trial)      = param_result.value();
            fit_result.add(std::move(param_result));
        }

        config.log(fit_result, prefix);

        return values;
    };

    if (!param_spaces.empty())
    {
        config.tuner().optimize(param_spaces, callback);
    }
    else
    {
        callback(tensor2d_t{1, 0});
    }

    return fit_result;
}
} // namespace nano::ml
