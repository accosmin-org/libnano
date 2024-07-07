#pragma once

#include <nano/core/parallel.h>
#include <nano/mlearn/params.h>
#include <nano/mlearn/result.h>

namespace nano::ml
{
///
/// \brief tune hyper-parameters required to fit a machine learning model.
///
/// NB: each set of hyper-parameter values is evaluated using the given callback.
/// NB: the tuning is performed in parallel across the current set of hyper-parameter values to evaluate and the folds.
///
template <class tevaluator>
result_t tune(const string_t& prefix, const indices_t& samples, const params_t& fit_params, param_spaces_t param_spaces,
              const tevaluator& evaluator)
{
    const auto splits = fit_params.splitter().split(samples);
    const auto folds  = static_cast<tensor_size_t>(splits.size());

    auto tpool  = parallel::pool_t{};
    auto result = result_t{std::move(param_spaces), folds};

    // tune hyper-parameters (if any) in parallel by hyper-parameter trials and folds
    const auto callback = [&](const tensor2d_t& new_params)
    {
        const auto old_trials = result.trials();
        const auto new_trials = new_params.size<0>();

        result.add(new_params);

        const auto thread_callback = [&](const tensor_size_t index, size_t)
        {
            const auto fold  = index % folds;
            const auto trial = index / folds;

            const auto params        = new_params.tensor(trial);
            const auto closest_trial = result.closest_trial(params, old_trials);

            const auto& [tr_samples, vd_samples] = splits[static_cast<size_t>(fold)];

            auto [tr_values, vd_values, extra] =
                evaluator(tr_samples, vd_samples, params, result.extra(closest_trial, fold));

            result.store(old_trials + trial, fold, std::move(tr_values), std::move(vd_values), std::move(extra));
        };

        tpool.map(folds * new_trials, thread_callback);

        fit_params.log(result, prefix);

        tensor1d_t values{new_trials};
        for (tensor_size_t trial = 0; trial < new_trials; ++trial)
        {
            values(trial) = result.value(old_trials + trial);
        }
        return values;
    };

    if (!result.param_spaces().empty())
    {
        fit_params.tuner().optimize(result.param_spaces(), callback);
    }
    else
    {
        callback(tensor2d_t{1, 0});
    }

    return result;
}
} // namespace nano::ml
