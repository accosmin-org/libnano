#include <nano/core/parallel.h>
#include <nano/machine/params.h>
#include <nano/machine/result.h>
#include <nano/machine/tune.h>

using namespace nano::ml;

result_t nano::ml::tune(const string_t& prefix, const indices_t& samples, const params_t& fit_params,
                        param_spaces_t param_spaces, const tune_callback_t& callback)
{
    const auto splits = fit_params.splitter().split(samples);
    const auto folds  = static_cast<tensor_size_t>(splits.size());

    auto tpool  = parallel::pool_t{};
    auto result = result_t{std::move(param_spaces), folds};

    // tune hyper-parameters (if any) in parallel by hyper-parameter trials and folds
    const auto tuner_callback = [&](const tensor2d_t& new_params)
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
            const auto logger                    = make_file_logger(result.log_path(old_trials + trial, fold));
            const auto closest                   = result.extra(closest_trial, fold);

            auto [tr_values, vd_values, extra] = callback(tr_samples, vd_samples, params, closest, logger);

            result.store(old_trials + trial, fold, std::move(tr_values), std::move(vd_values), std::move(extra));
        };

        tpool.map(folds * new_trials, thread_callback);

        fit_params.log(result, old_trials, prefix);

        return result.values(make_range(old_trials, old_trials + new_trials));
    };

    if (!result.param_spaces().empty())
    {
        fit_params.tuner().optimize(result.param_spaces(), tuner_callback, fit_params.logger());
    }
    else
    {
        tuner_callback(tensor2d_t{1, 0});
    }

    return result;
}
