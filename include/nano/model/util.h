#pragma once

#include <nano/core/parallel.h>
#include <nano/model.h>

namespace nano::ml
{
    NANO_PUBLIC const std::any& closest_extra(const fit_result_t&, const tensor1d_cmap_t& params, tensor_size_t fold);

    NANO_PUBLIC fit_result_t::params_t make_param_results(const tensor2d_t& all_params, tensor_size_t folds);

    template <typename tlogger, typename tevaluator>
    auto tune(const indices_t& samples, const splitter_t& splitter, const tuner_t& tuner, strings_t param_names,
              const param_spaces_t& param_spaces, const tlogger& logger, const tevaluator& evaluator)
    {
        // TODO: allocate the iterators once: (train, valid)xfolds + refit
        const auto splits = splitter.split(samples);
        const auto folds  = static_cast<tensor_size_t>(splits.size());

        auto thread_pool = parallel::pool_t{1U};
        auto fit_result  = fit_result_t{std::move(param_names)};

        // tune hyper-parameters (if any) in parallel by hyper-parameter trials and folds
        const auto callback = [&](const tensor2d_t& all_params)
        {
            auto param_results = make_param_results(all_params, folds);

            const auto callback = [&](const tensor_size_t index, size_t)
            {
                const auto fold  = index % folds;
                const auto trial = index / folds;

                const auto  params        = all_params.tensor(trial);
                const auto& closest_extra = ::nano::ml::closest_extra(fit_result, params, fold);

                const auto& [train_samples, valid_samples] = splits[static_cast<size_t>(fold)];

                auto [train_values, valid_values, extra] =
                    evaluator(train_samples, valid_samples, params, closest_extra);

                auto& param_result = param_results[static_cast<size_t>(trial)];
                param_result.evaluate(fold, std::move(train_values), std::move(valid_values), std::move(extra));
            };

            const auto trials = all_params.size<0>();
            thread_pool.map(folds * trials, callback);

            tensor1d_t values{trials};
            for (tensor_size_t trial = 0; trial < trials; ++trial)
            {
                auto& param_result = param_results[static_cast<size_t>(trial)];
                values(trial)      = param_result.value();
                fit_result.add(std::move(param_result));
            }

            logger(fit_result);

            return values;
        };

        if (!param_spaces.empty())
        {
            tuner.optimize(param_spaces, callback);
        }
        else
        {
            callback(tensor2d_t{1, 0});
        }

        return fit_result;
    }
} // namespace nano::ml
