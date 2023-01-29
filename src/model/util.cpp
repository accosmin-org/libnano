#include <nano/model/util.h>

using namespace nano;

const std::any& ml::closest_extra(const fit_result_t& fit_result, const tensor1d_cmap_t& params,
                                  const tensor_size_t fold)
{
    static const auto extra0  = std::any{};
    const auto* const closest = fit_result.closest(params);

    return closest != nullptr ? closest->extra(fold) : extra0;
}

fit_result_t::params_t ml::make_param_results(const tensor2d_t& all_params, const tensor_size_t folds)
{
    const auto trials = all_params.size<0>();

    auto param_results = fit_result_t::params_t{};
    param_results.reserve(static_cast<size_t>(trials));
    for (tensor_size_t trial = 0; trial < trials; ++trial)
    {
        param_results.emplace_back(all_params.tensor(trial), folds);
    }

    return param_results;
}
