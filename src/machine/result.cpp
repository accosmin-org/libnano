#include <chrono>
#include <filesystem>
#include <nano/core/scat.h>
#include <nano/machine/result.h>

using namespace nano;
using namespace nano::ml;

namespace
{
template <typename... targs>
auto make_random_path(const targs... args)
{
    const auto time = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / scat(time, '_', args..., ".log");
    return path.string();
}
} // namespace

result_t::result_t() = default;

result_t::result_t(param_spaces_t param_spaces, const tensor_size_t folds)
    : m_spaces(std::move(param_spaces))
    , m_params(0, static_cast<tensor_size_t>(m_spaces.size()))
    , m_values(make_full_tensor<scalar_t>(make_dims(0, folds, 2, 2, 12), std::numeric_limits<scalar_t>::quiet_NaN()))
    , m_optims(make_full_tensor<scalar_t>(make_dims(2, 12), std::numeric_limits<scalar_t>::quiet_NaN()))
    , m_refit_log_path(make_random_path("refit"))
{
}

void result_t::add(const tensor2d_t& params_to_try)
{
    assert(params_to_try.size<0>() > 0);
    assert(params_to_try.size<1>() == static_cast<tensor_size_t>(m_spaces.size()));

    const auto trials     = params_to_try.size<0>();
    const auto folds      = this->folds();
    const auto old_trials = this->trials();
    const auto new_trials = old_trials + trials;

    const auto old_params = m_params;
    m_params.resize(old_trials + trials, params_to_try.size<1>());
    m_params.slice(0, old_trials)          = old_params;
    m_params.slice(old_trials, new_trials) = params_to_try;

    const auto old_values = m_values;
    m_values.resize(old_trials + trials, folds, 2, 2, 12);
    m_values.slice(0, old_trials) = old_values;
    m_values.slice(old_trials, new_trials).full(std::numeric_limits<scalar_t>::quiet_NaN());

    for (tensor_size_t fold = 0; fold < folds; ++fold)
    {
        for (tensor_size_t trial = 0; trial < trials; ++trial)
        {
            m_extras.emplace_back();
            m_log_paths.emplace_back(make_random_path("fold", fold, "_trial", old_trials + trial));
        }
    }

    assert(m_extras.size() == static_cast<size_t>(folds * new_trials));
    assert(m_log_paths.size() == static_cast<size_t>(folds * new_trials));
}

tensor_size_t result_t::optimum_trial() const
{
    auto best_trial = tensor_size_t{0};
    auto best_value = std::numeric_limits<scalar_t>::max();

    for (tensor_size_t trial = 0; trial < trials(); ++trial)
    {
        const auto value = this->value(trial);
        if (value < best_value)
        {
            best_trial = trial;
            best_value = value;
        }
    }

    return best_trial;
}

tensor_size_t result_t::closest_trial(tensor1d_cmap_t params, const tensor_size_t max_trials) const
{
    assert(max_trials >= 0 && max_trials <= trials());

    auto best_trial    = tensor_size_t{0};
    auto best_distance = std::numeric_limits<scalar_t>::max();

    for (tensor_size_t trial = 0; trial < max_trials; ++trial)
    {
        const auto distance = (m_params.tensor(trial) - params).lpNorm<2>();
        if (distance < best_distance)
        {
            best_trial    = trial;
            best_distance = distance;
        }
    }

    return best_trial;
}

void result_t::store(tensor2d_t errors_losses, std::any extra)
{
    assert(errors_losses.size<0>() == 2);

    ::store_stats(errors_losses.tensor(0), m_optims.tensor(0));
    ::store_stats(errors_losses.tensor(1), m_optims.tensor(1));

    m_extra = std::move(extra);
}

void result_t::store(const tensor_size_t trial, const tensor_size_t fold, tensor2d_t train_errors_losses,
                     tensor2d_t valid_errors_losses, std::any extra)
{
    assert(fold >= 0 && fold < folds());
    assert(trial >= 0 && trial < trials());
    assert(train_errors_losses.size<0>() == 2);
    assert(valid_errors_losses.size<0>() == 2);

    store_stats(train_errors_losses.tensor(0), m_values.tensor(trial, fold, 0, 0));
    store_stats(train_errors_losses.tensor(1), m_values.tensor(trial, fold, 0, 1));

    store_stats(valid_errors_losses.tensor(0), m_values.tensor(trial, fold, 1, 0));
    store_stats(valid_errors_losses.tensor(1), m_values.tensor(trial, fold, 1, 1));

    m_extras[static_cast<size_t>(trial * folds() + fold)] = std::move(extra);
}

tensor1d_cmap_t result_t::params(const tensor_size_t trial) const
{
    assert(trial >= 0 && trial < trials());

    return m_params.tensor(trial);
}

scalar_t result_t::value(const tensor_size_t trial, const split_type split, const value_type value) const
{
    assert(trial >= 0 && trial < trials());

    auto sum_mean = 0.0;
    for (tensor_size_t fold = 0, folds = this->folds(); fold < folds; ++fold)
    {
        const auto stats = this->stats(trial, fold, split, value);
        sum_mean += stats.m_mean;
    }

    return sum_mean / static_cast<scalar_t>(folds());
}

tensor1d_t result_t::values(const tensor_range_t trial_range, const split_type split, const value_type value) const
{
    tensor1d_t values(trial_range.size());
    for (tensor_size_t trial = trial_range.begin(); trial < trial_range.end(); ++trial)
    {
        values(trial - trial_range.begin()) = this->value(trial, split, value);
    }
    return values;
}

stats_t result_t::stats(const value_type value) const
{
    const auto ivalue = value == value_type::errors ? 0 : 1;

    return load_stats(m_optims.tensor(ivalue));
}

stats_t result_t::stats(const tensor_size_t trial, const tensor_size_t fold, const split_type split,
                        const value_type value) const
{
    assert(fold >= 0 && fold < folds());
    assert(trial >= 0 && trial < trials());

    const auto isplit = split == split_type::train ? 0 : 1;
    const auto ivalue = value == value_type::errors ? 0 : 1;

    return load_stats(m_values.tensor(trial, fold, isplit, ivalue));
}

const std::any& result_t::extra(const tensor_size_t trial, const tensor_size_t fold) const
{
    assert(fold >= 0 && fold < folds());
    assert(trial >= 0 && trial < trials());

    return m_extras[static_cast<size_t>(trial * folds() + fold)];
}

const string_t& result_t::log_path(const tensor_size_t trial, const tensor_size_t fold) const
{
    assert(fold >= 0 && fold < folds());
    assert(trial >= 0 && trial < trials());

    return m_log_paths[static_cast<size_t>(trial * folds() + fold)];
}

const string_t& result_t::refit_log_path() const
{
    return m_refit_log_path;
}

const std::any& result_t::extra() const
{
    return m_extra;
}
