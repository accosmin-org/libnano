#include <nano/mlearn/fit_hparam.h>

using namespace nano;
using namespace nano::ml;

fit_hparam_t::fit_hparam_t(tensor1d_t params, const tensor_size_t folds)
    : m_params(std::move(params))
    , m_values(folds, 2, 2, 12)
    , m_extras(static_cast<size_t>(folds))
{
    m_values.full(std::numeric_limits<scalar_t>::quiet_NaN());
}

void fit_hparam_t::evaluate(const tensor_size_t fold, tensor2d_t train_errors_losses, tensor2d_t valid_errors_losses)
{
    assert(fold >= 0 && fold < folds());

    ::store_stats(train_errors_losses.tensor(0), m_values.tensor(fold, 0, 0));
    ::store_stats(train_errors_losses.tensor(1), m_values.tensor(fold, 0, 1));

    ::store_stats(valid_errors_losses.tensor(0), m_values.tensor(fold, 1, 0));
    ::store_stats(valid_errors_losses.tensor(1), m_values.tensor(fold, 1, 1));

    m_extras[static_cast<size_t>(fold)] = std::move(extra);
}

stats_t fit_hparam_t::stats(const tensor_size_t fold, const split_type split, const value_type value) const
{
    assert(fold >= 0 && fold < folds());

    const auto split_index = split == split_type::train ? 0 : 1;
    const auto value_index = value == value_type::errors ? 0 : 1;

    return load_stats(m_values.tensor(fold, split_index, value_index));
}

scalar_t fit_hparam_t::value(const split_type split, const value_type value) const
{
    auto sum_mean = 0.0;
    for (tensor_size_t fold = 0, folds = this->folds(); fold < folds; ++fold)
    {
        const auto stats = this->stats(fold, split, value);
        sum_mean += stats.m_mean;
    }

    return sum_mean / static_cast<scalar_t>(folds());
}

const string_t& fit_hparam_t::serial_path(const tensor_size_t fold) const
{
    assert(fold >= 0 && fold < folds());

    return m_serial_paths[static_cast<size_t>(fold)];
}

const string_t& fit_hparam_t::logger_path(const tensor_size_t fold) const
{
    assert(fold >= 0 && fold < folds());

    return m_logger_paths[static_cast<size_t>(fold)];
}
