#include <nano/mlearn/result.h>

using namespace nano;
using namespace nano::ml;

result_t::param_t::param_t(tensor1d_t params, const tensor_size_t folds)
    : m_params(std::move(params))
    , m_values(folds, 2, 2, 12)
    , m_extras(static_cast<size_t>(folds))
{
    m_values.full(std::numeric_limits<scalar_t>::quiet_NaN());
}

void result_t::param_t::evaluate(const tensor_size_t fold, tensor2d_t train_errors_losses,
                                 tensor2d_t valid_errors_losses, std::any extra)
{
    assert(fold >= 0 && fold < folds());

    store_stats(train_errors_losses.tensor(0), m_values.tensor(fold, 0, 0));
    store_stats(train_errors_losses.tensor(1), m_values.tensor(fold, 0, 1));

    store_stats(valid_errors_losses.tensor(0), m_values.tensor(fold, 1, 0));
    store_stats(valid_errors_losses.tensor(1), m_values.tensor(fold, 1, 1));

    m_extras[static_cast<size_t>(fold)] = std::move(extra);
}

stats_t result_t::param_t::stats(const tensor_size_t fold, const split_type split, const value_type value) const
{
    assert(fold >= 0 && fold < folds());

    const auto split_index = split == split_type::train ? 0 : 1;
    const auto value_index = value == value_type::errors ? 0 : 1;

    return load_stats(m_values.tensor(fold, split_index, value_index));
}

scalar_t result_t::param_t::value(const split_type split, const value_type value) const
{
    auto sum_mean = 0.0;
    for (tensor_size_t fold = 0, folds = this->folds(); fold < folds; ++fold)
    {
        const auto stats = this->stats(fold, split, value);
        sum_mean += stats.m_mean;
    }

    return std::log(sum_mean + std::numeric_limits<scalar_t>::epsilon());
}

const std::any& result_t::param_t::extra(const tensor_size_t fold) const
{
    assert(fold >= 0 && fold < folds());

    return m_extras[static_cast<size_t>(fold)];
}

result_t::result_t(strings_t param_names)
    : m_param_names(std::move(param_names))
    , m_optim_values(make_full_tensor<scalar_t>(make_dims(2, 12), std::numeric_limits<scalar_t>::quiet_NaN()))
{
}

void result_t::add(param_t param)
{
    assert(param.params().size() == static_cast<tensor_size_t>(m_param_names.size()));

    m_param_results.emplace_back(std::move(param));
}

const result_t::param_t& result_t::optimum() const
{
    static const auto noparams = param_t{};

    if (m_param_results.empty())
    {
        assert(m_param_names.empty());

        return noparams;
    }
    else
    {
        const auto it = std::min_element(m_param_results.begin(), m_param_results.end());
        return *it;
    }
}

void result_t::evaluate(tensor2d_t errors_losses)
{
    ::store_stats(errors_losses.tensor(0), m_optim_values.tensor(0));
    ::store_stats(errors_losses.tensor(1), m_optim_values.tensor(1));
}

stats_t result_t::stats(const value_type value) const
{
    const auto value_index = value == value_type::errors ? 0 : 1;

    return load_stats(m_optim_values.tensor(value_index));
}

const result_t::param_t* result_t::closest(const tensor1d_cmap_t& params) const
{
    const result_t::param_t* closest = nullptr;

    auto distance = std::numeric_limits<scalar_t>::max();
    for (const auto& param_result : m_param_results)
    {
        const auto distance_ = (param_result.params().vector() - params.vector()).lpNorm<2>();
        if (distance_ < distance)
        {
            closest  = &param_result;
            distance = distance_;
        }
    }

    return closest;
}
