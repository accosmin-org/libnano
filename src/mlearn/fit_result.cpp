#include <nano/mlearn/fit_result.h>

using namespace nano;
using namespace nano::ml;

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
