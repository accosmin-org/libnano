#include <nano/solver.h>
#include <nano/model/tuner.h>
#include <nano/core/logger.h>
#include <nano/model/surrogate.h>

using namespace nano;

static auto make_min(const tensor1d_t& grid_values)
{
    return (grid_values.size() < 1) ?
        std::numeric_limits<scalar_t>::quiet_NaN() :
        *std::min_element(begin(grid_values), end(grid_values));
}

static auto make_max(const tensor1d_t& grid_values)
{
    return (grid_values.size() < 1) ?
        std::numeric_limits<scalar_t>::quiet_NaN() :
        *std::max_element(begin(grid_values), end(grid_values));
}

param_space_t::param_space_t(param_space_t::type type, tensor1d_t grid_values) :
    m_type(type),
    m_grid_values(std::move(grid_values)),
    m_min(make_min(m_grid_values)),
    m_max(make_max(m_grid_values))
{
    critical(
        m_grid_values.size() < 2,
        "parameter space: at least two grid values must be given!");

    critical(
        !std::is_sorted(begin(m_grid_values), end(m_grid_values)),
        "parameter space: the grid values must be sorted!");

    critical(
        std::unique(begin(m_grid_values), end(m_grid_values)) != end(m_grid_values),
        "parameter space: the grid values must be distinct!");

    critical(
        m_type == type::log10 &&
        *std::min_element(begin(m_grid_values), end(m_grid_values)) < epsilon0<scalar_t>(),
        "parameter space: the grid values must be strictly positive if using the logarithmic scale!");
}

scalar_t param_space_t::to_surrogate(scalar_t value) const
{
    critical(
        value < m_min || value > m_max,
        "parameter space: cannot map value (" , value, ") outside the parameter grid range [", m_min, ",", m_max, "]!");

    switch (m_type)
    {
    case type::linear:
        return (value - m_min) / (m_max - m_min);

    default:
        return std::log10(value);
    }
}

scalar_t param_space_t::from_surrogate(scalar_t value) const
{
    switch (m_type)
    {
    case type::linear:
        return std::clamp(m_min + value * (m_max - m_min), m_min, m_max);

    default:
        return std::clamp(std::pow(10.0, value), m_min, m_max);
    }
}

scalar_t param_space_t::closest_grid_value_from_surrogate(scalar_t value) const
{
    scalar_t min_distance = std::numeric_limits<scalar_t>::max(), closest_grid_value = value;
    for (const auto grid_value : m_grid_values)
    {
        const auto distance = std::fabs(to_surrogate(grid_value) - value);
        if (distance < min_distance)
        {
            min_distance = distance;
            closest_grid_value = grid_value;
        }
    }

    return closest_grid_value;
}
