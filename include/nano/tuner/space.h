#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>

namespace nano
{
class param_space_t;
using param_spaces_t = std::vector<param_space_t>;

///
/// \brief models the hyper-parameters in machine learning applications.
///
/// NB: the hyper-parameter values are restricted to a fixed grid of values.
/// NB: the grid values can be mapped to and from the continuous space [0, 1] of the surrogate smooth models.
///
class NANO_PUBLIC param_space_t
{
public:
    enum class type : uint8_t
    {
        log10,  ///< logarithmic mapping to [0, 1]
        linear, ///< linear mapping to [0, 1]
    };

    ///
    /// \brief constructor
    ///
    param_space_t(type, tensor1d_t grid_values);

    ///
    /// \brief constructor
    ///
    template <class... tscalars>
    explicit param_space_t(type type_, const tscalars... scalars)
        : param_space_t(type_, make_grid_values(scalars...))
    {
    }

    ///
    /// \brief map a hyper-parameter value to the surrogate space [0, 1].
    ///
    scalar_t to_surrogate(scalar_t value) const;

    ///
    /// \brief map from the surrogate space [0, 1] to a hyper-parameter value.
    ///
    scalar_t from_surrogate(scalar_t value) const;

    ///
    /// \brief returns the closest grid point from the surrogate space [0, 1] to a grid cell.
    ///
    tensor_size_t closest_grid_point_from_surrogate(scalar_t value) const;

    ///
    /// \brief returns the closest grid value from the surrogate space [0, 1].
    ///
    scalar_t closest_grid_value_from_surrogate(scalar_t value) const;

    ///
    /// \brief returns the grid of values.
    ///
    const tensor1d_t& values() const { return m_grid_values; }

private:
    static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

    template <class... tscalars>
    static auto make_grid_values(const tscalars... scalars)
    {
        const auto size = static_cast<tensor_size_t>(sizeof...(scalars));
        return make_tensor<scalar_t>(make_dims(size), scalars...);
    }

    // attributes
    type       m_type{type::linear}; ///<
    tensor1d_t m_grid_values;        ///<
    scalar_t   m_min{NaN};           ///<
    scalar_t   m_max{NaN};           ///<
};

template <class... tscalars>
auto make_param_space(const param_space_t::type type, const tscalars... scalars)
{
    return param_space_t{type, scalars...};
}
} // namespace nano
