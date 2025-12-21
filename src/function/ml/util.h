#pragma once

#include <nano/enum.h>
#include <nano/tensor.h>

namespace nano
{
enum class optimization_type : uint8_t
{
    constrained,
    unconstrained,
};

template <>
inline enum_map_t<optimization_type> enum_string<optimization_type>()
{
    return {
        {  optimization_type::constrained,   "constrained"},
        {optimization_type::unconstrained, "unconstrained"},
    };
}

inline tensor_size_t make_size(const tensor_size_t     dims,
                               const optimization_type type = optimization_type::unconstrained)
{
    const auto size = std::max(dims, tensor_size_t{2});

    switch (type)
    {
    case optimization_type::unconstrained:
        // solve for (x,)
        return size;

    default:
        // solve for (x, z)
        return 2 * size;
    }
}

inline tensor_size_t make_inputs(const tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

inline tensor_size_t make_outputs([[maybe_unused]] const tensor_size_t dims)
{
    return tensor_size_t{1};
}

inline tensor_size_t make_samples(const tensor_size_t dims, const scalar_t sratio)
{
    return static_cast<tensor_size_t>(std::max(sratio * static_cast<scalar_t>(dims), 10.0));
}
} // namespace nano
