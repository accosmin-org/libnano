#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
namespace detail
{
template <typename ttensor, typename... ttensors>
void copy(const tensor_size_t index, ttensor& tensor, ttensors&&... tensors)
{
    assert(index + 1 < tensor.template size<0>());

    if constexpr (ttensor::rank() == 1U)
    {
        tensor(index) = tensor(index + 1);
    }
    else
    {
        tensor.tensor(index) = tensor.tensor(index + 1);
    }

    if constexpr (sizeof...(tensors) > 0)
    {
        copy(index, tensors...);
    }
}
} // namespace detail

///
/// \brief remove all sub-tensors indexed by the first dimension as selected by the given operator,
///     compact the remaining ones from the begining and return their size.
///
template <typename toperator, typename ttensor, typename... ttensors>
auto remove_if(const toperator& op, ttensor&& tensor, ttensors&&... tensors) ->
    typename std::enable_if_t<is_tensor_v<ttensor> && (... && is_tensor_v<ttensors>), tensor_size_t>
{
    auto index = tensor_size_t{0};
    for (auto size = tensor.template size<0>(); index < size;)
    {
        if (op(index))
        {
            if (index + 1 < size)
            {
                detail::copy(index, tensor, tensors...);
            }
            --size;
        }
        else
        {
            ++index;
        }
    }

    return index;
}
} // namespace nano
