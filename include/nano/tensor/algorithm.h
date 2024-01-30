#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
namespace detail
{
template <typename ttensor, typename... ttensors>
auto size(const ttensor& tensor, const ttensors&... tensors)
{
    const auto size = tensor.template size<0>();
    (assert(size == tensors.template size<0>()), ...);
    return size;
}

template <typename ttensor>
void copy(const tensor_size_t isrc, const tensor_size_t idst, ttensor& tensor)
{
    assert(isrc < tensor.template size<0>());
    assert(idst < tensor.template size<0>());

    if constexpr (ttensor::rank() == 1U)
    {
        tensor(idst) = tensor(isrc);
    }
    else
    {
        tensor.tensor(idst) = tensor.tensor(isrc);
    }
}
} // namespace detail

///
/// \brief remove all sub-tensors indexed by the first dimension flagged by the given operator.
///
/// NB: the remaining ones are compacted starting the begining and their number is returned.
/// NB: no allocation is performed.
///
template <typename toperator, typename... ttensors>
auto remove_if(const toperator& op, ttensors&... tensors) noexcept ->
    typename std::enable_if_t<(is_tensor_v<ttensors> && ...), tensor_size_t>
{
    const auto size = detail::size(tensors...);

    auto last = tensor_size_t{0};
    for (; last < size && !op(last); ++last)
    {
    }

    for (auto curr = last; curr < size; ++curr)
    {
        if (!op(curr))
        {
            (detail::copy(curr, last, tensors), ...);
            ++last;
        }
    }

    return last;
}
} // namespace nano
