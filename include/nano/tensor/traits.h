#pragma once

#include <type_traits>

namespace nano
{
template <template <typename, size_t> class, typename, size_t>
class tensor_t;

///
/// \brief traits to check if a given type is a tensor.
///
template <class T>
struct is_tensor : std::false_type
{
};

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
struct is_tensor<tensor_t<tstorage, tscalar, trank>> : std::true_type
{
};

template <class T>
inline constexpr bool is_tensor_v = is_tensor<T>::value;
} // namespace nano
