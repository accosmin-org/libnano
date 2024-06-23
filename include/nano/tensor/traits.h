#pragma once

#include <type_traits>

namespace nano
{
template <template <class, size_t> class, class, size_t>
class tensor_t;

///
/// \brief traits to check if a given type is a tensor.
///
template <class T>
struct is_tensor : std::false_type
{
};

template <template <class, size_t> class tstorage, class tscalar, size_t trank>
struct is_tensor<tensor_t<tstorage, tscalar, trank>> : std::true_type
{
};

template <class T>
inline constexpr bool is_tensor_v = is_tensor<T>::value;
} // namespace nano
