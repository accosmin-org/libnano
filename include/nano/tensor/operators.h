#pragma once

#include <nano/tensor/pprint.h>

namespace nano
{
///
/// \brief pretty-print the given tensor.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
std::ostream& operator<<(std::ostream& stream, const tensor_t<tstorage, tscalar, trank>& tensor)
{
    return pprint(stream, tensor);
}

///
/// \brief compare two tensors element-wise.
///
template <template <typename, size_t> class tstorage1, template <typename, size_t> class tstorage2, typename tscalar,
          size_t trank>
bool operator==(const tensor_t<tstorage1, tscalar, trank>& lhs, const tensor_t<tstorage2, tscalar, trank>& rhs)
{
    return lhs.dims() == rhs.dims() && lhs.vector() == rhs.vector();
}

///
/// \brief compare two tensors element-wise.
///
template <template <typename, size_t> class tstorage1, template <typename, size_t> class tstorage2, typename tscalar,
          size_t trank>
bool operator!=(const tensor_t<tstorage1, tscalar, trank>& lhs, const tensor_t<tstorage2, tscalar, trank>& rhs)
{
    return lhs.dims() != rhs.dims() || lhs.vector() != rhs.vector();
}

///
/// \brief mathematical operators for 1D tensors that return Eigen vector expressions.
///
template <template <typename, size_t> class tstorage, typename tscalar, typename tscalar_factor,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool>        = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar_factor>, bool> = true>
auto operator/(const tensor_t<tstorage, tscalar, 1U>& lhs, const tscalar_factor factor)
{
    return lhs.vector() / static_cast<tscalar>(factor);
}

template <template <typename, size_t> class tstorage, typename tscalar, typename tscalar_factor,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool>        = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar_factor>, bool> = true>
auto operator*(const tensor_t<tstorage, tscalar, 1U>& lhs, const tscalar_factor factor)
{
    return lhs.vector() * static_cast<tscalar>(factor);
}

template <typename tscalar_factor, template <typename, size_t> class tstorage, typename tscalar,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool>        = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar_factor>, bool> = true>
auto operator*(const tscalar_factor factor, const tensor_t<tstorage, tscalar, 1U>& lhs)
{
    return lhs.vector() * static_cast<tscalar>(factor);
}

template <template <typename, size_t> class tstorage_lhs, typename tscalar,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator-(const tensor_t<tstorage_lhs, tscalar, 1U>& lhs)
{
    return -lhs.vector();
}

template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator-(const tensor_t<tstorage_lhs, tscalar, 1U>& lhs, const tensor_t<tstorage_rhs, tscalar, 1U>& rhs)
{
    return lhs.vector() - rhs.vector();
}

template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator+(const tensor_t<tstorage_lhs, tscalar, 1U>& lhs, const tensor_t<tstorage_rhs, tscalar, 1U>& rhs)
{
    return lhs.vector() + rhs.vector();
}

template <template <typename, size_t> class tstorage, typename tscalar, typename texpression,
          std::enable_if_t<is_eigen_v<texpression>, bool>       = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator-(const tensor_t<tstorage, tscalar, 1U>& lhs, const texpression& expression)
{
    return lhs.vector() - expression;
}

template <template <typename, size_t> class tstorage, typename tscalar, typename texpression,
          std::enable_if_t<is_eigen_v<texpression>, bool>       = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator+(const tensor_t<tstorage, tscalar, 1U>& lhs, const texpression& expression)
{
    return lhs.vector() + expression;
}

template <typename texpression, template <typename, size_t> class tstorage, typename tscalar,
          std::enable_if_t<is_eigen_v<texpression>, bool>       = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator-(const texpression& expression, const tensor_t<tstorage, tscalar, 1U>& rhs)
{
    return expression - rhs.vector();
}

template <typename texpression, template <typename, size_t> class tstorage, typename tscalar,
          std::enable_if_t<is_eigen_v<texpression>, bool>       = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator+(const texpression& expression, const tensor_t<tstorage, tscalar, 1U>& rhs)
{
    return expression + rhs.vector();
}

///
/// \brief mathematical operators for 2D tensors that return Eigen matrix expressions.
///
template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator-(const tensor_t<tstorage_lhs, tscalar, 2U>& lhs, const tensor_t<tstorage_rhs, tscalar, 2U>& rhs)
{
    return lhs.matrix() - rhs.matrix();
}

template <template <typename, size_t> class tstorage, typename tscalar, typename texpression,
          std::enable_if_t<is_eigen_v<texpression>, bool>       = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator-(const tensor_t<tstorage, tscalar, 2U>& lhs, const texpression& expression)
{
    return lhs.matrix() - expression;
}

template <template <typename, size_t> class tstorage, typename tscalar, typename texpression,
          std::enable_if_t<is_eigen_v<texpression>, bool>       = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator*(const tensor_t<tstorage, tscalar, 2U>& lhs, const texpression& expression)
{
    return lhs.matrix() * expression;
}

template <typename texpression, template <typename, size_t> class tstorage, typename tscalar,
          std::enable_if_t<is_eigen_v<texpression>, bool>       = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator*(const texpression& expression, const tensor_t<tstorage, tscalar, 1U>& rhs)
{
    return expression * rhs.vector();
}

template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator*(const tensor_t<tstorage_lhs, tscalar, 2U>& lhs, const tensor_t<tstorage_rhs, tscalar, 1U>& rhs)
{
    return lhs.matrix() * rhs.vector();
}

template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
auto operator*(const tensor_t<tstorage_lhs, tscalar, 2U>& lhs, const tensor_t<tstorage_rhs, tscalar, 2U>& rhs)
{
    return lhs.matrix() * rhs.matrix();
}
} // namespace nano
