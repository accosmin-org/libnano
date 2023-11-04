#pragma once

#include <nano/core/numeric.h>
#include <nano/tensor/eigen.h>
#include <nano/tensor/index.h>

namespace nano
{
template <template <typename, size_t> class, typename, size_t>
class tensor_t;

///
/// \brief returns true if the two tensors are close, ignoring not-finite values if present.
///
template <template <typename, size_t> class tstorage1, template <typename, size_t> class tstorage2, typename tscalar,
          size_t trank>
bool close(const tensor_t<tstorage1, tscalar, trank>& lhs, const tensor_t<tstorage2, tscalar, trank>& rhs,
           const double epsilon)
{
    if (lhs.dims() != rhs.dims())
    {
        return false;
    }
    for (tensor_size_t i = 0, size = lhs.size(); i < size; ++i)
    {
        const auto lhs_finite = ::nano::isfinite(lhs(i));
        const auto rhs_finite = ::nano::isfinite(rhs(i));
        if ((lhs_finite != rhs_finite) ||
            (lhs_finite && !close(static_cast<double>(lhs(i)), static_cast<double>(rhs(i)), epsilon)))
        {
            return false;
        }
    }
    return true;
}

///
/// \brief compare two tensors element-wise.
///
template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, size_t trank>
bool operator==(const tensor_t<tstorage_lhs, tscalar, trank>& lhs, const tensor_t<tstorage_rhs, tscalar, trank>& rhs)
{
    return lhs.dims() == rhs.dims() && lhs.vector() == rhs.vector();
}

///
/// \brief compare two tensors element-wise.
///
template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, size_t trank>
bool operator!=(const tensor_t<tstorage_lhs, tscalar, trank>& lhs, const tensor_t<tstorage_rhs, tscalar, trank>& rhs)
{
    return lhs.dims() != rhs.dims() || lhs.vector() != rhs.vector();
}

///
/// \brief divide the tensor element-wise by the given factor and return the associated Eigen expression.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank, typename tscalar_factor,
          std::enable_if_t<std::is_arithmetic_v<tscalar_factor>, bool> = true>
auto operator/(const tensor_t<tstorage, tscalar, trank>& lhs, const tscalar_factor factor)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return lhs.vector() / static_cast<tscalar>(factor);
    }
    else
    {
        return lhs.matrix() / static_cast<tscalar>(factor);
    }
}

///
/// \brief multiply the tensor element-wise with the given factor and return the associated Eigen expression.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank, typename tscalar_factor,
          std::enable_if_t<std::is_arithmetic_v<tscalar_factor>, bool> = true>
auto operator*(const tensor_t<tstorage, tscalar, trank>& lhs, const tscalar_factor factor)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return lhs.vector() * static_cast<tscalar>(factor);
    }
    else
    {
        return lhs.matrix() * static_cast<tscalar>(factor);
    }
}

template <typename tscalar_factor, template <typename, size_t> class tstorage, typename tscalar, size_t trank,
          std::enable_if_t<std::is_arithmetic_v<tscalar_factor>, bool> = true>
auto operator*(const tscalar_factor factor, const tensor_t<tstorage, tscalar, trank>& rhs)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return static_cast<tscalar>(factor) * rhs.vector();
    }
    else
    {
        return static_cast<tscalar>(factor) * rhs.matrix();
    }
}

///
/// \brief negate the tensor element-wise and return the associated Eigen expression.
///
template <template <typename, size_t> class tstorage_lhs, typename tscalar, size_t trank>
auto operator-(const tensor_t<tstorage_lhs, tscalar, trank>& lhs)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return -lhs.vector();
    }
    else
    {
        return -lhs.matrix();
    }
}

///
/// \brief subtract the two tensors or Eigen expressions and return the associated Eigen expression.
///
template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, size_t trank>
auto operator-(const tensor_t<tstorage_lhs, tscalar, trank>& lhs, const tensor_t<tstorage_rhs, tscalar, trank>& rhs)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return lhs.vector() - rhs.vector();
    }
    else
    {
        return lhs.matrix() - rhs.matrix();
    }
}

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank, typename texpression,
          std::enable_if_t<is_eigen_v<texpression>, bool> = true>
auto operator-(const tensor_t<tstorage, tscalar, trank>& lhs, const texpression& expression)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return lhs.vector() - expression;
    }
    else
    {
        return lhs.matrix() - expression;
    }
}

template <typename texpression, template <typename, size_t> class tstorage, typename tscalar, size_t trank,
          std::enable_if_t<is_eigen_v<texpression>, bool> = true>
auto operator-(const texpression& expression, const tensor_t<tstorage, tscalar, trank>& lhs)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return expression - lhs.vector();
    }
    else
    {
        return expression - lhs.matrix();
    }
}

///
/// \brief add the two tensors or Eigen expressions and return the associated Eigen expression.
///
template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, size_t trank>
auto operator+(const tensor_t<tstorage_lhs, tscalar, trank>& lhs, const tensor_t<tstorage_rhs, tscalar, trank>& rhs)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return lhs.vector() + rhs.vector();
    }
    else
    {
        return lhs.matrix() + rhs.matrix();
    }
}

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank, typename texpression,
          std::enable_if_t<is_eigen_v<texpression>, bool> = true>
auto operator+(const tensor_t<tstorage, tscalar, trank>& lhs, const texpression& expression)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return lhs.vector() + expression;
    }
    else
    {
        return lhs.matrix() + expression;
    }
}

template <typename texpression, template <typename, size_t> class tstorage, typename tscalar, size_t trank,
          std::enable_if_t<is_eigen_v<texpression>, bool> = true>
auto operator+(const texpression& expression, const tensor_t<tstorage, tscalar, trank>& lhs)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return expression + lhs.vector();
    }
    else
    {
        return expression + lhs.matrix();
    }
}

///
/// \brief multiply the two tensors or Eigen expressions and return the associated Eigen expression.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank, typename texpression,
          std::enable_if_t<is_eigen_v<texpression>, bool> = true>
auto operator*(const tensor_t<tstorage, tscalar, trank>& lhs, const texpression& expression)
{
    static_assert(trank == 2U);

    return lhs.matrix() * expression;
}

template <typename texpression, template <typename, size_t> class tstorage, typename tscalar, size_t trank,
          std::enable_if_t<is_eigen_v<texpression>, bool> = true>
auto operator*(const texpression& expression, const tensor_t<tstorage, tscalar, trank>& rhs)
{
    static_assert(trank == 1U || trank == 2U);

    if constexpr (trank == 1U)
    {
        return expression * rhs.vector();
    }
    else
    {
        return expression * rhs.matrix();
    }
}

template <template <typename, size_t> class tstorage_lhs, template <typename, size_t> class tstorage_rhs,
          typename tscalar, size_t trank_lhs, size_t trank_rhs>
auto operator*(const tensor_t<tstorage_lhs, tscalar, trank_lhs>& lhs,
               const tensor_t<tstorage_rhs, tscalar, trank_rhs>& rhs)
{
    static_assert(trank_lhs == 2U && (trank_rhs == 1U || trank_rhs == 2U));

    if constexpr (trank_rhs == 1U)
    {
        return lhs.matrix() * rhs.vector();
    }
    else
    {
        return lhs.matrix() * rhs.matrix();
    }
}
} // namespace nano
