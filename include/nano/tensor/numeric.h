#pragma once

#include <nano/core/numeric.h>
#include <nano/tensor/eigen.h>
#include <nano/tensor/index.h>

namespace nano
{
template <template <class, size_t> class, class, size_t>
class tensor_t;

///
/// \brief returns true if the two tensors are close, ignoring not-finite values if present.
///
template <template <class, size_t> class tstorage1, template <class, size_t> class tstorage2, class tscalar,
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
template <template <class, size_t> class tstorage_lhs, template <class, size_t> class tstorage_rhs, class tscalar,
          size_t trank>
bool operator==(const tensor_t<tstorage_lhs, tscalar, trank>& lhs, const tensor_t<tstorage_rhs, tscalar, trank>& rhs)
{
    return lhs.dims() == rhs.dims() && lhs.vector() == rhs.vector();
}

///
/// \brief compare two tensors element-wise.
///
template <template <class, size_t> class tstorage_lhs, template <class, size_t> class tstorage_rhs, class tscalar,
          size_t trank>
bool operator!=(const tensor_t<tstorage_lhs, tscalar, trank>& lhs, const tensor_t<tstorage_rhs, tscalar, trank>& rhs)
{
    return lhs.dims() != rhs.dims() || lhs.vector() != rhs.vector();
}

///
/// \brief divide the tensor element-wise by the given factor and return the associated Eigen expression.
///
template <template <class, size_t> class tstorage, class tscalar, size_t trank, class tscalar_factor>
requires std::is_arithmetic_v<tscalar_factor> auto operator/(const tensor_t<tstorage, tscalar, trank>& lhs,
                                                             const tscalar_factor                      factor)
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
template <template <class, size_t> class tstorage, class tscalar, size_t trank, class tscalar_factor>
requires std::is_arithmetic_v<tscalar_factor> auto operator*(const tensor_t<tstorage, tscalar, trank>& lhs,
                                                             const tscalar_factor                      factor)
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

template <class tscalar_factor, template <class, size_t> class tstorage, class tscalar, size_t trank>
requires std::is_arithmetic_v<tscalar_factor> auto operator*(const tscalar_factor                      factor,
                                                             const tensor_t<tstorage, tscalar, trank>& rhs)
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
template <template <class, size_t> class tstorage_lhs, class tscalar, size_t trank>
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
template <template <class, size_t> class tstorage_lhs, template <class, size_t> class tstorage_rhs, class tscalar,
          size_t trank>
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

template <template <class, size_t> class tstorage, class tscalar, size_t trank, class texpression>
requires is_eigen_v<texpression> auto operator-(const tensor_t<tstorage, tscalar, trank>& lhs,
                                                const texpression&                        expression)
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

template <class texpression, template <class, size_t> class tstorage, class tscalar, size_t trank>
requires is_eigen_v<texpression> auto operator-(const texpression&                        expression,
                                                const tensor_t<tstorage, tscalar, trank>& lhs)
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
template <template <class, size_t> class tstorage_lhs, template <class, size_t> class tstorage_rhs, class tscalar,
          size_t trank>
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

template <template <class, size_t> class tstorage, class tscalar, size_t trank, class texpression>
requires is_eigen_v<texpression> auto operator+(const tensor_t<tstorage, tscalar, trank>& lhs,
                                                const texpression&                        expression)
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

template <class texpression, template <class, size_t> class tstorage, class tscalar, size_t trank>
requires is_eigen_v<texpression> auto operator+(const texpression&                        expression,
                                                const tensor_t<tstorage, tscalar, trank>& lhs)
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
template <template <class, size_t> class tstorage, class tscalar, size_t trank, class texpression>
requires is_eigen_v<texpression> auto operator*(const tensor_t<tstorage, tscalar, trank>& lhs,
                                                const texpression&                        expression)
{
    static_assert(trank == 2U);

    return lhs.matrix() * expression;
}

template <class texpression, template <class, size_t> class tstorage, class tscalar, size_t trank>
requires is_eigen_v<texpression> auto operator*(const texpression&                        expression,
                                                const tensor_t<tstorage, tscalar, trank>& rhs)
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

template <template <class, size_t> class tstorage_lhs, template <class, size_t> class tstorage_rhs, class tscalar,
          size_t trank_lhs, size_t trank_rhs>
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
