#pragma once

#include <nano/program/constraint.h>

namespace nano::program
{
///
/// \brief models a linear equality constraint: A * x <= b.
///
template <class tmatrixA, class tvectorb>
struct inequality_t : public constraint_t<tmatrixA, tvectorb>
{
    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(vector_cmap_t x, const scalar_t epsilon = epsilon0<scalar_t>()) const
    {
        return deviation(x) < epsilon;
    }

    ///
    /// \brief return the deviation of the given point from the constraint.
    ///
    scalar_t deviation(vector_cmap_t x) const
    {
        return this->valid() ? (this->m_A * x - this->m_b).array().maxCoeff() : std::numeric_limits<scalar_t>::max();
    }
};

///
/// \brief create a generic inequality constraint: A * x <= b.
///
template <class tmatrixA, class tvectorb, std::enable_if_t<is_eigen_v<tmatrixA> || is_tensor_v<tmatrixA>, bool> = true,
          std::enable_if_t<is_eigen_v<tvectorb> || is_tensor_v<tvectorb>, bool> = true>
inline auto make_inequality(tmatrixA A, tvectorb b)
{
    if constexpr (is_tensor_v<tmatrixA>)
    {
        static_assert(tmatrixA::rank() == 2U);
    }
    if constexpr (is_tensor_v<tvectorb>)
    {
        static_assert(tvectorb::rank() == 1U);
    }
    return inequality_t<std::remove_cv_t<tmatrixA>, std::remove_cv_t<tvectorb>>{std::move(A), std::move(b)};
}

///
/// \brief create a scalar inequality constraint: a.dot(x) <= b.
///
template <class tvectora, std::enable_if_t<is_eigen_v<tvectora> || is_tensor_v<tvectora>, bool> = true>
inline auto make_inequality(const tvectora& a, const scalar_t b)
{
    if constexpr (is_eigen_v<tvectora>)
    {
        assert(a.cols() == 1);
    }
    else
    {
        static_assert(tvectora::rank() == 1U);
    }
    return make_inequality(a.transpose(), vector_t::constant(1, b));
}

///
/// \brief create a one-sided inequality contraint: x <= upper (element-wise).
///
inline auto make_less(const tensor_size_t dims, const scalar_t upper)
{
    return make_inequality(matrix_t::identity(dims, dims), vector_t::constant(dims, upper));
}

inline auto make_less(const vector_t& upper)
{
    const auto dims = upper.size();
    return make_inequality(matrix_t::identity(dims, dims), upper);
}

///
/// \brief create a one-sided inequality contraint: lower <= x (element-wise).
///
inline auto make_greater(const tensor_size_t dims, const scalar_t lower)
{
    return make_inequality(-matrix_t::identity(dims, dims), -vector_t::constant(dims, lower));
}

inline auto make_greater(const vector_t& lower)
{
    const auto dims = lower.size();
    return make_inequality(-matrix_t::identity(dims, dims), -lower.vector());
}

///
/// \brief traits to check if a given type is a linear inequality constraint.
///
template <class T>
struct is_inequality : std::false_type
{
};

template <class tmatrixA, class tvectorb>
struct is_inequality<inequality_t<tmatrixA, tvectorb>> : std::true_type
{
};

template <class T>
inline constexpr bool is_inequality_v = is_inequality<T>::value;
} // namespace nano::program
