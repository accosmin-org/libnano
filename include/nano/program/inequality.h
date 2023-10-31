#pragma once

#include <nano/tensor.h>

namespace nano::program
{
///
/// \brief models a linear equality constraint: A * x <= b.
///
template <typename tmatrixA, typename tvectorb>
struct inequality_t
{
    static_assert(is_eigen_v<tmatrixA> || is_tensor_v<tmatrixA>);
    static_assert(is_eigen_v<tvectorb> || is_tensor_v<tvectorb>);

    ///
    /// \brief return true if the constraint is given.
    ///
    bool valid() const { return (m_A.size() > 0 && m_b.size() > 0) && m_A.rows() == m_b.size(); }

    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    template <typename tvector, std::enable_if_t<is_eigen_v<tvector>, bool> = true>
    bool feasible(const tvector& x, const scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon()) const
    {
        return deviation(x) < epsilon;
    }

    ///
    /// \brief return the deviation of the given point from the constraint.
    ///
    template <typename tvector, std::enable_if_t<is_eigen_v<tvector>, bool> = true>
    scalar_t deviation(const tvector& x) const
    {
        return valid() ? (m_A * x - m_b).array().maxCoeff() : std::numeric_limits<scalar_t>::max();
    }

    // attributes
    tmatrixA m_A; ///<
    tvectorb m_b; ///<
};

///
/// \brief create a generic inequality constraint: A * x <= b.
///
template <typename tmatrixA, typename tvectorb>
inline auto make_inequality(tmatrixA A, tvectorb b)
{
    return inequality_t<std::remove_cv_t<tmatrixA>, std::remove_cv_t<tvectorb>>{std::move(A), std::move(b)};
}

///
/// \brief create a scalar inequality constraint: a.dot(x) <= b.
///
template <typename tmatrixA>
inline auto make_inequality(const tmatrixA& a, const scalar_t b)
{
    assert(a.cols() == 1);
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

template <typename tmatrixA, typename tvectorb>
struct is_inequality<inequality_t<tmatrixA, tvectorb>> : std::true_type
{
};

template <class T>
inline constexpr bool is_inequality_v = is_inequality<T>::value;
} // namespace nano::program
