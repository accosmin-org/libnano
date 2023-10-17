#pragma once

#include <nano/eigen.h>

namespace nano::program
{
///
/// \brief models a linear equality constraint: A * x = b.
///
template <typename tmatrixA, typename tvectorb, std::enable_if_t<is_eigen_v<tmatrixA>, bool> = true,
          std::enable_if_t<is_eigen_v<tvectorb>, bool> = true>
struct equality_t
{
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
        return valid() ? (m_A * x - m_b).array().abs().maxCoeff() : std::numeric_limits<scalar_t>::max();
    }

    // attributes
    tmatrixA m_A; ///<
    tvectorb m_b; ///<
};

///
/// \brief create a generic equality constraint: A * x = b.
///
template <typename tmatrixA, typename tvectorb, std::enable_if_t<is_eigen_v<tmatrixA>, bool> = true,
          std::enable_if_t<is_eigen_v<tvectorb>, bool> = true>
inline auto make_equality(tmatrixA A, tvectorb b)
{
    return equality_t<std::remove_cv_t<tmatrixA>, std::remove_cv_t<tvectorb>>{std::move(A), std::move(b)};
}

///
/// \brief create a scalar equality constraint: a.dot(x) = b.
///
template <typename tmatrixA, std::enable_if_t<is_eigen_v<tmatrixA>, bool> = true>
inline auto make_equality(const tmatrixA& a, const scalar_t b)
{
    assert(a.cols() == 1);
    return program::make_equality(a.transpose(), vector_t::Constant(1, b));
}

///
/// \brief traits to check if a given type is a linear equality constraint.
///
template <class T>
struct is_equality : std::false_type
{
};

template <typename tmatrixA, typename tvectorb>
struct is_equality<equality_t<tmatrixA, tvectorb>> : std::true_type
{
};

template <class T>
inline constexpr bool is_equality_v = is_equality<T>::value;
} // namespace nano::program
