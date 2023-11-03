#pragma once

#include <nano/program/constraint.h>

namespace nano::program
{
///
/// \brief models a linear equality constraint: A * x = b.
///
template <typename tmatrixA, typename tvectorb>
struct equality_t : public constraint_t<tmatrixA, tvectorb>
{
    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(vector_cmap_t x, const scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon()) const
    {
        return deviation(x) < epsilon;
    }

    ///
    /// \brief return the deviation of the given point from the constraint.
    ///
    scalar_t deviation(vector_cmap_t x) const
    {
        return this->valid() ? (this->A() * x.vector() - this->b()).array().abs().maxCoeff()
                             : std::numeric_limits<scalar_t>::max();
    }
};

///
/// \brief create a generic equality constraint: A * x = b.
///
template <typename tmatrixA, typename tvectorb>
inline auto make_equality(tmatrixA A, tvectorb b)
{
    return equality_t<std::remove_cv_t<tmatrixA>, std::remove_cv_t<tvectorb>>{std::move(A), std::move(b)};
}

///
/// \brief create a scalar equality constraint: a.dot(x) = b.
///
template <typename tmatrixA>
inline auto make_equality(const tmatrixA& a, const scalar_t b)
{
    assert(a.cols() == 1);
    return program::make_equality(a.transpose(), vector_t::constant(1, b));
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
