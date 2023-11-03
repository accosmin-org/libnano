#pragma once

#include <nano/tensor.h>

namespace nano::program
{
///
/// \brief models a linear constraint: A * x ? b.
///
template <typename tmatrixA, typename tvectorb>
struct constraint_t
{
    static_assert(is_eigen_v<tmatrixA> || is_tensor_v<tmatrixA>);
    static_assert(is_eigen_v<tvectorb> || is_tensor_v<tvectorb>);

    ///
    /// \brief return true if the constraint is given.
    ///
    bool valid() const { return (m_A.size() > 0 && m_b.size() > 0) && m_A.rows() == m_b.size(); }

    ///
    /// \brief return an Eigen matrix expression of `A`.
    ///
    auto A() const
    {
        if constexpr (is_tensor_v<tmatrixA>)
        {
            return m_A.matrix();
        }
        else
        {
            return m_A;
        }
    }

    ///
    /// \brief return an Eigen vector expression of `b`.
    ///
    auto b() const
    {
        if constexpr (is_tensor_v<tvectorb>)
        {
            return m_b.vector();
        }
        else
        {
            return m_b;
        }
    }

    // attributes
    tmatrixA m_A; ///<
    tvectorb m_b; ///<
};
} // namespace nano::program
