#pragma once

#include <nano/tensor.h>

namespace nano::program
{
///
/// \brief models a linear constraint: A * x ? b.
///
template <class tmatrixA, class tvectorb>
struct constraint_t
{
    static_assert(is_eigen_v<tmatrixA> || is_tensor_v<tmatrixA>);
    static_assert(is_eigen_v<tvectorb> || is_tensor_v<tvectorb>);

    ///
    /// \brief return true if the constraint is given.
    ///
    bool valid() const { return (m_A.size() > 0 && m_b.size() > 0) && m_A.rows() == m_b.size(); }

    // attributes
    tmatrixA m_A; ///<
    tvectorb m_b; ///<
};
} // namespace nano::program
