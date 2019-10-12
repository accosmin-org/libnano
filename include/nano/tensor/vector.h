#pragma once

#include <Eigen/Core>
#include <type_traits>

namespace nano
{
    ///
    /// \brief vector types.
    ///
    template
    <
        typename tscalar_,
        int trows = Eigen::Dynamic,
        typename tscalar = typename std::remove_const<tscalar_>::type
    >
    using tensor_vector_t = Eigen::Matrix<tscalar, trows, 1, Eigen::ColMajor>;

    ///
    /// \brief map non-constant arrays to vectors.
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = typename std::remove_const<tscalar_>::type,
        typename tresult = Eigen::Map<tensor_vector_t<tscalar>, alignment>
    >
    tresult map_vector(tscalar_* data, const tsize rows) // NOLINT(readability-avoid-const-params-in-decls)
    {
        return tresult(data, rows);
    }

    ///
    /// \brief map constant arrays to vectors.
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = typename std::remove_const<tscalar_>::type,
        typename tresult = Eigen::Map<const tensor_vector_t<tscalar>, alignment>
    >
    tresult map_vector(const tscalar_* data, const tsize rows) // NOLINT(readability-avoid-const-params-in-decls)
    {
        return tresult(data, rows);
    }
}
