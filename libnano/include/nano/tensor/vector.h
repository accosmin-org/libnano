#pragma once

#include <type_traits>
#include <eigen3/Eigen/Core>

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
    /// \brief map non-constant data to vectors.
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = typename std::remove_const<tscalar_>::type,
        typename tresult = Eigen::Map<tensor_vector_t<tscalar>, alignment>
    >
    tresult map_vector(tscalar_* data, const tsize rows)
    {
        return tresult(data, rows);
    }

    ///
    /// \brief map constant data to vectors
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = typename std::remove_const<tscalar_>::type,
        typename tresult = Eigen::Map<const tensor_vector_t<tscalar>, alignment>
    >
    tresult map_vector(const tscalar_* data, const tsize rows)
    {
        return tresult(data, rows);
    }
}
