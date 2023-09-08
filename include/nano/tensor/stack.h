#pragma once

#include <nano/tensor/eigen.h>

namespace nano
{
namespace detail
{
template <typename tscalar, typename tblock, typename... tblocks>
void stack(tensor_matrix_t<tscalar>& matrix, const Eigen::Index row, const Eigen::Index col, const tblock& block,
           const tblocks&... blocks)
{
    static_assert(is_eigen_v<tblock>);
    assert(col + block.cols() <= matrix.cols());
    assert(row + block.rows() <= matrix.rows());

    matrix.block(row, col, block.rows(), block.cols()) = block;

    if constexpr (sizeof...(blocks) > 0)
    {
        if (col + block.cols() >= matrix.cols())
        {
            stack<tscalar>(matrix, row + block.rows(), 0, blocks...);
        }
        else
        {
            stack<tscalar>(matrix, row, col + block.cols(), blocks...);
        }
    }
    else
    {
        assert(row + block.rows() == matrix.rows());
        assert(col + block.cols() == matrix.cols());
    }
}

template <typename tscalar, typename tblock, typename... tblocks>
void stack(tensor_vector_t<tscalar>& vector, const Eigen::Index row, const tblock& block, const tblocks&... blocks)
{
    static_assert(is_eigen_v<tblock>);
    assert(block.cols() == 1);

    vector.segment(row, block.rows()) = block;

    if constexpr (sizeof...(blocks) > 0)
    {
        stack<tscalar>(vector, row + block.rows(), blocks...);
    }
    else
    {
        assert(row + block.rows() == vector.rows());
    }
}
} // namespace detail

///
/// \brief stack the given Eigen blocks (e.g. matrices, vectors, operators) in a matrix with the given size.
///
/// NB: the blocks are given in row-major fashion and are assumed to be compatible in size and without gaps.
///
/// example with blocks in the order (matrix1, mastrix2, matrix3, matrix4, matrix5, transposed_vector):
///  +--------------------|---------------+
///  |                    |               |
///  |     matrix1        |    matrix2    |
///  |                    |               |
///  +------------|-------|-----|---------+
///  |            |             |         |
///  |  matrix3   |   matrix4   | matrix5 |
///  |            |             |         |
///  +------------------------------------+
///  |          transposed_vector         |
///  +------------------------------------+
///
template <typename tscalar, typename... tblocks>
typename std::enable_if<(is_eigen_v<tblocks> && ...), tensor_matrix_t<tscalar>>::type
stack(const Eigen::Index rows, const Eigen::Index cols, const tblocks&... blocks)
{
    auto matrix = tensor_matrix_t<tscalar>(rows, cols);

    detail::stack<tscalar>(matrix, 0, 0, blocks...);

    return matrix;
}

///
/// \brief stack the given Eigen segments (e.g. vectors, operators) in a vector with the given size.
///
/// NB: the segments are given in row-major fashion and are assumed to be compatible in size and without gaps.
///
template <typename tscalar, typename... tblocks>
typename std::enable_if<(is_eigen_v<tblocks> && ...), tensor_vector_t<tscalar>>::type stack(const Eigen::Index rows,
                                                                                            const tblocks&... blocks)
{
    auto vector = tensor_vector_t<tscalar>(rows);

    detail::stack<tscalar>(vector, 0, blocks...);

    return vector;
}
} // namespace nano
