#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
namespace detail
{
template <typename tscalar, typename tblock, typename... tblocks>
void stack(tensor_mem_t<tscalar, 2U>& matrix, const tensor_size_t row, const tensor_size_t col, const tblock& block,
           const tblocks&... blocks)
{
    static_assert(is_eigen_v<tblock> || is_tensor_v<tblock>);

    const auto next =
        [&]([[maybe_unused]] const tensor_size_t block_rows, [[maybe_unused]] const tensor_size_t block_cols)
    {
        if constexpr (sizeof...(blocks) > 0)
        {
            if (col + block_cols >= matrix.cols())
            {
                stack(matrix, row + block_rows, 0, blocks...);
            }
            else
            {
                stack(matrix, row, col + block_cols, blocks...);
            }
        }
        else
        {
            assert(row + block_rows == matrix.rows());
            assert(col + block_cols == matrix.cols());
        }
    };

    if constexpr (is_eigen_v<tblock>)
    {
        assert(col + block.cols() <= matrix.cols());
        assert(row + block.rows() <= matrix.rows());

        matrix.block(row, col, block.rows(), block.cols()) = block;
        next(block.rows(), block.cols());
    }
    else if constexpr (block.rank() == 2U)
    {
        assert(col + block.cols() <= matrix.cols());
        assert(row + block.rows() <= matrix.rows());

        matrix.block(row, col, block.rows(), block.cols()) = block.matrix();
        next(block.rows(), block.cols());
    }
    else
    {
        static_assert(block.rank() == 1U);
        assert(col + 1 <= matrix.cols());
        assert(row + block.size() <= matrix.rows());

        matrix.block(row, col, block.size(), 1) = block.vector();
        next(block.size(), 1);
    }
}

template <typename tscalar, typename tblock, typename... tblocks>
void stack(tensor_mem_t<tscalar, 1U>& vector, const tensor_size_t row, const tblock& block, const tblocks&... blocks)
{
    static_assert(is_eigen_v<tblock> || is_tensor_v<tblock>);

    if constexpr (is_eigen_v<tblock>)
    {
        assert(block.cols() == 1);
        assert(row + block.size() <= vector.size());

        vector.segment(row, block.size()) = block;
    }
    else
    {
        static_assert(block.rank() == 1U);
        assert(row + block.size() <= vector.size());

        vector.segment(row, block.size()) = block.vector();
    }

    if constexpr (sizeof...(blocks) > 0)
    {
        stack(vector, row + block.size(), blocks...);
    }
    else
    {
        assert(row + block.size() == vector.size());
    }
}
} // namespace detail

///
/// \brief stack the given Eigen blocks (e.g. matrices, vectors, expressions) in a matrix with the given size.
///
/// NB: the blocks are given in row-major fashion and are assumed to be compatible in size and without gaps.
///
/// example:
///     inputs: (matrix1, mastrix2, matrix3, matrix4, matrix5, transposed_vector)
///     result: +--------------------|---------------+
///             |                    |               |
///             |     matrix1        |    matrix2    |
///             |                    |               |
///             +------------|-------|-----|---------+
///             |            |             |         |
///             |  matrix3   |   matrix4   | matrix5 |
///             |            |             |         |
///             +------------------------------------+
///             |          transposed_vector         |
///             +------------------------------------+
///
template <typename tscalar, typename... tblocks>
auto stack(const tensor_size_t rows, const tensor_size_t cols, const tblocks&... blocks) ->
    typename std::enable_if<((is_eigen_v<tblocks> || is_tensor_v<tblocks>)&&...), tensor_mem_t<tscalar, 2U>>::type
{
    auto matrix = tensor_mem_t<tscalar, 2U>{rows, cols};

    detail::stack<tscalar>(matrix, 0, 0, blocks...);

    return matrix;
}

///
/// \brief stack the given Eigen segments (e.g. vectors, expressions) in a vector with the given size.
///
/// NB: the segments are given in row-major fashion and are assumed to be compatible in size and without gaps.
///
template <typename tscalar, typename... tblocks>
auto stack(const tensor_size_t rows, const tblocks&... blocks) ->
    typename std::enable_if<((is_eigen_v<tblocks> || is_tensor_v<tblocks>)&&...), tensor_mem_t<tscalar, 1U>>::type
{
    auto vector = tensor_mem_t<tscalar, 1U>{rows};

    detail::stack<tscalar>(vector, 0, blocks...);

    return vector;
}
} // namespace nano
