#pragma once

#include <Eigen/Core>
#include <nano/core/random.h>
#include <type_traits>

namespace nano
{
///
/// \brief vector types.
///
template <typename tscalar_, int trows = Eigen::Dynamic, typename tscalar = std::remove_const_t<tscalar_>>
using eigen_vector_t = Eigen::Matrix<tscalar, trows, 1, Eigen::ColMajor>;

///
/// \brief map non-constant arrays to vectors.
///
template <int alignment    = Eigen::Unaligned, typename tscalar_, typename tsize,
          typename tscalar = std::remove_const_t<tscalar_>,
          typename tresult = Eigen::Map<eigen_vector_t<tscalar>, alignment>>
tresult map_vector(tscalar_* data, const tsize rows) noexcept
{
    return {data, static_cast<Eigen::Index>(rows)};
}

///
/// \brief map constant arrays to vectors.
///
template <int alignment    = Eigen::Unaligned, typename tscalar_, typename tsize,
          typename tscalar = std::remove_const_t<tscalar_>,
          typename tresult = Eigen::Map<const eigen_vector_t<tscalar>, alignment>>
tresult map_vector(const tscalar_* data, const tsize rows) noexcept
{
    return {data, static_cast<Eigen::Index>(rows)};
}

///
/// \brief matrix types.
///
template <typename tscalar_, int trows = Eigen::Dynamic, int tcols = Eigen::Dynamic,
          typename tscalar = std::remove_const_t<tscalar_>>
using eigen_matrix_t = Eigen::Matrix<tscalar, trows, tcols, Eigen::RowMajor>;

///
/// \brief map non-constant data to matrices.
///
template <int alignment    = Eigen::Unaligned, typename tscalar_, typename tsize1, typename tsize2,
          typename tscalar = std::remove_const_t<tscalar_>,
          typename tresult = Eigen::Map<eigen_matrix_t<tscalar>, alignment>>
tresult map_matrix(tscalar_* data, const tsize1 rows, const tsize2 cols) noexcept
{
    return {data, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols)};
}

///
/// \brief map constant data to Eigen matrices.
///
template <int alignment    = Eigen::Unaligned, typename tscalar_, typename tsize1, typename tsize2,
          typename tscalar = std::remove_const_t<tscalar_>,
          typename tresult = Eigen::Map<const eigen_matrix_t<tscalar>, alignment>>
tresult map_matrix(const tscalar_* data, const tsize1 rows, const tsize2 cols) noexcept
{
    return {data, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols)};
}

///
/// \brief traits to check if a given type is an Eigen vector, matrix or expression.
///
template <class T>
struct is_eigen : std::false_type
{
};

template <class T>
struct is_eigen<Eigen::Map<T>> : std::true_type
{
};

template <class T>
struct is_eigen<Eigen::EigenBase<T>> : std::true_type
{
};

template <typename tscalar, int trows, int tcols, int toptions>
struct is_eigen<Eigen::Matrix<tscalar, trows, tcols, toptions>> : std::true_type
{
};

template <typename UnaryOp, typename XprType>
struct is_eigen<Eigen::CwiseNullaryOp<UnaryOp, XprType>> : std::true_type
{
};

template <typename UnaryOp, typename XprType>
struct is_eigen<Eigen::CwiseUnaryOp<UnaryOp, XprType>> : std::true_type
{
};

template <typename BinaryOp, typename LhsType, typename RhsType>
struct is_eigen<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> : std::true_type
{
};

template <typename MatrixType, typename BinaryOp, int Direction>
struct is_eigen<Eigen::PartialReduxExpr<MatrixType, BinaryOp, Direction>> : std::true_type
{
};

template <typename LhsType, typename RhsType, int toptions>
struct is_eigen<Eigen::Product<LhsType, RhsType, toptions>> : std::true_type
{
};

template <class T>
struct is_eigen<Eigen::ArrayWrapper<T>> : std::true_type
{
};

template <class T>
struct is_eigen<Eigen::Transpose<T>> : std::true_type
{
};

template <class T>
struct is_eigen<Eigen::VectorBlock<T>> : std::true_type
{
};

template <class T>
inline constexpr bool is_eigen_v = is_eigen<T>::value;

///
/// \brief returns true if the two Eigen vectors or matrices are close.
///
template <typename teigen1, typename teigen2, typename tscalar, std::enable_if_t<is_eigen_v<teigen1>, bool> = true,
          std::enable_if_t<is_eigen_v<teigen2>, bool>           = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
bool close(const teigen1& lhs, const teigen2& rhs, tscalar epsilon) noexcept
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    else if (lhs.size() == 0)
    {
        return true;
    }
    else
    {
        return (lhs - rhs).array().abs().maxCoeff() <
               epsilon * (1 + lhs.array().abs().maxCoeff() + rhs.array().abs().maxCoeff());
    }
}
} // namespace nano

namespace Eigen
{
///
/// \brief iterators for Eigen matrices for STL compatibility.
///
template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
auto begin(Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m) noexcept
{
    return m.data();
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
auto begin(const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m) noexcept
{
    return m.data();
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
auto end(Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m) noexcept
{
    return m.data() + m.size();
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
auto end(const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m) noexcept
{
    return m.data() + m.size();
}
} // namespace Eigen
