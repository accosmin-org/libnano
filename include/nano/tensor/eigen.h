#pragma once

#include <Eigen/Core>
#include <nano/core/random.h>
#include <type_traits>

namespace nano
{
///
/// \brief vector types.
///
template <class tscalar_, int trows = Eigen::Dynamic, class tscalar = std::remove_const_t<tscalar_>>
using eigen_vector_t = Eigen::Matrix<tscalar, trows, 1, Eigen::ColMajor>;

///
/// \brief map non-constant arrays to vectors.
///
template <int alignment = Eigen::Unaligned, class tscalar_, class tsize, class tscalar = std::remove_const_t<tscalar_>,
          class tresult = Eigen::Map<eigen_vector_t<tscalar>, alignment>>
tresult map_vector(tscalar_* data, const tsize rows) noexcept
{
    return {data, static_cast<Eigen::Index>(rows)};
}

///
/// \brief map constant arrays to vectors.
///
template <int alignment = Eigen::Unaligned, class tscalar_, class tsize, class tscalar = std::remove_const_t<tscalar_>,
          class tresult = Eigen::Map<const eigen_vector_t<tscalar>, alignment>>
tresult map_vector(const tscalar_* data, const tsize rows) noexcept
{
    return {data, static_cast<Eigen::Index>(rows)};
}

///
/// \brief matrix types.
///
template <class tscalar_, int trows = Eigen::Dynamic, int tcols = Eigen::Dynamic,
          class tscalar = std::remove_const_t<tscalar_>>
using eigen_matrix_t = Eigen::Matrix<tscalar, trows, tcols, Eigen::RowMajor>;

///
/// \brief map non-constant data to matrices.
///
template <int alignment = Eigen::Unaligned, class tscalar_, class tsize1, class tsize2,
          class tscalar = std::remove_const_t<tscalar_>, class tresult = Eigen::Map<eigen_matrix_t<tscalar>, alignment>>
tresult map_matrix(tscalar_* data, const tsize1 rows, const tsize2 cols) noexcept
{
    return {data, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols)};
}

///
/// \brief map constant data to Eigen matrices.
///
template <int alignment = Eigen::Unaligned, class tscalar_, class tsize1, class tsize2,
          class tscalar = std::remove_const_t<tscalar_>,
          class tresult = Eigen::Map<const eigen_matrix_t<tscalar>, alignment>>
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

template <class tscalar, int trows, int tcols, int toptions>
struct is_eigen<Eigen::Matrix<tscalar, trows, tcols, toptions>> : std::true_type
{
};

template <class UnaryOp, class XprType>
struct is_eigen<Eigen::CwiseNullaryOp<UnaryOp, XprType>> : std::true_type
{
};

template <class UnaryOp, class XprType>
struct is_eigen<Eigen::CwiseUnaryOp<UnaryOp, XprType>> : std::true_type
{
};

template <class BinaryOp, class LhsType, class RhsType>
struct is_eigen<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> : std::true_type
{
};

template <class MatrixType, class BinaryOp, int Direction>
struct is_eigen<Eigen::PartialReduxExpr<MatrixType, BinaryOp, Direction>> : std::true_type
{
};

template <class LhsType, class RhsType, int toptions>
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
struct is_eigen<Eigen::Inverse<T>> : std::true_type
{
};

template <class T>
struct is_eigen<Eigen::VectorBlock<T>> : std::true_type
{
};

template <class XprType, int BlockRows, int BlockCols, bool InnerPanel>
struct is_eigen<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>> : std::true_type
{
};

template <class T>
inline constexpr bool is_eigen_v = is_eigen<T>::value;

///
/// \brief returns true if the two Eigen vectors or matrices are close.
///
template <class teigen1, class teigen2, class tscalar, std::enable_if_t<is_eigen_v<teigen1>, bool> = true,
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

namespace std // NOLINT(cert-dcl58-cpp)
{
///
/// \brief iterators for Eigen matrices for STL compatibility.
///
template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
auto begin(Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m) noexcept // NOLINT(cert-dcl58-cpp)
{
    return m.data();
}

template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
auto begin( // NOLINT(cert-dcl58-cpp)
    const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m) noexcept
{
    return m.data();
}

template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
auto end(Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m) noexcept // NOLINT(cert-dcl58-cpp)
{
    return m.data() + m.size();
}

template <class Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
auto end( // NOLINT(cert-dcl58-cpp)
    const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m) noexcept
{
    return m.data() + m.size();
}
} // namespace std
