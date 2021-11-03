
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
        typename tscalar = std::remove_const_t<tscalar_>
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
        typename tscalar = std::remove_const_t<tscalar_>,
        typename tresult = Eigen::Map<tensor_vector_t<tscalar>, alignment>
    >
    tresult map_vector(tscalar_* data, tsize rows)
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
        typename tscalar = std::remove_const_t<tscalar_>,
        typename tresult = Eigen::Map<const tensor_vector_t<tscalar>, alignment>
    >
    tresult map_vector(const tscalar_* data, tsize rows)
    {
        return tresult(data, rows);
    }

    ///
    /// \brief matrix types.
    ///
    template
    <
        typename tscalar_,
        int trows = Eigen::Dynamic,
        int tcols = Eigen::Dynamic,
        typename tscalar = std::remove_const_t<tscalar_>
    >
    using tensor_matrix_t = Eigen::Matrix<tscalar, trows, tcols, Eigen::RowMajor>;

    ///
    /// \brief map non-constant data to matrices.
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = std::remove_const_t<tscalar_>,
        typename tresult = Eigen::Map<tensor_matrix_t<tscalar>, alignment>
    >
    tresult map_matrix(tscalar_* data, tsize rows, tsize cols)
    {
        return tresult(data, rows, cols);
    }

    ///
    /// \brief map constant data to Eigen matrices.
    ///
    template
    <
        int alignment = Eigen::Unaligned,
        typename tscalar_,
        typename tsize,
        typename tscalar = std::remove_const_t<tscalar_>,
        typename tresult = Eigen::Map<const tensor_matrix_t<tscalar>, alignment>
    >
    tresult map_matrix(const tscalar_* data, tsize rows, tsize cols)
    {
        return tresult(data, rows, cols);
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

    template <typename BinaryOp, typename LhsType, typename RhsType>
    struct is_eigen<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> : std::true_type
    {
    };

    template <typename UnaryOp, typename XprType>
    struct is_eigen<Eigen::CwiseUnaryOp<UnaryOp, XprType>> : std::true_type
    {
    };

    template <class T>
    inline constexpr bool is_eigen_v = is_eigen<T>::value;

    ///
    /// \brief returns true if the two Eigen vectors or matrices are close.
    ///
    template
    <
        typename teigen1,
        typename teigen2,
        typename tscalar,
        std::enable_if_t<is_eigen_v<teigen1>, bool> = true,
        std::enable_if_t<is_eigen_v<teigen2>, bool> = true,
        std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true
    >
    bool close(const teigen1& lhs, const teigen2& rhs, tscalar epsilon)
    {
        return  (lhs - rhs).array().abs().maxCoeff() <
                epsilon * (1 + lhs.array().abs().maxCoeff() + rhs.array().abs().maxCoeff());
    }
}

namespace Eigen
{
    ///
    /// \brief iterators for Eigen matrices for STL compatibility.
    ///
    template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
    auto begin(Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m)
    {
        return m.data();
    }

    template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
    auto begin(const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m)
    {
        return m.data();
    }

    template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
    auto end(Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m)
    {
        return m.data() + m.size();
    }

    template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options>
    auto end(const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options>& m)
    {
        return m.data() + m.size();
    }
}
