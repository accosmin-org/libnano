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
using tensor_vector_t = Eigen::Matrix<tscalar, trows, 1, Eigen::ColMajor>;

///
/// \brief map non-constant arrays to vectors.
///
template <int alignment    = Eigen::Unaligned, typename tscalar_, typename tsize,
          typename tscalar = std::remove_const_t<tscalar_>,
          typename tresult = Eigen::Map<tensor_vector_t<tscalar>, alignment>>
tresult map_vector(tscalar_* data, const tsize rows) noexcept
{
    return {data, static_cast<Eigen::Index>(rows)};
}

///
/// \brief map constant arrays to vectors.
///
template <int alignment    = Eigen::Unaligned, typename tscalar_, typename tsize,
          typename tscalar = std::remove_const_t<tscalar_>,
          typename tresult = Eigen::Map<const tensor_vector_t<tscalar>, alignment>>
tresult map_vector(const tscalar_* data, const tsize rows) noexcept
{
    return {data, static_cast<Eigen::Index>(rows)};
}

///
/// \brief matrix types.
///
template <typename tscalar_, int trows = Eigen::Dynamic, int tcols = Eigen::Dynamic,
          typename tscalar = std::remove_const_t<tscalar_>>
using tensor_matrix_t = Eigen::Matrix<tscalar, trows, tcols, Eigen::RowMajor>;

///
/// \brief map non-constant data to matrices.
///
template <int alignment    = Eigen::Unaligned, typename tscalar_, typename tsize1, typename tsize2,
          typename tscalar = std::remove_const_t<tscalar_>,
          typename tresult = Eigen::Map<tensor_matrix_t<tscalar>, alignment>>
tresult map_matrix(tscalar_* data, const tsize1 rows, const tsize2 cols) noexcept
{
    return {data, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols)};
}

///
/// \brief map constant data to Eigen matrices.
///
template <int alignment    = Eigen::Unaligned, typename tscalar_, typename tsize1, typename tsize2,
          typename tscalar = std::remove_const_t<tscalar_>,
          typename tresult = Eigen::Map<const tensor_matrix_t<tscalar>, alignment>>
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

template <typename BinaryOp, typename LhsType, typename RhsType>
struct is_eigen<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> : std::true_type
{
};

template <typename UnaryOp, typename XprType>
struct is_eigen<Eigen::CwiseUnaryOp<UnaryOp, XprType>> : std::true_type
{
};

template <class T>
struct is_eigen<Eigen::ArrayWrapper<T>> : std::true_type
{
};

template <class T>
inline constexpr bool is_eigen_v = is_eigen<T>::value;

///
/// \brief create a matrix from an initializer list.
///
template <typename tscalar, typename... tvalues>
auto make_matrix(const Eigen::Index rows, tvalues... values)
{
    const auto list = {static_cast<tscalar>(values)...};
    const auto size = static_cast<Eigen::Index>(list.size());
    assert(size % rows == 0);

    tensor_matrix_t<tscalar> matrix{rows, size / rows};
    std::copy(list.begin(), list.end(), begin(matrix));
    return matrix;
}

///
/// \brief create a vector from an initializer list.
///
template <typename tscalar, typename... tvalues>
auto make_vector(tvalues... values)
{
    const auto list = {static_cast<tscalar>(values)...};

    tensor_vector_t<tscalar> vector{static_cast<Eigen::Index>(list.size())};
    std::copy(list.begin(), list.end(), begin(vector));
    return vector;
}

///
/// \brief create a vector and fill it with the given value.
///
template <typename tscalar, typename tscalar_value>
auto make_full_vector(const Eigen::Index rows, const tscalar_value value)
{
    return tensor_vector_t<tscalar>{tensor_vector_t<tscalar>::Constant(rows, static_cast<tscalar>(value))};
}

///
/// \brief create a vector and fill it with random values uniformly distributed in the given range.
///
template <typename tscalar, typename tscalar_value = tscalar>
auto make_random_vector(const Eigen::Index rows, const tscalar_value min_value = -1, const tscalar_value max_value = +1,
                        const seed_t seed = seed_t{})
{
    tensor_vector_t<tscalar> vector(rows);
    urand(static_cast<tscalar>(min_value), static_cast<tscalar>(max_value), begin(vector), end(vector), make_rng(seed));
    return vector;
} // LCOV_EXCL_LINE

///
/// \brief create a matrix and fill it with random values uniformly distributed in the given range.
///
template <typename tscalar, typename tscalar_value = tscalar>
auto make_random_matrix(const Eigen::Index rows, const Eigen::Index cols, const tscalar_value min_value = -1,
                        const tscalar_value max_value = +1, const seed_t seed = seed_t{})
{
    tensor_matrix_t<tscalar> matrix(rows, cols);
    urand(static_cast<tscalar>(min_value), static_cast<tscalar>(max_value), begin(matrix), end(matrix), make_rng(seed));
    return matrix;
} // LCOV_EXCL_LINE

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
