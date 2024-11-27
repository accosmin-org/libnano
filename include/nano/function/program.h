#pragma once

#include <nano/function/linear.h>
#include <nano/function/quadratic.h>

namespace nano
{
///
/// \brief models a linear equality constraint: A * x = b.
///
template <class tmatrixA, class tvectorb>
struct equality_t
{
    // attributes
    tmatrixA m_A; ///<
    tvectorb m_b; ///<
};

///
/// \brief models a linear equality constraint: A * x <= b.
///
template <class tmatrixA, class tvectorb>
struct inequality_t
{
    // attributes
    tmatrixA m_A; ///<
    tvectorb m_b; ///<
};

///
/// \brief create a generic equality constraint: A * x = b.
///
template <class tmatrixA, class tvectorb,
          std::enable_if_t<is_eigen_v<tmatrixA> || is_tensor_v<tmatrixA>, bool> = true, ///<
          std::enable_if_t<is_eigen_v<tvectorb> || is_tensor_v<tvectorb>, bool> = true> ///<
inline auto make_equality(tmatrixA A, tvectorb b)
{
    if constexpr (is_tensor_v<tmatrixA>)
    {
        static_assert(tmatrixA::rank() == 2U);
    }
    if constexpr (is_tensor_v<tvectorb>)
    {
        static_assert(tvectorb::rank() == 1U);
    }
    return equality_t<std::remove_cv_t<tmatrixA>, std::remove_cv_t<tvectorb>>{std::move(A), std::move(b)};
}

///
/// \brief create a scalar equality constraint: a.dot(x) = b.
///
template <class tvectora, std::enable_if_t<is_eigen_v<tvectora> || is_tensor_v<tvectora>, bool> = true>
inline auto make_equality(const tvectora& a, const scalar_t b)
{
    if constexpr (is_eigen_v<tvectora>)
    {
        assert(a.cols() == 1);
    }
    else
    {
        static_assert(tvectora::rank() == 1U);
    }
    return program::make_equality(a.transpose(), vector_t::constant(1, b));
}

///
/// \brief create a generic inequality constraint: A * x <= b.
///
template <class tmatrixA, class tvectorb,
          std::enable_if_t<is_eigen_v<tmatrixA> || is_tensor_v<tmatrixA>, bool> = true, ///<
          std::enable_if_t<is_eigen_v<tvectorb> || is_tensor_v<tvectorb>, bool> = true> ///<
inline auto make_inequality(tmatrixA A, tvectorb b)
{
    if constexpr (is_tensor_v<tmatrixA>)
    {
        static_assert(tmatrixA::rank() == 2U);
    }
    if constexpr (is_tensor_v<tvectorb>)
    {
        static_assert(tvectorb::rank() == 1U);
    }
    return inequality_t<std::remove_cv_t<tmatrixA>, std::remove_cv_t<tvectorb>>{std::move(A), std::move(b)};
}

///
/// \brief create a scalar inequality constraint: a.dot(x) <= b.
///
template <class tvectora, std::enable_if_t<is_eigen_v<tvectora> || is_tensor_v<tvectora>, bool> = true>
inline auto make_inequality(const tvectora& a, const scalar_t b)
{
    if constexpr (is_eigen_v<tvectora>)
    {
        assert(a.cols() == 1);
    }
    else
    {
        static_assert(tvectora::rank() == 1U);
    }
    return make_inequality(a.transpose(), vector_t::constant(1, b));
}

///
/// \brief create a one-sided inequality contraint: x <= upper (element-wise).
///
inline auto make_less(const tensor_size_t dims, const scalar_t upper)
{
    return make_inequality(matrix_t::identity(dims, dims), vector_t::constant(dims, upper));
}

inline auto make_less(const vector_t& upper)
{
    const auto dims = upper.size();
    return make_inequality(matrix_t::identity(dims, dims), upper);
}

///
/// \brief create a one-sided inequality contraint: lower <= x (element-wise).
///
inline auto make_greater(const tensor_size_t dims, const scalar_t lower)
{
    return make_inequality(-matrix_t::identity(dims, dims), -vector_t::constant(dims, lower));
}

inline auto make_greater(const vector_t& lower)
{
    const auto dims = lower.size();
    return make_inequality(-matrix_t::identity(dims, dims), -lower.vector());
}

///
/// \brief construct a linear program from the given objective and the equality and inequality constraints.
///
template <class... tconstraints>
auto make_linear_program(string_t id, vector_t c, const tconstraints&... constraints)
{
    auto program = linear_program_t{std::move(id), std::move(c)};
    for (tensor_size_t i = 0; i < A.rows(); ++i)
    {
        function.constrain(constraint::linear_equality_t{A.row(i), -b(i)});
    }
    for (tensor_size_t i = 0; i < G.rows(); ++i)
    {
        function.constrain(constraint::linear_inequality_t{G.row(i), -h(i)});
    }
    return program;
}

///
/// \brief construct a quadratic program from the given objective and the equality and inequality constraints.
///
template <class... tconstraints>
auto make_quadratic_program(string_t id, matrix_t Q, vector_t c, const tconstraints&... constraints)
{
    auto program = quadratic_program_t{std::move(id), std::move(Q), std::move(c)};
    for (tensor_size_t i = 0; i < A.rows(); ++i)
    {
        function.constrain(constraint::linear_equality_t{A.row(i), -b(i)});
    }
    for (tensor_size_t i = 0; i < G.rows(); ++i)
    {
        function.constrain(constraint::linear_inequality_t{G.row(i), -h(i)});
    }
    return program;
}

///
/// \brief construct a quadratic program from the given objective and the equality and inequality constraints.
///
template <class... tconstraints>
auto make_quadratic_program(string_t id, const vector_t& Q_upper_triangular, vector_t c,
                            const tconstraints&... constraints)
{
    auto program = quadratic_program_t{std::move(id), Q_upper_triangular, std::move(c)};
    for (tensor_size_t i = 0; i < A.rows(); ++i)
    {
        function.constrain(constraint::linear_equality_t{A.row(i), -b(i)});
    }
    for (tensor_size_t i = 0; i < G.rows(); ++i)
    {
        function.constrain(constraint::linear_inequality_t{G.row(i), -h(i)});
    }
    return program;
}
} // namespace nano
