#pragma once

#include <nano/function.h>

namespace nano
{
template <class tmatrix>
using is_matrix_v = is_eigen_v<tmatrix> || (is_tensor_v<tmatrix> && tmatrix::rank() == 2U);

template <class tvector>
using is_vector_v = is_eigen_v<tvector> || (is_tensor_v<tvector> && tvector::rank() == 1U);

template <class tmatrix, std::enable_if_t<is_matrix_v<tmatrix>, bool> = true>
struct lhs_multiplied_function_t
{
    // attributes
    const tmatrix& m_matrix;   ///<
    function_t&    m_function; ///<
};

template <class tmatrix>
auto operator*(const tmatrix& matrix, function_t& function)
{
    return lhs_multiplied_function_t<tmatrix>{matrix, function};
}

///
/// \brief register a linear equality constraint: A * x = b.
///
template <class tmatrix, class tvector, std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator==(const lhs_multiplied_function_t<tmatrix>& lhs_multiplied_function, const tvectorb& b)
{
    const auto& [A, f] = lhs_multiplied_function;

    bool ok = (A.rows() == b.size()) && (A.cols() == function.size());
    for (tensor_size_t i = 0; i < A.rows() && ok; ++i)
    {
        ok = f.constrain(constraint::linear_equality_t{A.row(i), -b(i)});
    }
    return ok;
}

///
/// \brief register a linear equality constraint: A * x <= b.
///
template <class tmatrix, class tvector, std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator<=(const lhs_multiplied_function_t<tmatrix>& lhs_multiplied_function, const tvectorb& b)
{
    const auto& [A, f] = lhs_multiplied_function;

    bool ok = (A.rows() == b.size()) && (A.cols() == function.size());
    for (tensor_size_t i = 0; i < A.rows() && ok; ++i)
    {
        ok = f.constrain(constraint::linear_inequality_t{A.row(i), -b(i)});
    }
    return ok;
}

///
/// \brief register a linear equality constraint: A * x >= b.
///
template <class tmatrix, class tvector, std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator>=(const lhs_multiplied_function_t<tmatrix>& lhs_multiplied_function, const tvectorb& b)
{
    const auto& [A, f] = lhs_multiplied_function;

    bool ok = (A.rows() == b.size()) && (A.cols() == function.size());
    for (tensor_size_t i = 0; i < A.rows() && ok; ++i)
    {
        ok = f.constrain(constraint::linear_inequality_t{-A.row(i), b(i)});
    }
    return ok;
}

///
/// \brief register a linear equality constraint: a.dot(x) = b.
///
template <class tvector>
bool operator==(const lhs_multiplied_function_t<tvector>& lhs_multiplied_function, const scalar_t b)
{
    const auto& [a, f] = lhs_multiplied_function;

    if constexpr (is_eigen_v<tvector>)
    {
        assert(a.cols() == 1);
    }
    else
    {
        static_assert(tvectora::rank() == 1U);
    }

    return (a.rows() == f.size()) && f.constrain(constraint::linear_equality_t{a.transpose(), b});
}

///
/// \brief register a linear equality constraint: a.dot(x) <= b.
///
template <class tvector>
bool operator<=(const lhs_multiplied_function_t<tvector>& lhs_multiplied_function, const scalar_t b)
{
    const auto& [a, f] = lhs_multiplied_function;

    if constexpr (is_eigen_v<tvector>)
    {
        assert(a.cols() == 1);
    }
    else
    {
        static_assert(tvectora::rank() == 1U);
    }

    return (a.rows() == f.size()) && f.constrain(constraint::linear_inequality_t{a.transpose(), -b});
}

///
/// \brief register a linear equality constraint: a.dot(x) >= b.
///
template <class tvector>
bool operator>=(const lhs_multiplied_function_t<tvector>& lhs_multiplied_function, const scalar_t b)
{
    const auto& [a, f] = lhs_multiplied_function;

    if constexpr (is_eigen_v<tvector>)
    {
        assert(a.cols() == 1);
    }
    else
    {
        static_assert(tvectora::rank() == 1U);
    }

    return (a.rows() == f.size()) && f.constrain(constraint::linear_inequality_t{-a.transpose(), b});
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] <= upper[i].
///
template <class tvector, std::enable_if_t<is_function_comparable_v<tvector>, bool> = true>
bool operator<=(function_t& function, const tvector& upper)
{
    bool ok = upper.size() == function.size();
    for (tensor_size_t i = 0; i < upper.size() && ok; ++i)
    {
        ok = function.constrain(constraint::maximum_t{upper(i), i});
    }
    return ok;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] <= upper.
///
inline bool operator<=(function_t& function, const scalar_t upper)
{
    return function <= vector_t::constant(function.size(), upper);
}

///
/// \brief register a one-sided inequality contraint for the given dimension: x[dimension] <= upper.
///
inline bool operator<=(function_t& function, const std::pair<scalar_t, tensor_size_t>& upper_and_dimension)
{
    const auto [upper, dimension] = upper_and_dimension;
    return dimension >= 0 && dimension < function.size() && function.constrain(maximum_t{upper, dimension});
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] >= lower[i].
///
template <class tvector, std::enable_if_t<is_function_comparable_v<tvector>, bool> = true>
bool operator>=(function_t& function, const tvector& lower)
{
    bool ok = lower.size() == function.size();
    for (tensor_size_t i = 0; i < lower.size() && ok; ++i)
    {
        ok = function.constrain(constraint::minimum_t{lower(i), i});
    }
    return ok;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] >= lower.
///
inline bool operator>=(function_t& function, const scalar_t lower)
{
    return function >= vector_t::constant(function.size(), lower);
}

///
/// \brief register a one-sided inequality contraint for the given dimension: x[dimension] >= lower.
///
inline bool operator>=(function_t& function, const std::pair<scalar_t, tensor_size_t>& lower_and_dimension)
{
    const auto [lower, dimension] = lower_and_dimension;
    return dimension >= 0 && dimension < function.size() && function.constrain(minimum_t{lower, dimension});
}
} // namespace nano
