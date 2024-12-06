#pragma once

#include <nano/function.h>

namespace nano
{
template <class T>
inline constexpr bool is_matrix_v = is_eigen_v<T> || (is_tensor_v<T> && T::rank() <= 2U);

template <class T>
inline constexpr bool is_vector_v = is_eigen_v<T> || (is_tensor_v<T> && T::rank() == 1U);
/*
///
/// \brief proxy object to model the left-handside multiplication of a matrix or vector with the variable of a function,
///     useful for easily defining constraints.
///
template <class ttensor, std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true>
struct lhs_multiplied_variable_t
{
    template <class tlambda>
    bool call_scalar(const tlambda& lambda) const
    {
        const auto& a = m_tensor;
        const auto& f = m_variable.m_function;

        if constexpr (is_eigen_v<ttensor>)
        {
            return a.rows() == f.size() && a.cols() == 1 && f.constrain(lambda(a.transpose()));
        }
        else
        {
            static_assert(ttensor::rank() == 1U);

            return a.size() == f.size() && f.constrain(lambda(a.transpose()));
        }
    }

    template <class tvector, class tlambda, std::enable_if_t<is_vector_v<tvector>, bool> = true>
    bool call_vector(const tvector& b, const tlambda& lambda) const
    {
        const auto& A = m_tensor;
        const auto& f = m_variable.m_function;

        if constexpr (is_eigen_v<ttensor>)
        {
            bool ok = A.rows() == b.size() && A.cols() == f.size();
            for (tensor_size_t i = 0; i < A.rows() && ok; ++i)
            {
                ok = f.constrain(lambda(A.row(i), b(i)));
            }
            return ok;
        }
        else
        {
            static_assert(ttensor::rank() == 2U);

            bool ok = A.rows() == b.size() && A.cols() == f.size();
            for (tensor_size_t i = 0; i < A.rows() && ok; ++i)
            {
                ok = f.constrain(lambda(A.row(i), b(i)));
            }
            return ok;
        }
    }

    // attributes
    const ttensor&      m_tensor;   ///<
    function_variable_t m_variable; ///<
};

template <class ttensor, ///<
          std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true>
auto operator*(const ttensor& tensor, const function_variable_t& variable)
{
    return lhs_multiplied_variable_t<ttensor>{tensor, variable};
}

///
/// \brief register a linear equality constraint: A * x = b.
///
template <class ttensor, class tvector, ///<
          std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true,
          std::enable_if_t<is_vector_v<tvector>, bool>                         = true>
bool operator==(const lhs_multiplied_variable_t<ttensor>& lhs_multiplied_variable, const tvector& vb)
{
    const auto op = [&](const auto& a, const scalar_t b) { return constraint::linear_equality_t{a, -b}; };
    return lhs_multiplied_variable.call_vector(vb, op);
}

///
/// \brief register a linear equality constraint: A * x <= b.
///
template <class ttensor, class tvector, ///<
          std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true,
          std::enable_if_t<is_vector_v<tvector>, bool>                         = true>
bool operator<=(const lhs_multiplied_variable_t<ttensor>& lhs_multiplied_variable, const tvector& vb)
{
    const auto op = [&](const auto& a, const scalar_t b) { return constraint::linear_inequality_t{a, -b}; };
    return lhs_multiplied_variable.call_vector(vb, op);
}

///
/// \brief register a linear equality constraint: A * x >= b.
///
template <class ttensor, class tvector, ///<
          std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true,
          std::enable_if_t<is_vector_v<tvector>, bool>                         = true>
bool operator>=(const lhs_multiplied_variable_t<ttensor>& lhs_multiplied_variable, const tvector& vb)
{
    const auto op = [&](const auto& a, const scalar_t b) { return constraint::linear_inequality_t{-a, b}; };
    return lhs_multiplied_variable.call_vector(vb, op);
}

///
/// \brief register a linear equality constraint: a.dot(x) = b.
///
template <class tvector, ///<
          std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator==(const lhs_multiplied_variable_t<tvector>& lhs_multiplied_variable, const scalar_t b)
{
    const auto op = [&](const auto& a) { return constraint::linear_equality_t{a, -b}; };
    return lhs_multiplied_variable.call_scalar(op);
}

///
/// \brief register a linear equality constraint: a.dot(x) <= b.
///
template <class tvector, ///<
          std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator<=(const lhs_multiplied_variable_t<tvector>& lhs_multiplied_variable, const scalar_t b)
{
    const auto op = [&](const auto& a) { return constraint::linear_inequality_t{a, -b}; };
    return lhs_multiplied_variable.call_scalar(op);
}

///
/// \brief register a linear equality constraint: a.dot(x) >= b.
///
template <class tvector, ///<
          std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator>=(const lhs_multiplied_variable_t<tvector>& lhs_multiplied_variable, const scalar_t b)
{
    const auto op = [&](const auto& a) { return constraint::linear_inequality_t{-a, b}; };
    return lhs_multiplied_variable.call_scalar(op);
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] <= upper[i].
///
template <class tvector, ///<
          std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator<=(const function_variable_t& variable, const tvector& upper)
{
    bool ok = upper.size() == variable.m_function.size();
    for (tensor_size_t i = 0; i < upper.size() && ok; ++i)
    {
        ok = variable.m_function.constrain(constraint::maximum_t{upper(i), i});
    }
    return ok;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] <= upper.
///
inline bool operator<=(const function_variable_t& variable, const scalar_t upper)
{
    return variable <= vector_t::constant(variable.m_function.size(), upper);
}

///
/// \brief register a one-sided inequality contraint for the given dimension: x[dimension] <= upper.
///
inline bool operator<=(const function_variable_dimension_t& ivariable, const scalar_t upper)
{
    return ivariable.m_dimension >= 0 && ivariable.m_dimension < ivariable.m_function.size() &&
           ivariable.m_function.constrain(constraint::maximum_t{upper, ivariable.m_dimension});
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] >= lower[i].
///
template <class tvector, ///<
          std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator>=(const function_variable_t& variable, const tvector& lower)
{
    bool ok = lower.size() == variable.m_function.size();
    for (tensor_size_t i = 0; i < lower.size() && ok; ++i)
    {
        ok = variable.m_function.constrain(constraint::minimum_t{lower(i), i});
    }
    return ok;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] >= lower.
///
inline bool operator>=(const function_variable_t& variable, const scalar_t lower)
{
    return variable >= vector_t::constant(variable.m_function.size(), lower);
}

///
/// \brief register a one-sided inequality contraint for the given dimension: x[dimension] >= lower.
///
inline bool operator>=(const function_variable_dimension_t& ivariable, const scalar_t lower)
{
    return ivariable.m_dimension >= 0 && ivariable.m_dimension < ivariable.m_function.size() &&
           ivariable.m_function.constrain(constraint::minimum_t{lower, ivariable.m_dimension});
}
*/
} // namespace nano
