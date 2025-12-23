#pragma once

#include <nano/enum.h>

namespace nano
{
///
/// \brief lasso-like regularized machine learning models can be casted:
///
///     * either as a constrained numerical optimization problem:
///         min_{x, z} ERM(x) + alpha1 * z.sum()
///         s.t.       -x <= z and
///                    +x <= z.
///
///     * or as an unconstrained numerical optimization problem:
///         min_{x}    ERM(x) + alpha1 * |x|.
///
enum class lasso_type : uint8_t
{
    constrained,   ///<
    unconstrained, ///<
};

template <>
inline enum_map_t<lasso_type> enum_string<lasso_type>()
{
    return {
        {  lasso_type::constrained,   "constrained"},
        {lasso_type::unconstrained, "unconstrained"},
    };
}
} // namespace nano
