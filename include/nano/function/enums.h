#pragma once

#include <nano/enum.h>

namespace nano
{
enum class convexity : uint8_t
{
    ignore,
    yes,
    no
};

enum class smoothness : uint8_t
{
    yes,
    no
};

///
/// \brief classification of (benchmark) functions.
///
enum class function_type : uint8_t
{
    any,               ///< any function with or without constraints
    convex,            ///< convex function (smooth or non-smooth) without constraints
    smooth,            ///< smooth function (convex or non-convex) without constraints
    convex_smooth,     ///< convex smooth function without constraints
    convex_nonsmooth,  ///< convex non-smooth function without constraints
    linear_program,    ///< linear program (linear objective with linear constraints)
    quadratic_program, ///< quadratic problem (quadratic convex objective with linear constraints)
};

template <>
inline enum_map_t<function_type> enum_string()
{
    return {
        {              function_type::any,               "any"},
        {           function_type::convex,            "convex"},
        {           function_type::smooth,            "smooth"},
        {    function_type::convex_smooth,     "convex-smooth"},
        { function_type::convex_nonsmooth,  "convex-nonsmooth"},
        {   function_type::linear_program,    "linear-program"},
        {function_type::quadratic_program, "quadratic-program"},
    };
}

} // namespace nano
