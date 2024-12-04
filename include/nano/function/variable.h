#pragma once

#include <nano/tensor/index.h>

namespace nano
{
class function_t;

///
/// \brief models the variable of a function, useful for constructing constraints.
///
struct function_variable_t
{
    // attributes
    function_t&   m_function;      ///<
};

///
/// \brief models the variable of a function, useful for constructing per-dimension constraints.
///
struct function_variable_dimension_t
{
    // attributes
    tensor_size_t m_dimension{-1}; ///<
    function_t&   m_function;      ///<
};
}
