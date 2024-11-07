#pragma once

#include <nano/function.h>
#include <nano/program/linear.h>
#include <nano/program/quadratic.h>

namespace nano
{
///
/// \brief return the equivalent constrained function of the given linear program.
///
NANO_PUBLIC rfunction_t make_function(const program::linear_program_t& program);

///
/// \brief return the equivalent constrained function of the given quadratic program.
///
NANO_PUBLIC rfunction_t make_function(const program::quadratic_program_t& program);
} // namespace nano
