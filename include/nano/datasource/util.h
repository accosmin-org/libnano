#pragma once

#include <cstdlib>
#include <nano/string.h>

namespace nano
{
///
/// \brief returns the value associated to the given environment variable.
///
string_t getenv(const char* env_var_name);
} // namespace nano
