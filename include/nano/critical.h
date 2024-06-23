#pragma once

#include <nano/core/scat.h>
#include <stdexcept>

namespace nano
{
///
/// \brief throws an exception as a critical condition is satisfied.
/// FIXME: use std::source_location when moving to C++20 to automatically add this information
///
template <class... tmessage>
[[noreturn]] void critical0(const tmessage&... message)
{
    throw std::runtime_error(scat("critical check failed: ", message..., "!"));
}

///
/// \brief checks and throws an exception if the given condition is satisfied.
/// FIXME: use std::source_location when moving to C++20 to automatically add this information
///
template <class tcondition, class... tmessage>
void critical(const tcondition& condition, const tmessage&... message)
{
    if (static_cast<bool>(condition))
    {
        critical0(message...);
    }
}
} // namespace nano
