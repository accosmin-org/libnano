#pragma once

#include <type_traits>

namespace nano
{
// https://en.cppreference.com/w/cpp/numeric/bit_cast
// FIXME: replace this with std::bit_cast when moving to c++20!
template <class To, class From>
auto bit_cast(const From& src)
    -> std::enable_if_t<
        sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>, To>
{
    static_assert(std::is_trivially_constructible_v<To>, "This implementation additionally requires "
                                                         "destination type to be trivially constructible");

    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}
} // namespace nano
