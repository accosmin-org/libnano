#pragma once

#include <cstdint>
#include <ostream>
#include <regex>
#include <type_traits>
#include <utility>
#include <vector>

namespace nano
{
template <class tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
using enum_map_t = std::vector<std::pair<tenum, const char*>>;

///
/// \brief maps all possible values of an enum to string.
///
template <class tenum>
enum_map_t<tenum> enum_string();

///
/// \brief stream enum using its associated string representation.
///
template <class tenum>
requires std::is_enum_v<tenum> std::ostream& operator<<(std::ostream& stream, const tenum value)
{
    static const auto enum_strings = enum_string<tenum>();
    for (const auto& elem : enum_strings)
    {
        if (elem.first == value)
        {
            stream << elem.second;
        }
    }
    return stream;
}

///
/// \brief collect all the values of an enum type, optionally filtered by the given regular expression.
///
template <class tenum>
requires std::is_enum_v<tenum> auto enum_values(const std::regex& enum_regex = std::regex(".+"))
{
    std::vector<tenum> enums;
    for (const auto& elem : enum_string<tenum>())
    {
        if (std::regex_match(elem.second, enum_regex))
        {
            enums.push_back(elem.first);
        }
    }
    return enums;
} // LCOV_EXCL_LINE
} // namespace nano
