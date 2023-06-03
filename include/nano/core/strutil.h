#pragma once

#include <algorithm>
#include <charconv>
#include <nano/string.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <typeinfo>
#include <utility>

namespace nano
{
///
/// \brief text alignment options
///
enum class alignment : int
{
    left,
    center,
    right
};

template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
using enum_map_t = std::vector<std::pair<tenum, const char*>>;

///
/// \brief maps all possible values of an enum to string.
///
template <typename tenum>
enum_map_t<tenum> enum_string(); // FIXME: make it constexpr!

///
/// \brief stream enum using its associated string representation.
///
template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
std::ostream& operator<<(std::ostream& stream, const tenum value)
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
template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
auto enum_values(const std::regex& enum_regex = std::regex(".+"))
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

///
/// \brief concatenate a list of potentially heterogeneous values into a formatted string.
///
namespace detail
{
template <typename tvalue>
void scat(std::ostringstream& stream, const tvalue& value)
{
    if constexpr (std::is_enum_v<std::remove_reference_t<tvalue>>)
    {
        // FIXME: make it constexpr!
        static const auto enum_strings = enum_string<tvalue>();
        for (const auto& elem : enum_strings)
        {
            if (elem.first == value)
            { // cppcheck-suppress useStlAlgorithm
                stream << elem.second;
                return;
            }
        }

        const auto str = std::to_string(static_cast<int>(value));
        const auto msg = string_t("missing mapping for enumeration ") + typeid(tvalue).name() + " <" + str + ">!";
        throw std::invalid_argument(msg);
    }
    else
    {
        stream << value;
    }
}

template <typename tvalue>
void scat(std::ostringstream& stream, const std::vector<tvalue>& values)
{
    for (auto begin = values.begin(), end = values.end(); begin != end;)
    {
        scat(stream, *begin);
        if (++begin != end)
        {
            stream << ",";
        }
    }
}

template <typename... tvalues>
void scat(std::ostringstream& stream, const tvalues&... values)
{
    (scat(stream, values), ...);
}
} // namespace detail

template <typename... tvalues>
string_t scat(const tvalues&... values)
{
    std::ostringstream stream;
    detail::scat(stream, values...);
    return stream.str();
}

///
/// \brief cast string to value.
///
template <typename tvalue>
tvalue from_string(const std::string_view& str)
{
    if constexpr (std::is_arithmetic_v<tvalue>)
    {
        tvalue result{};
        [[maybe_unused]] const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
        if (ec == std::errc::invalid_argument)
        {
            throw std::invalid_argument("cannot interpret the string as an integer");
        }
        else if (ec == std::errc::result_out_of_range)
        {
            throw std::out_of_range("out of range integer");
        }
        return result;
    }
    else if constexpr (std::is_same_v<tvalue, string_t> || std::is_same_v<tvalue, std::string_view>)
    {
        return tvalue{str};
    }
    else if constexpr (std::is_enum_v<std::remove_reference_t<tvalue>>)
    {
        static const auto options = enum_string<tvalue>();
        for (const auto& option : options)
        {
            if (option.second == str)
            { // cppcheck-suppress useStlAlgorithm
                return option.first;
            }
        }
        for (const auto& option : options)
        {
            if (str.find(option.second) == 0)
            {
                return option.first;
            }
        }
        throw std::invalid_argument(scat("invalid ", typeid(tvalue).name(), " <", ">!"));
        // cppcheck-suppress missingReturn
    }
}

///
/// \brief cast string to value and use the given default value if casting fails.
///
template <typename tvalue>
tvalue from_string(const std::string_view& str, const tvalue& default_value)
{
    try
    {
        return from_string<tvalue>(str);
    }
    catch (std::exception&)
    {
        return default_value;
    }
}

///
/// \brief check if a string starts with a token (case sensitive).
///
inline bool starts_with(const std::string_view& str, const std::string_view& token)
{
    return str.size() >= token.size() && std::equal(token.begin(), token.end(), str.begin());
}

///
/// \brief check if a string ends with a token (case sensitive).
///
inline bool ends_with(const std::string_view& str, const std::string_view& token) noexcept
{
    return str.size() >= token.size() && std::equal(token.rbegin(), token.rend(), str.rbegin());
}

///
/// \brief align a string to fill the given size (if possible).
///
inline string_t align(const std::string_view& str, const size_t str_size, const alignment mode = alignment::left,
                      const char fill_char = ' ')
{
    const auto fill_size = (str.size() > str_size) ? size_t{0U} : (str_size - str.size());

    string_t ret;
    switch (mode)
    {
    case alignment::center:
        ret.append(fill_size / 2, fill_char);
        ret.append(str);
        ret.append(fill_size - fill_size / 2, fill_char);
        break;

    case alignment::right:
        ret.append(fill_size, fill_char);
        ret.append(str);
        break;

    default:
        ret.append(str);
        ret.append(fill_size, fill_char);
        break;
    }

    return ret;
} // LCOV_EXCL_LINE
} // namespace nano
