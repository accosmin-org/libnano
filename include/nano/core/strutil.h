#pragma once

#include <cstdint>
#include <nano/arch.h>
#include <nano/core/scat.h>

namespace nano
{
///
/// \brief text alignment options
///
enum class alignment : uint8_t
{
    left,
    center,
    right
};

///
/// \brief cast string to scalar value.
/// NB: the enumerations are handled separetely.
///
template <typename tvalue>
tvalue from_string(const std::string_view& str)
{
    if constexpr (std::is_arithmetic_v<tvalue>)
    {
#ifdef NANO_HAS_FROM_CHARS_FLOAT
        return detail::from_chars<tvalue>(str);
#else
        if constexpr (std::is_integral_v<tvalue>)
        {
            return detail::from_chars<tvalue>(str);
        }
        else if constexpr (std::is_floating_point_v<tvalue>)
        {
            return static_cast<tvalue>(std::stold(string_t{str}));
        }
#endif
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
NANO_PUBLIC bool starts_with(const std::string_view& str, const std::string_view& token);

///
/// \brief check if a string ends with a token (case sensitive).
///
NANO_PUBLIC bool ends_with(const std::string_view& str, const std::string_view& token);

///
/// \brief align a string to fill the given size (if possible).
///
NANO_PUBLIC string_t align(const std::string_view& str, size_t str_size, alignment mode = alignment::left,
                           char fill_char = ' ');
} // namespace nano
