#pragma once

#include <charconv>
#include <cstdint>
#include <nano/enum.h>
#include <nano/string.h>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

namespace nano
{
///
/// \brief concatenate a list of potentially heterogeneous values into a formatted string.
///
namespace detail
{
template <class tvalue>
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

template <class tvalue>
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

template <class... tvalues>
void scat(std::ostringstream& stream, const tvalues&... values)
{
    (scat(stream, values), ...);
}

template <class tvalue>
tvalue from_chars(const std::string_view& str)
{
    auto result                           = tvalue{};
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
} // namespace detail

template <class... tvalues>
string_t scat(const tvalues&... values)
{
    std::ostringstream stream;
    detail::scat(stream, values...);
    return stream.str();
}
} // namespace nano
