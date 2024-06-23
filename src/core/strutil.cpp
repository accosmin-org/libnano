#include <nano/core/strutil.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <nano/enum.h>
#include <nano/string.h>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

using namespace nano;

bool nano::starts_with(std::string_view str, std::string_view token)
{
    return str.size() >= token.size() && std::equal(token.begin(), token.end(), str.begin());
}

bool nano::ends_with(std::string_view str, std::string_view token)
{
    return str.size() >= token.size() && std::equal(token.rbegin(), token.rend(), str.rbegin());
}

string_t nano::align(std::string_view str, const size_t str_size, const alignment mode, const char fill_char)
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
}
