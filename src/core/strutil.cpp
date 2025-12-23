#include <algorithm>
#include <nano/core/strutil.h>

using namespace nano;

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
