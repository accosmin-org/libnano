#pragma once

#include <nano/core/strutil.h>

namespace nano
{
enum class enum_type : uint8_t
{
    type1,
    type2,
    type3
};

template <>
inline enum_map_t<enum_type> enum_string<enum_type>()
{
    return {
        {enum_type::type1, "type1"},
        {enum_type::type2, "type2"},
        {enum_type::type3, "type3"}
    };
}
} // namespace nano
