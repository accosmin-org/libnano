#pragma once

#include <nano/core/strutil.h>

namespace nano
{
    enum class enum_type
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

    inline std::ostream& operator<<(std::ostream& os, enum_type type) { return os << scat(type); }
} // namespace nano
