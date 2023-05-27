#pragma once

// FIXME: remove these macros when introspection is available!
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NANO_MAKE_ENUM_BEGIN(enum_type)                                                                                \
    template <>                                                                                                        \
    inline enum_map_t<enum_type> enum_string<enum_type>()                                                              \
    {                                                                                                                  \
        return                                                                                                         \
        {

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NANO_MAKE_ENUM_END()                                                                                           \
    }                                                                                                                  \
    ;                                                                                                                  \
    }

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NANO_MAKE_ENUM_VALUE(enum_type, value) {enum_type::value, #value},

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NANO_MAKE_ENUM2(enum_type, value1, value2)                                                                     \
    NANO_MAKE_ENUM_BEGIN(enum_type)                                                                                    \
    NANO_MAKE_ENUM_VALUE(enum_type, value1)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value2)                                                                            \
    NANO_MAKE_ENUM_END()

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NANO_MAKE_ENUM3(enum_type, value1, value2, value3)                                                             \
    NANO_MAKE_ENUM_BEGIN(enum_type)                                                                                    \
    NANO_MAKE_ENUM_VALUE(enum_type, value1)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value2)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value3)                                                                            \
    NANO_MAKE_ENUM_END()

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NANO_MAKE_ENUM4(enum_type, value1, value2, value3, value4)                                                     \
    NANO_MAKE_ENUM_BEGIN(enum_type)                                                                                    \
    NANO_MAKE_ENUM_VALUE(enum_type, value1)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value2)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value3)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value4)                                                                            \
    NANO_MAKE_ENUM_END()

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NANO_MAKE_ENUM5(enum_type, value1, value2, value3, value4, value5)                                             \
    NANO_MAKE_ENUM_BEGIN(enum_type)                                                                                    \
    NANO_MAKE_ENUM_VALUE(enum_type, value1)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value2)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value3)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value4)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value5)                                                                            \
    NANO_MAKE_ENUM_END()

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NANO_MAKE_ENUM12(enum_type, value1, value2, value3, value4, value5, value6, value7, value8, value9, value10,   \
                         value11, value12)                                                                             \
    NANO_MAKE_ENUM_BEGIN(enum_type)                                                                                    \
    NANO_MAKE_ENUM_VALUE(enum_type, value1)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value2)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value3)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value4)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value5)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value6)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value7)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value8)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value9)                                                                            \
    NANO_MAKE_ENUM_VALUE(enum_type, value10)                                                                           \
    NANO_MAKE_ENUM_VALUE(enum_type, value11)                                                                           \
    NANO_MAKE_ENUM_VALUE(enum_type, value12)                                                                           \
    NANO_MAKE_ENUM_END()
