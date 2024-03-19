#pragma once

#include <cassert>

#ifdef NDEBUG
    #define NANO_ASSERT(condition, callback_when_condition_fails)
#else
    #define NANO_ASSERT(condition, callback_when_condition_fails)                                                      \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            callback_when_condition_fails();                                                                           \
            assert(condition);                                                                                         \
        }
#endif
